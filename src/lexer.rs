#![allow(unexpected_cfgs, rustdoc::invalid_rust_codeblocks)]
use std::sync::LazyLock;

use beef::lean::Cow;
use chumsky::span::{SimpleSpan, Span};
use logos::{Lexer as RealLexer, Logos, SpannedIter};
use regex::Regex;

use crate::exec::Argc;
use crate::parser::types::{Λ, *};
static EMOJI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\p{Emoji}&&[^0-9]]").unwrap());

macro_rules! tokens {
    ($(
        $(#[$attr:ident = $attrval:expr])*
        $z:literal $( | $y:literal)? => $v:ident  $($eq:literal @)? $(_ $expr:tt)?,)+
        //&

        //$(
            //$(#[$fattr:ident = $fattrval:expr])*
            //$fname:ident $expr:tt),* $(,)?
    ) => {
        #[derive(Logos, Debug, PartialEq, Clone)]
        #[logos(skip r"[\n\s]+")]
        #[allow(dead_code)]
        pub enum Token<'strings> {
            #[regex("/[^\n/]+/?", priority = 8)]
            Comment(&'strings str),
            #[regex(r"[0-9]+", |lex| lex.slice().parse().ok())]
            #[regex(r"0[xX][0-9a-fA-F]+", |lex| u64::from_str_radix(&lex.slice()[2..], 16).ok())]
            #[regex(r"0[bB][01]+", |lex| u64::from_str_radix(&lex.slice()[2..], 2).ok())]
            Int(u64),
            #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse().ok())]
            Float(f64),
            #[regex(r#""([^\\"\n])*""#, callback = |lex| Cow::from(&lex.slice()[1..lex.slice().len()-1]), priority = 12)]
            #[regex(r#""[^"]*""#, callback = |lex| Cow::from(lex.slice()[1..lex.slice().len()-1].replace(r"\n", "\n")), priority = 8)]
            String(Cow<'strings, str>),
            #[regex(r"'.'", |lex| lex.slice().as_bytes()[1] as char)]
            Char(char),
            // todo ignore alot
            #[regex(r"[^\s\(\)\[\]\{\}⎬0-9@'\-←→=≢≡+×\|*√<\-¯∧∨⊻÷%]", priority = 7, callback = |lex| {
                EMOJI.is_match(lex.slice())
                  .then_some(logos::Filter::Skip)
                  .unwrap_or(logos::Filter::Emit(lex.slice()))
            })]
            #[regex(r"'[^'0-9][^']+'", priority = 8, callback = |lex| &lex.slice()[1..lex.slice().len() - 1])]
            Ident(&'strings str),
            #[token("[", chr::<'['>)]
            #[token("(", chr::<'('>)]
            #[token("{", chr::<'{'>)]
            OpeningBracket(char),
            #[token("]", chr::<']'>)]
            #[token(")", chr::<')'>)]
            #[token("}", chr::<'}'>)]
            ClosingBracket(char),

            $(
                $(#[$attr = $attrval])*
                $(#[doc = concat!("link",  stringify!($eq), " [`Function::" , stringify!($v), "`]\n")])?
                #[doc=concat!("character: `", $z, "`")]
                #[token($z, priority = 8)]
                $(#[token($y, priority = 8)])?
            $v,)+

            Unknown,
        }
        impl <'s> Function<'s> {
            pub(crate) fn basic() -> crate::parser::util::parser![Self]  {
                use chumsky::Parser;
                chumsky::select! {
                    Token::Zap => Self::Zap(None),
                    Token::ClosingBracket('}') => Self::Setify,
                    Token::Ident(x) => Self::Ident(x),
                    $($(Token::$v =>  {stringify!($eq); Self::$v },)?)+
                }.labelled("token")
            }
        }
        impl std::fmt::Display for Token<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                match self {
                    $(Self::$v => write!(f, $z),)+
                    Self::Unknown => write!(f, "unknown"),
                    Self::Char(x) => write!(f, "'{x}'"),
                    Self::String(s) => write!(f, "{s}"),
                    Self::Float(n) => write!(f, "{n}"),
                    Self::Int(n) => write!(f, "{n}"),
                    Self::OpeningBracket(x) | Self::ClosingBracket(x) => write!(f,"{x}"),
                    Self::Comment(_) => write!(f, ""),
                    Self::Ident(x) => write!(f, "{x}"),
                }
            }
        }

#[derive(Debug, Clone)]
pub enum Function<'s> {
    #[doc(hidden)]
    Setify,
    #[allow(dead_code)]
    #[doc(hidden)]
    If { then: Λ<'s>, or: Λ<'s> },
    #[doc(hidden)]
    Ident(&'s str),
    #[doc(hidden)]
    Define(&'s str),
    // $($(#[$fattr = $fattrval])* $fname $expr,)*
    $(
        $(#[$attr = $attrval])*
        $(#[doc = concat!("<h1>", $z, "</h1>")] #[cfg(not(target_family = $eq))] $v,)?
        $(#[doc = concat!("<h1>", $z, "</h1>")] $v $expr,)?
    )+
    /// [n = drops n items.
    ///
    /// <h1>[</h1>
    Take(u64),
}

    }
}

tokens! {
    "λ" => Lambda,
    /// Create an array from n items off the stack:
    /// ```
    /// 1 2 3 4 ⎦4
    /// ```
    /// Produces an array of [1, 2, 3, 4]
    "⎦" => Array _ (Option<u64>),
    "→" => Place,

    /// Array contains.
    "∈" => In "0" @,


    /// Duplicates the top value of the stack.
    "^" => Dup "0" @,
    /// Runs n functions on the same values in the stack, pulling y items where y is the maximum arguments of any of the functions.
    "&" => And _ (Vec<Spanned<Λ<'s>>>),
    /// Runs one function n times, with new stack items every time, where n is the number of instances of this symbol.
    /// ```kale
    /// 1 2 3 4 +|
    /// / results in 7 3
    /// ```
    "|" => Both _ (Spanned<Λ<'s>>, usize),
    /// Flips the top two values on the stack.
    "🔀" => Flip "0" @,
    /// Zaps the top value on the stack, or the nth value, if followed by a number.
    "⤵️" | "⤵" => Zap _ (Option<u64>),
    /// Pops an array off of the stack to use it as a stack.
    "⬇️" | "⬇" => With _ (Spanned<Λ<'s>>),
    /// Pops a number, creates an array from 0-n
    "⏫" => Range "0" @,
    // "🪪" => Type "0" @,
    /// Pops an array, gets the length.
    "📏" => Length "0" @,
    /// Groups an array by a mask, grouping by the ones in the mask array.
    /// ```kale
    /// 5⏫^2≢👩‍👩‍👧‍👧
    /// ```
    /// Would split by two, producing [0, 1] and [3, 4].
    "👩‍👩‍👧‍👧" => Group "0" @,
    /// Pops a string, opens that file, producing an array of utf8 integers.
    "📂" => Open "0" @,
    "⏪" => Shl "0" @,
    "⏩" => Shr "0" @,
    /// Takes an array, index, removes the element in the array at that index.
    "❎" => Del "0" @,
    /// Sorts an array.
    "📶" => Sort "0" @,
    /// Given an array, and a mask, masks the array by the mask.
    /// In other words, picks only the values for which are one in the mask.
    /// ```
    /// 10⏫^5<
    /// 🔓
    /// ```
    /// Produces [0, 1, 2, 3, 4],
    "🔓" => Mask "0" @,
    /// Given an array of indexes, and an array, indexes the array by those indices.
    "🔒" => Index "0" @,
    "#️⃣🗺" => HashMap "0" @,
    "≣#️⃣" => IndexHashMap "0" @,
    "∅" => EmptySet "0" @,
    /// Places a value in an array.
    "💽" => Append "0" @,
    /// Get first item in array.
    "⬅️" | "⬅" => First "0" @,
    /// Get last item in array.
    "➡️" | "➡" => Last "0" @,
    /// Reduce an array by a function.
    "↘️" | "↘" => Reduce _ (Spanned<Λ<'s>>),
    /// Scan an array by a function. So it reduces, keeping the product.
    "↖️" | "↖" => Scan _ (Spanned<Λ<'s>>),
    /// Fold an array by a function (note that the accumulator is taken after the array).
    "⏭️" | "⏭" => Fold _ (Spanned<Λ<'s>>),
    /// Map an array by a function.
    "🗺" => Map _ (Spanned<Λ<'s>>),
    "🐋" => If,
    "🐬" => EagerIf,
    /// Zip two arrays together.
    "🇳🇿" => Zip "0" @,
    /// Get the windows of an array.
    "🪟" => Windows "0" @,
    /// Debugs the stack at the current point in time.
    "🧐" => Debug "0" @,
    /// Takes one value, pushes one value.
    "." => Identity "0" @,
    /// Invoke python. Requires specifying the signature. In python land, the executor will be able to access a `s` variable, containing the stack. If the python does not satisfy the signature, an error will occur.
    /// ```
    /// 5 "s.append(s.pop()+1)"🐍1 → 1
    /// ```
    /// Produces six.
    "🐍" => Python _ (Argc),


    "≡" => Eq "0" @,
    "≣" => Matches "0" @,
    "≢" => Ne "0" @,
    "+" => Add "0" @,
    "-" => Sub "0" @,
    "×" => Mul "0" @,
    "ⁿ" => Pow "0" @,
    "<" => Lt "0" @,
    ">" => Gt "0" @,
    "≤" => Le "0" @,
    "≥" => Ge "0" @,
    "÷" => Div "0" @,
    "%" => Mod "0" @,
    "∧" => BitAnd "0" @,
    "∨" => Or "0" @,
    "⊕" => Xor "0" @,
    "!" => Not "0" @,
    "¯" => Neg "0" @,
    "√" => Sqrt "0" @,
}

pub fn lex(s: &str) -> Lexer<'_> {
    Lexer {
        inner: Token::lexer(s).spanned(),
    }
}

fn chr<'src, const CHR: char>(
    _: &mut RealLexer<'src, Token<'src>>,
) -> Result<char, ()> {
    Ok(CHR)
}
pub struct Lexer<'s> {
    inner: SpannedIter<'s, Token<'s>>,
}

impl<'s> Iterator for Lexer<'s> {
    type Item = (Token<'s>, SimpleSpan<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .find_map(|(x, s)| match x.unwrap_or(Token::Unknown) {
                Token::Comment(_) => None,
                x => Some((x, SimpleSpan::new((), s))),
            })
    }
}

#[test]
fn lexer() {
    let lex = lex(r#""1abc25hriwm4"
    / { str → int } /
    line ← "0" @(
        '0'>🔎'9'<🔎
        '9'-
        / modifiers are placed in front /
        🐘⬅➡
        10×+
    )
    
    / if true { + } else { - } /"#);
    // while let Some((x, _)) = lex.next() {
    //     print!("{x} ");
    // }
    macro_rules! test {
        ($($tok:ident$(($var:literal))?)+) => {{
            $(assert_eq!(lex.next().map(|(x,_)|x), Some(Token::$tok$(($var.into()))?));)+
            assert_eq!(lex.next(), None);
        }}
    }
}
