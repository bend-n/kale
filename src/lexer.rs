use beef::lean::Cow;
use chumsky::span::SimpleSpan;
use logos::{Lexer as RealLexer, Logos, SpannedIter};

macro_rules! tokens {
    ($($z:literal $( | $y:literal)? => $v:ident,)+) => {
        #[derive(Logos, Debug, PartialEq, Clone)]
        #[logos(skip r"[\n\s]+")]
        #[allow(dead_code)]
        pub enum Token<'strings> {
            #[regex("/[^\n/]+/", priority = 8)]
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
            // todo ignore alot
            #[regex(r"[^\s0-9][^\s]*", priority = 7)]
            Ident(&'strings str),

            #[token("[", chr::<'['>)]
            #[token("(", chr::<'('>)]
            #[token("{", chr::<'{'>)]
            OpeningBracket(char),
            #[token("]", chr::<']'>)]
            #[token(")", chr::<')'>)]
            #[token("}", chr::<'}'>)]
            ClosingBracket(char),

            $(#[token($z, priority = 8)] $(#[token($y, priority = 8)])? $v,)+
        }

        impl std::fmt::Display for Token<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                match self {
                    $(Self::$v => write!(f, $z),)+
                    Self::String(s) => write!(f, "{s}"),
                    Self::Float(n) => write!(f, "{n}"),
                    Self::Int(n) => write!(f, "{n}"),
                    Self::OpeningBracket(x) | Self::ClosingBracket(x) => write!(f,"{x}"),
                    Self::Comment(_) => write!(f, ""),
                    Self::Ident(x) => write!(f, "{x}"),
                }
            }
        }
    }
}

tokens! {
    "λ" => Lambda,
    "←" => Place,
    "→" => Ret,
    "=" => Eq,
    "🐢" => Dup,
    "🐘" => Both,
    "🍴" => Fork,
    "🐈" => Flip,
    "↖" => Reverse,
    "⤵️" => Pop,
    "+" => Add,
    "×" => Mul,
    "*" => Pow,
    "√" => Sqrt,
    "≠" => Ne,
    "<" => Lt,
    "≤" | "≯" => Le,
    ">" => Gt,
    "≥" | "≮" => Ge,
    "⏪" => Shl,
    "⏩" => Shr,
    "¯" => Neg,
    "∧" => And,
    "∨" => Or,
    "⊻" => Xor,
    "÷" => Div,
    "%" => Mod,
    "🔎" => Keep,
    "🚧" => Split,
    "⬅" => First,
    "➡" => Last,
    "⏭️" => Each,
    "➡️" => Reduce,
    "↘️" => ReduceStack,
    "🐋" => If,
    "🐳" => Else,

}

pub fn lex(s: &str) -> Lexer {
    Lexer {
        inner: Token::lexer(s).spanned(),
    }
}

fn chr<'src, const CHR: char>(_: &mut RealLexer<'src, Token<'src>>) -> Result<char, ()> {
    Ok(CHR)
}
pub struct Lexer<'s> {
    inner: SpannedIter<'s, Token<'s>>,
}

impl<'s> Iterator for Lexer<'s> {
    type Item = (Token<'s>, SimpleSpan<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|(x, s)| match x.ok()? {
            Token::Comment(_) => None,
            x => Some((x, SimpleSpan::new(s.start, s.end))),
        })
    }
}

#[test]
fn lexer() {
    let mut lex = lex(r#""1abc25hriwm4"
    / { str → int } /
    line ← λ (
        '0'>🔎'9'<🔎
        '9'-
        / modifiers are placed in front /
        🐘⬅➡
        10×+
    )
    
    🐢≠'\n'🚧
    / run function on all values, pushing to the stack /
    ⏭️line
    / reduce the stack /
    ↘️+
    
    true 🐋 (+ 🐳 -)
    / if true { + } else { - } /"#);
    while let Some((x, _)) = lex.next() {
        print!("{x:?} ");
    }
    macro_rules! test {
        ($($tok:ident$(($var:literal))?)+) => {{
            $(assert_eq!(lex.next().map(|(x,_)|x), Some(Token::$tok$(($var.into()))?));)+
            assert_eq!(lex.next(), None);
        }}
    }
}
