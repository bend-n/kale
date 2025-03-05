use chumsky::Parser;
use chumsky::prelude::*;

use super::types::*;
use super::util::*;
use crate::exec::Argc;
use crate::lexer::Token;

#[derive(Debug, Clone)]
pub enum Function<'s> {
    Both(Spanned<Λ<'s>>, usize),
    And(Vec<Spanned<Λ<'s>>>),
    Take(u64),
    If { then: Λ<'s>, or: Λ<'s> },
    Array(Option<u64>),
    Append,
    Map(Spanned<Λ<'s>>),
    Dup,
    Flip,
    Python(Argc),
    Matches,
    Eq,
    Reverse,
    Zap(Option<u64>),
    Del,
    Debug,
    Add,
    Sub,
    IndexHashMap,
    Not,
    Mul,
    Windows,
    Pow,
    Type,
    Ne,
    Merge,
    Sqrt,
    Lt,
    Gt,
    Ge,
    Le,
    Shl,
    Shr,
    Neg,
    BitAnd,
    Length,
    Or,
    Xor,
    Div,
    Fold(Spanned<Λ<'s>>),
    Mod,
    Index,
    Mask,
    Group,
    Split,
    Open,
    First,
    Last,
    Reduce(Spanned<Λ<'s>>),
    Range,
    With(Spanned<Λ<'s>>),
    HashMap,
    Call,
    Sort,
    Zip,
    Identity,
    EmptySet,
    Setify,
    Ident(&'s str),
    Define(&'s str),
}

impl<'s> Λ<'s> {
    pub fn parse(
        exp: parser![Spanned<Expr<'s>>],
    ) -> parser![Spanned<Self>] {
        exp.repeated()
            .collect()
            .delimited_by(t!['('], t![')'])
            .map_with(|x, e| Spanned::from((Self::of(x), e.span())))
            .labelled("lambda")
    }
}

impl<'s> Function<'s> {
    pub fn parse(λ: parser![Λ<'s>]) -> parser![Self] {
        use Function::*;
        let basic = select! {
            Token::Dup => Dup,
            Token::Debug => Debug,
            Token::Flip => Flip,
            // Token::Reverse => Reverse,
            Token::Zap => Zap(None),
            Token::Add => Add,
            Token::ClosingBracket('}') => Setify,
            Token::Set => EmptySet,
            Token::Identity => Identity,
            Token::Del => Del,
            Token::HashMap => HashMap,
            Token::Get => IndexHashMap,
            Token::Sub => Sub,
            Token::Windows => Windows,
            Token::Mul => Mul,
            Token::Pow => Pow,
            Token::Sqrt => Sqrt,
            Token::Lt => Lt,
            Token::Not => Not,
            Token::Index => Index,
            Token::Merge => Merge,
            Token::Shl => Shl,
            Token::Group => Group,
            Token::Shr => Shr,
            Token::Append => Append,
            Token::Neg => Neg,
            Token::Eq => Eq,
            Token::Gt => Gt,
            Token::Ge => Ge,
            Token::Length => Length,
            Token::Range => Range,
            Token::Le => Le,
            Token::BitAnd => BitAnd,
            Token::Or => Or,
            Token::Xor => Xor,
            Token::Sort => Sort,
            Token::Zip => Zip,
            Token::Div => Div,
            Token::Mod => Mod,
            Token::Open => Open,
            Token::Mask => Mask,
            Token::First => First,
            Token::Ne => Ne,
            Token::Type => Type,
            Token::Matches => Matches,
            Token::Last => Last,
            Token::Ident(x) => Ident(x),
        }
        .labelled("token");

        let fn_param = choice((
            basic
                .map_with(|x, e| {
                    Λ::of(vec![Expr::Function(x).spun(e.span())])
                })
                .labelled("function"),
            λ.clone(),
        ))
        .labelled("operand");

        macro_rules! one {
            ($name:ident) => {
                fn_param
                    .clone()
                    .then_ignore(just(Token::$name))
                    .map_with(spanned!())
                    .map($name)
                    .labelled(stringify!($name))
            };
        }
        macro_rules! two {
            ($name:ident) => {
                fn_param
                    .clone()
                    .map_with(spanned!())
                    .then(fn_param.clone().map_with(spanned!()))
                    .then_ignore(just(Token::$name))
                    .map(|(a, b)| $name(a, b))
                    .labelled(stringify!($name))
            };
        }
        choice((
            λ.clone()
                .map_with(spanned!())
                .then(
                    fn_param
                        .clone()
                        .map_with(spanned!())
                        .repeated()
                        .at_least(1)
                        .collect::<Vec<_>>(),
                )
                .then_ignore(just(Token::And))
                .map(|(a, mut b)| {
                    b.insert(0, a);
                    And(b)
                })
                .boxed(),
            fn_param
                .clone()
                .map_with(spanned!())
                .then(
                    just(Token::Both)
                        .repeated()
                        .at_least(1)
                        .count()
                        .map(|x| x + 1),
                )
                .map(|(a, b)| Both(a, b))
                .labelled("both"),
            one![Reduce],
            one![Fold],
            one![Map],
            one![With],
            just(Token::Zap).ignore_then(t![int]).map(Some).map(Zap),
            t!['['].ignore_then(t![int]).map(Take),
            just(Token::Python)
                .ignore_then(t![int])
                .then_ignore(t![->])
                .then(t![int])
                .map(|(a, b)| Python(Argc::takes(a as _).into(b as _))),
            choice((
                just(Token::ArrayN)
                    .ignore_then(t![int].map(|x| Array(Some(x)))),
                t![']'].map(|_| Array(None)),
            ))
            .labelled("array")
            .boxed(),
            fn_param
                .clone()
                .then(fn_param.clone())
                .then_ignore(just(Token::If))
                .map(|(then, or)| If { then, or })
                .labelled("if-else")
                .boxed(),
            fn_param
                .clone()
                .then_ignore(just(Token::EagerIf).labelled("if"))
                .map(|then| If {
                    then,
                    or: Λ::default(),
                })
                .labelled("if")
                .boxed(),
            t![->].ignore_then(t![ident]).map(Define).labelled("def"),
            basic,
        ))
        .boxed()
        .labelled("function")
    }
}
