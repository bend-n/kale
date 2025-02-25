use super::types::*;
use super::util::*;
use crate::lexer::Token;
use chumsky::{Parser, prelude::*};

#[derive(Debug, Clone)]
pub enum NumberΛ<'s> {
    Number(u64),
    Λ(Spanned<Λ<'s>>),
}

#[derive(Debug, Clone)]
pub enum Function<'s> {
    Both(Λ<'s>),
    And(Spanned<Λ<'s>>, Spanned<Λ<'s>>),
    If { then: Λ<'s>, or: Λ<'s> },
    Array(Option<NumberΛ<'s>>),
    Map(Λ<'s>),
    Dup,
    Flip,
    Eq,
    Reverse,
    Zap,
    Add,
    Sub,
    Not,
    Mul,
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
    Mod,
    Index,
    Mask,
    Group(Λ<'s>),
    Split,
    First,
    Last,
    Reduce(Λ<'s>),
    Range,
    With,
    Call,
    Sort,
    Zip,
    Ident(&'s str),
    Define(&'s str),
}

impl<'s> Λ<'s> {
    pub fn parse(exp: parser![Spanned<Expr<'s>>]) -> parser![Spanned<Self>] {
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
            Token::Flip => Flip,
            // Token::Reverse => Reverse,
            Token::Zap => Zap,
            Token::Add => Add,
            Token::Sub => Sub,
            Token::Mul => Mul,
            Token::Pow => Pow,
            Token::Sqrt => Sqrt,
            Token::Lt => Lt,
            Token::Not => Not,
            Token::Index => Index,
            Token::Merge => Merge,
            Token::Shl => Shl,
            Token::Shr => Shr,
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
            Token::Mask => Mask,
            Token::With => With,
            Token::Split => Split,
            Token::First => First,
            Token::Ne => Ne,
            Token::Type => Type,
            Token::Last => Last,
            Token::Ident(x) => Ident(x),
        }
        .labelled("token");

        let fn_param = choice((
            basic
                .map_with(|x, e| Λ::of(vec![Expr::Function(x).spun(e.span())]))
                .labelled("function"),
            λ.clone(),
        ))
        .labelled("operand");

        macro_rules! one {
            ($name:ident) => {
                fn_param
                    .clone()
                    .then_ignore(just(Token::$name))
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
            two![And],
            one![Both],
            one![Reduce],
            one![Map],
            λ.clone()
                .then_ignore(just(Token::Group).labelled("group"))
                .map(Group),
            choice((
                just(Token::Array).ignore_then(
                    fn_param
                        .clone()
                        .map_with(spanned!())
                        .map(NumberΛ::Λ)
                        .or(select! { Token::Int(x) => NumberΛ::Number(x) })
                        .map(Some)
                        .map(Array),
                ),
                just(Token::Array).map(|_| Array(None)),
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
