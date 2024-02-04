use super::types::*;
use super::util::*;
use crate::lexer::Token;
use chumsky::{prelude::*, Parser};

#[derive(Debug, Clone)]
pub enum Function<'s> {
    Dup,
    Both(Lambda<'s>, Lambda<'s>),
    Fork(Lambda<'s>, Lambda<'s>),
    Gap(Lambda<'s>),
    Hold(Lambda<'s>),
    Flip,
    Duck(Lambda<'s>),
    Reverse,
    Zap,
    Add,
    Sub,
    Mul,
    Pow,
    Sqrt,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Neg,
    And,
    Or,
    Xor,
    Div,
    Mod,
    Keep,
    Split,
    First,
    Last,
    Each(Lambda<'s>),
    Reduce(Lambda<'s>),
    ReduceStack(Lambda<'s>),
    Range,
    Call,
}

impl<'s> Lambda<'s> {
    pub fn parse() -> parser![Self] {
        t![λ]
            .ignore_then(
                Expr::parse()
                    .repeated()
                    .collect()
                    .delimited_by(t!['('], t![')']),
            )
            .map(|x| Self(x))
    }
}

impl<'s> Function<'s> {
    pub fn parse() -> parser![Self] {
        use Function::*;
        let basic = select! {
            Token::Dup => Dup,
            Token::Flip => Flip,
            Token::Reverse => Reverse,
            Token::Zap => Zap,
            Token::Add => Add,
            Token::Sub => Sub,
            Token::Mul => Mul,
            Token::Pow => Pow,
            Token::Sqrt => Sqrt,
            Token::Ne => Ne,
            Token::Lt => Lt,
            Token::Le => Le,
            Token::Gt => Gt,
            Token::Ge => Ge,
            Token::Shl => Shl,
            Token::Shr => Shr,
            Token::Neg => Neg,
            Token::And => And,
            Token::Or => Or,
            Token::Xor => Xor,
            Token::Div => Div,
            Token::Mod => Mod,
            Token::Keep => Keep,
            Token::Split => Split,
            Token::First => First,
            Token::Last => Last,
        };
        macro_rules! two {
            ($name:ident) => {
                Lambda::parse()
                    .then(Lambda::parse())
                    .then_ignore(just(Token::$name))
                    .map(|(a, b)| $name(a, b))
            };
        }
        choice((basic, two![Both], two![Fork]))
    }
}
