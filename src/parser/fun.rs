use super::types::*;
use super::util::*;
use crate::lexer::Token;
use chumsky::{prelude::*, Parser};

#[derive(Debug, Clone)]
pub enum Function<'s> {
    Dup,
    Both(Λ<'s>, Λ<'s>),
    Fork(Λ<'s>, Λ<'s>),
    Gap(Λ<'s>),
    Hold(Λ<'s>),
    Flip,
    Duck(Λ<'s>),
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
    Each(Λ<'s>),
    Reduce(Λ<'s>),
    ReduceStack(Λ<'s>),
    Range,
    Call,
}

impl<'s> Λ<'s> {
    pub fn parse(exp: parser![Expr<'s>]) -> parser![Self] {
        let mut λ = Recursive::declare();
        λ.define(choice((
            t![λ]
                .ignore_then(exp.repeated().collect().delimited_by(t!['('], t![')']))
                .map(|x| Self(x)),
            Function::parse(λ.clone()).map(|x| Λ(vec![Expr::Function(x)])),
        )));
        λ.labelled("λ")
    }
}

impl<'s> Function<'s> {
    pub fn parse(λ: parser![Λ<'s>]) -> parser![Self] {
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
            ($name:ident) => {{
                let mut p = Recursive::declare();
                p.define(
                    λ.clone()
                        .then(λ.clone())
                        .then_ignore(just(Token::$name))
                        .map(|(a, b)| $name(a, b)),
                );
                p
            }};
        }
        choice((basic, two![Both], two![Fork]))
    }
}
