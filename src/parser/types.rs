use std::ops::Deref;

use crate::lexer::Token;
use beef::lean::Cow;
use chumsky::{
    input::{SpannedInput, Stream},
    prelude::*,
};
use match_deref::match_deref;
pub type Span = SimpleSpan<usize>;
pub type Error<'s> = Rich<'s, Token<'s>, Span>;
pub type Input<'s> = SpannedInput<Token<'s>, SimpleSpan, Stream<crate::lexer::Lexer<'s>>>;

pub enum Ast<'s> {
    Module(Vec<Expr<'s>>),
}

#[derive(Clone)]
pub enum Value<'s> {
    Float(f64),
    Int(u64),
    String(Cow<'s, str>),
    Unit,
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(x) => write!(f, "{x}f"),
            Self::Int(x) => write!(f, "{x}i"),
            Self::String(x) => write!(f, "\"{x}\""),
            Self::Unit => write!(f, "()"),
        }
    }
}

#[derive(Clone)]
pub enum Expr<'s> {
    NoOp,
    Value(Value<'s>),
    Ident(&'s str),
    Let {
        name: &'s str,
        rhs: Box<Expr<'s>>,
    },
    If {
        then: Box<Expr<'s>>,
        or: Box<Expr<'s>>,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub inner: T,
    pub span: Span,
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> Spanned<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            inner: f(self.inner),
            span: self.span,
        }
    }

    pub fn dummy(inner: T) -> Spanned<T> {
        Spanned {
            inner,
            span: SimpleSpan::new(0, 0),
        }
    }

    pub fn copys<U>(&self, with: U) -> Spanned<U> {
        Spanned {
            inner: with,
            span: self.span,
        }
    }
}

impl<T> From<(T, Span)> for Spanned<T> {
    fn from((inner, span): (T, Span)) -> Self {
        Self { inner, span }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[derive(Clone)]
pub enum Type<'s> {
    Tuple(Box<[Type<'s>]>),
    Path(&'s str),
    Unit,
}

impl std::fmt::Debug for Type<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tuple(x) => write!(
                f,
                "{}",
                std::iter::once("(".to_string())
                    .chain(x.iter().map(|x| format!("{x:?}")).intersperse(", ".into()),)
                    .chain([")".to_string()])
                    .reduce(|acc, x| acc + &x)
                    .unwrap()
            ),
            Self::Path(x) => write!(f, "{x}"),
            Self::Unit => write!(f, "()"),
        }
    }
}
