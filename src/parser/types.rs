use std::collections::HashSet;
use std::fmt::FormattingOptions;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

use beef::lean::Cow;
use chumsky::input::{MappedInput, Stream};
use chumsky::prelude::*;
use umath::FF64;

use crate::exec::Argc;
use crate::lexer::Token;
use crate::parser::util::Spanner;
pub type Span = SimpleSpan<usize>;
pub type Error<'s> = Rich<'s, Token<'s>, Span>;
pub type Input<'s> = MappedInput<
    Token<'s>,
    Span,
    Stream<crate::lexer::Lexer<'s>>,
    fn((Token<'_>, SimpleSpan)) -> (Token<'_>, SimpleSpan),
>;

#[derive(Clone, Default)]
pub struct Λ<'s>(pub Vec<Spanned<Expr<'s>>>, Argc);
impl Hash for Λ<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.1.hash(state);
    }
}
impl PartialOrd for Λ<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Λ<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.cmp(&other.1)
    }
}
impl Eq for Λ<'_> {}
impl PartialEq for Λ<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl<'s> Λ<'s> {
    pub fn of(x: Vec<Spanned<Expr<'s>>>) -> Self {
        let s = Λ::sized(&x);
        Self(x, s)
    }
    pub fn argc(&self) -> Argc {
        self.1
    }
}
impl std::fmt::Debug for Λ<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.0 {
            [] => write!(f, "λ()"),
            [a] => f.write_fmt(format_args!("λ({a:?})")),
            x => {
                if x.len() < 5 {
                    f.write_fmt(format_args!("λ({x:?})"))
                } else {
                    write!(f, "λ")?;
                    f.with_options(
                        *FormattingOptions::new().alternate(true),
                    )
                    .debug_list()
                    .entries(&self.0)
                    .finish()
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum Value<'s> {
    Float(f64),
    Int(u64),
    String(Cow<'s, str>),
    Lambda(Λ<'s>),
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(x) => write!(f, "{x}"),
            Self::Int(x) => write!(f, "{x}"),
            Self::String(x) => {
                write!(f, "\"{x}\"")
            }
            Self::Lambda(s) => s.fmt(f),
        }
    }
}

impl std::fmt::Debug for Expr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function(x) => x.fmt(f),
            Self::Value(x) => x.fmt(f),
        }
    }
}

#[derive(Clone)]
pub enum Expr<'s> {
    Function(super::fun::Function<'s>),
    Value(Value<'s>),
}

#[derive(Clone, Hash)]
pub struct Spanned<T> {
    pub(crate) span: SimpleSpan,
    pub inner: T,
}

impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<T: PartialEq> Eq for Spanned<T> {}
impl<T: std::fmt::Debug> std::fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
impl<T> Spanned<T> {
    fn new(inner: T, span: SimpleSpan) -> Self {
        Self { inner, span }
    }
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> Spanned<T> {
    pub fn map<U>(self, f: impl Fn(T) -> U) -> Spanned<U> {
        Spanned {
            inner: f(self.inner),
            span: self.span,
        }
    }

    pub fn try_map<U, E>(
        self,
        f: impl Fn(T, Span) -> Result<U, E>,
    ) -> Result<Spanned<U>, E> {
        let Self { inner, span } = self;
        f(inner, span).map(|x| x.spun(span))
    }

    pub fn unspan(self) -> T {
        self.inner
    }

    pub fn span(&self) -> SimpleSpan {
        self.span
    }
    pub fn raw(self) -> (T, SimpleSpan) {
        let Spanned { inner, span } = self;
        (inner, span)
    }

    pub fn dummy(inner: T) -> Spanned<T> {
        Spanned {
            inner,
            span: SimpleSpan::new((), 0..0),
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
                    .chain(
                        x.iter()
                            .map(|x| format!("{x:?}"))
                            .intersperse(", ".into()),
                    )
                    .chain([")".to_string()])
                    .reduce(|acc, x| acc + &x)
                    .unwrap()
            ),
            Self::Path(x) => write!(f, "{x}"),
            Self::Unit => write!(f, "()"),
        }
    }
}
