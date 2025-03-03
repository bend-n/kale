use super::types::*;

macro_rules! t {
    (ident) => {
        select! { Token::Ident(ident) => ident }.labelled("ident")
    };
    (if) => {
        just(Token::If)
    };
    (else) => {
        just(Token::Else)
    };
    (=) => {
        just(Token::Equal)
    };
    (λ) => {
        just(Token::Lambda)
    };
    (<-) => {
        just(Token::Place)
    };
    (,) => {
        just(Token::Comma)
    };
    (:) => {
        just(Token::Colon)
    };
    (->) => {
        just(Token::Place)
    };
    (()) => {
        just(Token::Call)
    };
    ('(') => {
        just(Token::OpeningBracket('('))
    };
    (')') => {
        just(Token::ClosingBracket(')'))
    };
    ('[') => {
        just(Token::OpeningBracket('['))
    };
    (']') => {
        just(Token::ClosingBracket(']'))
    };
    ('{') => {
        just(Token::OpeningBracket('{'))
    };
    ('}') => {
        just(Token::ClosingBracket('}'))
    };
    (int) => {
        select! { Token::Int(x) => x }
    };
}
macro_rules! parser {
    ($t:ty) => {
        impl Parser<'s, crate::parser::types::Input<'s>, $t, extra::Err<Error<'s>>> + Clone + 's
    }
}

pub trait TakeSpan {
    fn tspn<T>(&mut self, x: T) -> Spanned<T>;
}

impl<'a, 'b> TakeSpan
    for MapExtra<'a, 'b, Input<'a>, chumsky::extra::Err<Error<'a>>>
{
    fn tspn<T>(&mut self, x: T) -> Spanned<T> {
        Spanned::from((x, self.span()))
    }
}
macro_rules! spanned {
    () => {
        |a, extra| Spanned::from((a, extra.span()))
    };
}

use chumsky::input::MapExtra;
pub(crate) use {parser, spanned, t};

pub trait Unit<T> {
    fn empty(&self) -> T;
}

impl<T> Unit<Option<()>> for Option<T> {
    fn empty(&self) -> Option<()> {
        self.as_ref().map(|_| ())
    }
}

pub trait Spanner {
    fn spun(self, s: Span) -> Spanned<Self>
    where
        Self: Sized,
    {
        (self, s).into()
    }
}
impl<T> Spanner for T {}

pub trait MapLeft<T, V> {
    fn ml<U>(self, f: impl Fn(T) -> U) -> (U, V);
}

impl<T, V> MapLeft<T, V> for (T, V) {
    fn ml<U>(self, f: impl Fn(T) -> U) -> (U, V) {
        (f(self.0), self.1)
    }
}
