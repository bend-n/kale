pub mod types;
use crate::lexer::{Lexer, Token};
use chumsky::{Parser, input::Stream, prelude::*};
pub mod fun;
pub mod util;
use types::*;
use util::*;

use self::fun::Function;

impl<'s> Value<'s> {
    pub fn parse() -> parser![Spanned<Self>] {
        select! {
            Token::Char(x) => Value::Int(x as _),
            Token::Int(x) => Value::Int(x),
            Token::Float(x) => Value::Float(x),
            Token::String(s) => Value::String(s),
        }
        .map_with(spanned!())
        .labelled("value")
    }
}

impl<'s> Expr<'s> {
    pub fn parse() -> parser![Spanned<Expr<'s>>] {
        recursive(|expr| {
            let inline_expr: parser![Spanned<Expr>] = Value::parse().map(|x| x.map(Expr::Value));

            let λ = Λ::parse(expr.clone());
            choice((
                inline_expr,
                Function::parse(λ.clone().map(Spanned::unspan()))
                    .map(Expr::Function)
                    .map_with(spanned!()),
                λ.map(|x| x.map(|x| Expr::Value(Value::Lambda(x)))),
            ))
            .labelled("expr")
        })
    }
}

pub fn top<'s>() -> parser![Spanned<Λ<'s>>] {
    Expr::parse()
        .repeated()
        .collect()
        .map(Λ)
        .map_with(spanned!())
}

#[test]
fn parse_expr() {
    // parse_s("a ← λ ( +-🍴 )", Expr::parse());
    let src = r#"⏫⏫⏫
    "#;
    println!(
        "{:?}",
        crate::lexer::lex(src).map(|x| x.0).collect::<Vec<_>>()
    );
    parse_s(src, top());
}

pub fn stream(lexer: Lexer<'_>, len: usize) -> types::Input<'_> {
    Stream::from_iter(lexer).map(SimpleSpan::new((), len..len), |x| x)
}

pub fn code<'s>(x: &'s str) -> types::Input<'s> {
    stream(crate::lexer::lex(x), x.len())
}

pub fn parse_s<'s, T: std::fmt::Debug>(x: &'s str, p: parser![T]) -> T {
    match crate::ui::display(p.parse(code(x)).into_result(), x) {
        Ok(x) => dbg!(x),
        Err(()) => panic!(),
    }
}
