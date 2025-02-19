pub mod types;
use crate::lexer::{Lexer, Token};
use chumsky::{input::Stream, prelude::*, Parser};
pub mod fun;
pub mod util;
use types::*;
use util::*;

use self::fun::Function;

impl<'s> Value<'s> {
    pub fn parse() -> parser![Self] {
        select! {
            Token::Char(x) => Value::Int(x as _),
            Token::Int(x) => Value::Int(x),
            Token::Float(x) => Value::Float(x),
            Token::String(s) => Value::String(s),
        }
        .labelled("value")
    }
}

impl<'s> Expr<'s> {
    pub fn parse() -> parser![Self] {
        recursive::<_, Expr, _, _, _>(|expr| {
            let inline_expr = Value::parse().map(Expr::Value);

            let λ = Λ::parse(expr.clone());

            choice((
                inline_expr,
                Function::parse(λ.clone()).map(Expr::Function),
                λ.map(Expr::Lambda),
            ))
            .labelled("expr")
        })
    }
}
impl<'s> Ast<'s> {
    pub fn parse() -> parser![Self] {
        Expr::parse().repeated().collect().map(Ast::Module)
    }
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
    parse_s(src, Ast::parse());
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
