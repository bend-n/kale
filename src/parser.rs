pub mod types;
use crate::lexer::{Lexer, Token};
use chumsky::{
    input::{SpannedInput, Stream},
    prelude::*,
    Parser,
};
pub mod fun;
pub mod util;
use types::*;
use util::*;

use self::fun::Function;

impl<'s> Value<'s> {
    pub fn parse() -> parser![Self] {
        select! {
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
            let inline_expr = recursive(|inline_expr| {
                let val = select! {
                    Token::Int(x) => Expr::Value(Value::Int(x)),
                    Token::Float(x) => Expr::Value(Value::Float(x)),
                    Token::String(s) => Expr::Value(Value::String(s)),
                }
                .labelled("value");

                choice((t![ident].map(Expr::Ident), val)).boxed()
            });

            let λ = Λ::parse(expr.clone());

            let decl = t![ident]
                .then_ignore(t![<-])
                .then(inline_expr.clone().or(λ.clone().map(Expr::Lambda)))
                .map(|(name, body)| Expr::Let {
                    name,
                    rhs: Box::new(body),
                })
                .labelled("declare")
                .boxed();

            let r#if = t![if]
                .ignore_then(
                    expr.clone()
                        .then(t![else].or_not().ignore_then(expr.clone().or_not()))
                        .delimited_by(t!['('], t![')']),
                )
                .map(|(a, b)| Expr::If {
                    then: Box::new(a),
                    or: Box::new(b.unwrap_or_else(|| Expr::Value(Value::Unit))),
                })
                .labelled("🐋")
                .boxed();
            choice((
                decl,
                r#if,
                inline_expr,
                λ.clone().map(Expr::Lambda),
                Function::parse(λ).map(Expr::Function),
            ))
            .labelled("expr")
        })
    }
}

#[test]
fn parse_expr() {
    parse_s("a ← λ ( +- 🍴 )", Expr::parse());
    // dbg!(Expr::parse().parse(code("a ← λ ( +- 🍴 )")).unwrap());
}

pub fn stream(lexer: Lexer<'_>, len: usize) -> SpannedInput<Token<'_>, Span, Stream<Lexer<'_>>> {
    Stream::from_iter(lexer).spanned((len..len).into())
}

#[cfg(test)]
pub fn code<'s>(x: &'s str) -> SpannedInput<Token<'s>, Span, Stream<Lexer<'s>>> {
    stream(crate::lexer::lex(x), x.len())
}

#[cfg(test)]
pub fn parse_s<'s, T: std::fmt::Debug>(x: &'s str, p: parser![T]) -> T {
    match crate::ui::display(p.parse(code(x)).into_result(), x) {
        Ok(x) => dbg!(x),
        Err(()) => panic!(),
    }
}

pub fn parse(tokens: Lexer<'_>, len: usize) -> Result<Ast<'_>, Vec<Error<'_>>> {
    parser().parse(stream(tokens, len)).into_result()
}

fn parser<'s>() -> parser![Ast<'s>] {
    Expr::parse().repeated().collect().map(Ast::Module)
}
