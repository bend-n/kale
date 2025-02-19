#![feature(iter_intersperse, formatting_options)]

use parser::types::Ast;
mod array;
mod lexer;
mod parser;
mod ui;
fn main() {
    parser::parse_s(
        &std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap(),
        Ast::parse(),
    );
}
