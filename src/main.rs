#![feature(
    try_trait_v2,
    iter_intersperse,
    iterator_try_reduce,
    formatting_options,
    impl_trait_in_bindings,
    arbitrary_self_types
)]
mod array;
mod exec;
mod lexer;
mod parser;
mod ui;
fn main() {
    let x = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    let y = parser::parse_s(&x, parser::top());
    exec::exec(y, &x);
}
