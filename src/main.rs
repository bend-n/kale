#![feature(
    let_chains,
    try_trait_v2,
    if_let_guard,
    iter_intersperse,
    iterator_try_reduce,
    formatting_options,
    iterator_try_collect,
    impl_trait_in_bindings,
    arbitrary_self_types
)]
mod array;
mod exec;
mod lexer;
mod parser;
mod ui;
fn main() {
    let x =
        std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    println!("{x}");
    let y = parser::parse_s(&x, parser::top());
    exec::exec(y, &x);
}
