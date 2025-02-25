use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
    mem::take,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use chumsky::span::SimpleSpan;

use crate::parser::{
    fun::{Function, NumberΛ},
    types::*,
    util::Spanner,
};
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Argc {
    input: usize,
    output: usize,
}
impl std::fmt::Debug for Argc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.input, self.output)
    }
}

impl Argc {
    pub fn takes(input: usize) -> Self {
        Self { input, output: 0 }
    }
    pub fn produces(output: usize) -> Self {
        Self { input: 0, output }
    }
    pub fn into(self, output: usize) -> Self {
        Self { output, ..self }
    }
}

#[derive(Clone)]
pub enum Val<'s> {
    Array(Vec<Val<'s>>),
    Lambda(Λ<'s>),
    Int(i128),
    Float(f64),
}

impl Val<'_> {
    fn ty(&self) -> &'static str {
        match self {
            Self::Array(_) => "array",
            Self::Float(_) => "float",
            Self::Int(_) => "int",
            Self::Lambda(..) => "lambda",
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConcreteVal {
    Array(Vec<ConcreteVal>),
    Int(i128),
    Float(f64),
}
impl From<f64> for Val<'_> {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<i128> for Val<'_> {
    fn from(value: i128) -> Self {
        Self::Int(value)
    }
}

impl From<bool> for Val<'_> {
    fn from(value: bool) -> Self {
        Self::Int(value as i128)
    }
}

impl<'a> From<Vec<Val<'a>>> for Val<'a> {
    fn from(value: Vec<Val<'a>>) -> Self {
        Self::Array(value)
    }
}

impl ConcreteVal {
    fn val(self) -> Val<'static> {
        match self {
            ConcreteVal::Array(x) => {
                Val::Array(x.into_iter().map(ConcreteVal::val).collect::<Vec<_>>())
            }
            ConcreteVal::Int(x) => Val::Int(x),
            ConcreteVal::Float(x) => Val::Float(x),
        }
    }
}

impl<'s> Val<'s> {
    pub fn concrete(self: Spanned<Self>, user: SimpleSpan) -> Result<Spanned<ConcreteVal>, Error> {
        let (x, span) = self.raw();
        Ok(match x {
            Val::Array(x) => ConcreteVal::Array(
                x.into_iter()
                    .map(|x| x.spun(span).concrete(user).map(|x| x.inner))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Val::Float(x) => ConcreteVal::Float(x),
            Val::Int(x) => ConcreteVal::Int(x),
            Val::Lambda(..) => {
                return Err(Error {
                    name: "value not concrete (λ)".into(),
                    message: "concrete value required here".to_string().spun(user),
                    labels: vec!["created here".to_string().spun(span)],
                    notes: vec![],
                });
            }
        }
        .spun(span))
    }
}

impl std::fmt::Debug for Val<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(x) => x.fmt(f),
            Self::Lambda(x) => x.fmt(f),
            Self::Int(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
        }
    }
}

pub struct Context<'s, 'v> {
    pub inherits: Option<&'v Context<'s, 'v>>,
    pub variables: HashMap<&'s str, Spanned<Val<'s>>>,
}

impl<'s, 'v> Default for Context<'s, 'v> {
    fn default() -> Self {
        Self {
            inherits: None,
            variables: Default::default(),
        }
    }
}

impl<'s, 'v> Context<'s, 'v> {
    fn inherits(x: &'v Context<'s, 'v>) -> Self {
        Self {
            inherits: Some(x),
            variables: Default::default(),
        }
    }
}
pub fn exec(x: Spanned<Λ<'_>>, code: &str) {
    crate::ui::display_execution(
        exec_lambda(x, &mut Context::default(), &mut Stack::new()),
        code,
    );
}
impl std::fmt::Debug for Stack<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:#?}", self.cur())
    }
}

#[derive(Clone)]
pub struct Stack<'s>(Vec<Vec<Spanned<Val<'s>>>>);
impl<'s> Stack<'s> {
    fn new() -> Self {
        Self(Vec::from([Vec::with_capacity(200)]))
    }
    pub fn push(&mut self, x: Spanned<Val<'s>>) {
        self.curr().push(x);
    }
    #[track_caller]
    pub fn pop(&mut self) -> Option<Spanned<Val<'s>>> {
        self.curr().pop()
    }
    pub fn curr(&mut self) -> &mut Vec<Spanned<Val<'s>>> {
        self.0.last_mut().unwrap()
    }
    pub fn cur(&self) -> &[Spanned<Val<'s>>] {
        self.0.last().unwrap()
    }
}

#[derive(Debug)]
pub struct Error {
    pub name: String,
    pub message: Spanned<String>,
    pub labels: Vec<Spanned<String>>,
    pub notes: Vec<String>,
}

impl Error {
    pub fn stack_empty(span: SimpleSpan) -> Self {
        Error {
            name: "stack empty".into(),
            message: "empty stack".to_string().spun(span),
            labels: vec![],
            notes: vec![],
        }
    }

    pub fn ef(span: SimpleSpan, expected: impl Display, found: Spanned<impl Display>) -> Self {
        Error {
            name: "type mismatch".to_string(),
            labels: vec![format!("found {found}, not an {expected}").spun(found.span())],
            message: format!("expected {expected} found {found}").spun(span),
            notes: vec![],
        }
    }
}
trait Annotate {
    fn label(self, message: Spanned<impl Into<String>>) -> Self;
    fn note(self, message: impl Into<String>) -> Self;
}
impl Annotate for Error {
    fn label(mut self, message: Spanned<impl Into<String>>) -> Self {
        self.labels.push(message.map(|x| x.into()));
        self
    }
    fn note(mut self, message: impl Into<String>) -> Self {
        self.notes.push(message.into());
        self
    }
}

impl<T> Annotate for Result<T, Error> {
    fn label(self, message: Spanned<impl Into<String>>) -> Self {
        self.map_err(|x| x.label(message))
    }
    fn note(self, message: impl Into<String>) -> Self {
        self.map_err(|x| x.note(message))
    }
}

#[test]
fn x() {
    assert!(
        crate::parser::parse_s("5 + 1 2 ×", crate::parser::top()).argc() == Argc::takes(1).into(2)
    );
    assert!(crate::parser::parse_s("¯", crate::parser::top()).argc() == Argc::takes(1).into(1));
}

impl Add<Argc> for Argc {
    type Output = Self;
    fn add(mut self, rhs: Argc) -> Self {
        match self.output.checked_sub(rhs.input) {
            Some(x) => self.output = x + rhs.output,
            // borrow inputs
            None => {
                self.input += rhs.input - self.output;
                self.output = rhs.output;
            }
        }
        self
    }
}

fn size_fn<'s>(f: &Function<'s>) -> Argc {
    use Function::*;
    match f {
        Add | Mul | Div | Xor | Mod | Pow | Eq | Ne | BitAnd | Or => Argc::takes(2).into(1),
        Neg | Sqrt | Not => Argc::takes(1).into(1),
        _ => Argc {
            input: 0,
            output: 0,
        },
    }
}

fn size_expr<'s>(x: &Expr<'s>) -> Argc {
    match x {
        Expr::Function(function) => size_fn(function),
        Expr::Value(_) => Argc::produces(1),
    }
}

impl<'s> Λ<'s> {
    pub fn sized(x: &[Spanned<Expr<'s>>]) -> Argc {
        // 5             + (borrows) 1  2        *
        // { 0, 1 } -> { 1, 1 } -> { 1, 3 } -> { 1, 2 }
        x.iter()
            .fold(Argc::takes(0).into(0), |acc, x| acc + size_expr(&x.inner))
    }
}

fn exec_lambda<'s>(
    x: Spanned<Λ<'s>>,
    c: &mut Context<'s, '_>,
    stack: &mut Stack<'s>,
) -> Result<(), Error> {
    let (x, upper) = x.raw();
    for elem in x.0 {
        let (elem, span) = elem.raw();
        match elem {
            Expr::Function(x) => match x {
                Function::Ident(x) => {
                    let (x, span) = c
                        .variables
                        .get(x)
                        .unwrap_or_else(|| {
                            println!("couldnt find definition for variable {x} at ast node {x:?}");
                            std::process::exit(1);
                        })
                        .clone()
                        .raw();
                    match x {
                        Val::Lambda(x) => {
                            exec_lambda(x.spun(span), &mut Context::inherits(c), stack)?
                        }
                        x => stack.push(x.spun(span)),
                    }
                }
                Function::Define(x) => {
                    c.variables
                        .insert(x, stack.pop().ok_or(Error::stack_empty(span))?);
                }
                x => x.spun(span).execute(c, stack)?,
            },
            Expr::Value(x) => {
                stack.push(
                    match x {
                        Value::Int(x) => Val::Int(x as i128),
                        Value::Float(x) => Val::Float(x),
                        Value::String(x) => {
                            Val::Array(x.bytes().map(|x| Val::Int(x as i128)).collect())
                        }
                        Value::Lambda(x) => Val::Lambda(x),
                    }
                    .spun(span),
                );
            }
        }
    }
    println!("{stack:?}");
    Ok(())
}

fn pervasive_binop<'a>(
    a: &Val<'a>,
    b: &Val<'a>,
    map: impl Fn(&Val<'a>, &Val<'a>) -> Result<Val<'a>, Error> + Copy,
    mismatch: impl FnOnce() -> Error + Clone,
) -> Result<Val<'a>, Error> {
    use Val::*;
    match (a, b) {
        (Array(x), Array(y)) => {
            if x.len() != y.len() {
                return Err(mismatch());
            }

            x.into_iter()
                .zip(y)
                .map(|(x, y)| pervasive_binop(x, y, map, mismatch.clone()))
                .collect::<Result<Vec<_>, _>>()
                .map(Array)
        }
        (Array(x), y) | (y, Array(x)) => x
            .into_iter()
            .map(|x| pervasive_binop(&x, &y, map, mismatch.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map(Array),
        (x, y) => map(x, y),
    }
}

fn pervasive_unop<'s>(
    x: Val<'s>,
    f: impl Fn(Val<'s>) -> Result<Val<'s>, Error> + Copy,
) -> Result<Val<'s>, Error> {
    match x {
        Val::Array(x) => x
            .into_iter()
            .map(|x| match x {
                x @ Val::Array(_) => pervasive_unop(x, f),
                x => f(x),
            })
            .collect::<Result<_, _>>()
            .map(Val::Array),
        x => f(x),
    }
}

impl<'s> Function<'s> {
    pub fn execute(
        self: Spanned<Self>,
        c: &Context<'s, '_>,
        stack: &mut Stack<'s>,
    ) -> Result<(), Error> {
        let (x, span) = self.raw();
        macro_rules! concrete_ab {
            ($x:tt) => {
                concrete_ab!(|a, b| a $x b)
            };
            ($a: expr) => {{
                let b_ = stack
                    .pop()
                    .ok_or(Error::stack_empty(span, ))?;
                let a_ = stack
                    .pop()
                    .ok_or(Error::stack_empty(span).label(
                        "got second argument from here".spun(b_.span()),
                    ))?;
                stack.push(pervasive_binop(
                    &*a_, &*b_, |a, b| {
                        match (a, b) {
                            (Val::Float(x), Val::Int(y)) | (Val::Int(y), Val::Float(x)) =>
                                Ok(Val::from(($a)(x, &(*y as f64)))),
                            (Val::Int(x), Val::Int(y)) => Ok(Val::from(($a)(x, y))),
                            (Val::Float(x), Val::Float(y)) => Ok(Val::from(($a)(x, y))),
                            (x, Val::Float(_) | Val::Int(_)) =>
                                Err(Error::ef(span, "expected number", x.ty().spun(a_.span()))),
                            (Val::Float(_) | Val::Int(_), x) =>
                                Err(Error::ef(span, "expected number", x.ty().spun(b_.span()))),
                            _ => unreachable!(),
                        }
                    },
                    || Error{
                        name: "argument length mismatch".to_string(),
                        message: "for this function".to_string().spun(span),
                        labels: vec![],
                        notes:vec![],
                    }.label("first argument".spun(a_.span())).label("second argument".spun(b_.span())),
                )?.spun(span));
            }};

        }

        macro_rules! number_ab {
            ($a:expr) => {{
                let a_ = stack.pop().ok_or(Error::stack_empty(span))?;
                let b_ = stack.pop().ok_or(
                    Error::stack_empty(span).label("got first argument from here".spun(a_.span())),
                )?;
                stack.push(
                    pervasive_binop(
                        &*a_,
                        &*b_,
                        |a, b| match (a, b) {
                            (Val::Float(_), Val::Float(_)) => {
                                Err(Error::ef(span, "int", "float".spun(a_.span()))
                                    .label("float (not int)".spun(b_.span())))
                            }
                            (Val::Int(x), Val::Int(y)) => Ok(Val::from(($a)(x, y))),
                            (x, Val::Int(_)) => {
                                Err(Error::ef(span, "expected int", x.ty().spun(a_.span())))
                            }
                            (Val::Int(_), x) => {
                                Err(Error::ef(span, "expected int", x.ty().spun(b_.span())))
                            }
                            _ => unreachable!(),
                        },
                        || {
                            Error {
                                name: "argument length mismatch".to_string(),
                                message: "for this function".to_string().spun(span),
                                labels: vec![],
                                notes: vec![],
                            }
                            .label("first argument".spun(a_.span()))
                            .label("second argument".spun(b_.span()))
                        },
                    )?
                    .spun(span),
                )
            }};
        }
        macro_rules! unary_num {
            ($x:tt) => {{
                let x = stack
                    .pop()
                    .ok_or(Error::stack_empty(span))?;
                let Val::Int(x) = x.inner else {
                    return Err(Error::ef(span, "integer", x.ty().spun(x.span())));
                };
                stack.push(Val::Int($x x).spun(span));
            }};
        }
        macro_rules! unary {
            ($x:tt) => {
                unary!(|x| $x x)
            };
            ($x:expr) => {{
                let (x, xspan) = stack.pop().ok_or(Error::stack_empty(span))?.raw();
                stack.push(
                    pervasive_unop(x, |x| {
                        Ok(match x {
                            Val::Int(x) => Val::Int(annotate::<i128>($x)(&x)),
                            Val::Float(x) => Val::Float(annotate::<f64>($x)(&x)),
                            _ => {
                                return Err(Error::ef(span, "number", x.ty().spun(xspan)));
                            }
                        })
                    })?
                    .spun(span),
                );
            }};
        }
        trait Help {
            fn pow(&self, other: &Self) -> Self;
            fn rem(&self, other: &Self) -> Self;
            fn sqrt(&self) -> Self;
        }
        impl Help for i128 {
            fn pow(&self, other: &Self) -> Self {
                i128::pow(*self, (*other).try_into().expect("please no"))
            }

            fn rem(&self, other: &Self) -> Self {
                self.rem_euclid(*other)
            }

            fn sqrt(&self) -> Self {
                self.isqrt()
            }
        }
        impl Help for f64 {
            fn pow(&self, other: &Self) -> Self {
                self.powf(*other)
            }

            fn rem(&self, other: &Self) -> Self {
                self.rem_euclid(*other)
            }

            fn sqrt(&self) -> Self {
                f64::sqrt(*self)
            }
        }
        match x {
            Self::Add => concrete_ab!(+),
            Self::Sub => concrete_ab!(-),
            Self::Mul => concrete_ab!(*),
            Self::Div => concrete_ab!(/),
            Self::Mod => concrete_ab!(Help::rem),
            Self::Pow => concrete_ab!(Help::pow),
            Self::BitAnd => number_ab!(|a, b| a & b),
            Self::Or => number_ab!(|a, b| a | b),
            Self::Xor => number_ab!(|a, b| a ^ b),
            Self::Lt => concrete_ab!(<),
            Self::Gt => concrete_ab!(>),
            Self::Le => concrete_ab!(<=),
            Self::Ge => concrete_ab!(>=),
            Self::Not => unary_num!(!),
            Self::Neg => unary!(-),
            Self::Sqrt => unary!(Help::sqrt),
            Self::Array(Some(x)) => {
                let r = match x {
                    NumberΛ::Number(x) => x as usize,
                    NumberΛ::Λ(x) => {
                        exec_lambda(x, &mut Context::inherits(c), stack)?;
                        let (y, yspan) = stack.pop().ok_or(Error::stack_empty(span))?.raw();
                        match y {
                            Val::Int(x) => x as usize,
                            z => {
                                return Err(Error::ef(span, "int", z.ty().spun(yspan)));
                            }
                        }
                    }
                };
                let r = stack.curr().len() - r;
                let result = stack.curr().split_off(r);
                stack.push(Val::Array(result.into_iter().map(|x| x.inner).collect()).spun(span))
            }
            Self::Array(None) => {
                let drained = stack.curr().drain(..).map(|x| x.inner).collect();
                stack.push(Val::Array(drained).spun(span));
            }
            Self::Dup => {
                let x = stack.pop().ok_or(Error::stack_empty(span))?.clone();
                stack.push(x);
            }
            Self::Flip => {
                let x = stack.pop().ok_or(Error::stack_empty(span))?;
                let y = stack.pop().ok_or(Error::stack_empty(span))?;
                stack.push(y);
                stack.push(x);
            }
            // basically ⎬^⎬2 (x)🗺
            Self::And(x, y) => {
                let xargs = x.argc();
                let yargs = y.argc();
                let requires = yargs.input.max(xargs.input);

                let n = stack.curr().len();
                let s = Stack(vec![stack.curr().drain(n - requires..).collect::<Vec<_>>()]);
                let mut a = s.clone();
                exec_lambda(x, &mut Context::inherits(c), &mut a)?;
                let n = a.curr().len();
                let x = a.curr().drain(n - xargs.output..);

                let mut a = s.clone();
                exec_lambda(y, &mut Context::inherits(c), &mut a)?;
                let n = a.curr().len();
                let y = a.curr().drain(n - yargs.output..);

                stack.curr().extend(x);
                stack.curr().extend(y);
            }
            _ => (),
        }
        Ok(())
    }
}

fn annotate<'a, T>(f: impl FnOnce(&T) -> T) -> impl FnOnce(&T) -> T {
    f
}
