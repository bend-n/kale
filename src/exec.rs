use crate::parser::types::Span;
use crate::parser::{
    fun::{Function, NumberΛ},
    types::*,
    util::Spanner,
};
use chumsky::span::{SimpleSpan, Span as _};
use itertools::Itertools;
use std::mem::take;
use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, Deref, DerefMut},
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
type Result<T> = std::result::Result<T, Error>;

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

#[derive(Clone, PartialEq)]
pub enum Array {
    Array(Vec<Array>),
    Int(Vec<i128>),
    Float(Vec<f64>),
}
impl Array {
    fn ty(&self) -> &'static str {
        match self {
            Array::Array(_) => "array",
            Array::Int(_) => "int",
            Array::Float(_) => "float",
        }
    }
    fn assert_int(self: Spanned<Array>, span: Span) -> Result<Spanned<Vec<i128>>> {
        self.try_map(|x, s| match x {
            Array::Int(x) => Ok(x),
            x => Err(Error::ef(span, "array[int]", x.ty().spun(s))),
        })
    }
}

macro_rules! each {
    ($y:expr,$x:expr,$in:ty => $into: ty) => {
        match $y {
            Array::Int(x) => annote::<$in, $into>($x)(x),
            Array::Float(x) => annote::<$in, $into>($x)(x),
            Array::Array(x) => annote::<$in, $into>($x)(x),
        }
    };
}
impl Array {
    fn len(&self) -> usize {
        each!(self, |x| x.len(), &Vec<_> => usize)
    }
    fn iter(&self) -> Box<dyn Iterator<Item = Val<'static>> + '_> {
        match self {
            Array::Array(items) => Box::new(items.iter().cloned().map(Val::Array)),
            Array::Int(items) => Box::new(items.iter().copied().map(Val::Int)),
            Array::Float(items) => Box::new(items.iter().copied().map(Val::Float)),
        }
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(x) => x.fmt(f),
            Self::Int(x) => x.fmt(f),
            Self::Float(x) => x.fmt(f),
        }
    }
}

impl From<Vec<i128>> for Array {
    fn from(value: Vec<i128>) -> Self {
        Array::Int(value)
    }
}
impl From<Vec<f64>> for Array {
    fn from(value: Vec<f64>) -> Self {
        Array::Float(value)
    }
}
impl From<Vec<Array>> for Array {
    fn from(value: Vec<Array>) -> Self {
        Array::Array(value)
    }
}

impl Array {
    #[track_caller]
    fn new_unchecked<'s>(value: impl Iterator<Item = Val<'s>>) -> Self {
        let mut v = value.peekable();
        let Some(inner) = v.peek() else {
            return Array::Int(vec![]);
        };
        match inner {
            Val::Array(_) => Array::Array(
                v.into_iter()
                    .map(|x| match x {
                        Val::Array(x) => x,
                        _ => panic!(),
                    })
                    .collect(),
            ),
            Val::Float(_) => Array::Float(
                v.into_iter()
                    .map(|x| match x {
                        Val::Float(x) => x,
                        _ => panic!(),
                    })
                    .collect(),
            ),
            Val::Int(_) => Array::Int(
                v.into_iter()
                    .map(|x| match x {
                        Val::Int(x) => x,
                        _ => panic!(),
                    })
                    .collect(),
            ),
            Val::Lambda(_) => panic!(),
        }
    }
    fn of<'s>(entire: Span, value: impl Iterator<Item = Spanned<Val<'s>>>) -> Result<Self> {
        let mut v = value.peekable();
        let Some(Spanned { inner, span }) = v.peek() else {
            return Ok(Array::Int(vec![]));
        };
        Ok(match inner {
            Val::Array(_) => Array::Array(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Array(x) => Ok(x),
                        _ => Err(Error::ef(entire, "array", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Float(_) => Array::Float(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Float(x) => Ok(x),
                        _ => Err(Error::ef(entire, "float", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Int(_) => Array::Int(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Int(x) => Ok(x),
                        _ => Err(Error::ef(entire, "int", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Lambda(_) => {
                return Err(Error::ef(entire, "int | array | float", "λ".spun(*span)));
            }
        })
    }
    fn new(entire: Span, value: Vec<Spanned<Val<'_>>>) -> Result<Self> {
        let Some(Spanned { inner, span }) = value.first() else {
            return Ok(Array::Int(vec![]));
        };
        Ok(match inner {
            Val::Array(_) => Array::Array(
                value
                    .into_iter()
                    .map(|x| match x.inner {
                        Val::Array(x) => Ok(x),
                        _ => Err(Error::ef(entire, "array", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Float(_) => Array::Float(
                value
                    .into_iter()
                    .map(|x| match x.inner {
                        Val::Float(x) => Ok(x),
                        _ => Err(Error::ef(entire, "float", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Int(_) => Array::Int(
                value
                    .into_iter()
                    .map(|x| match x.inner {
                        Val::Int(x) => Ok(x),
                        _ => Err(Error::ef(entire, "int", x.ty().spun(x.span))),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Lambda(_) => {
                return Err(Error::ef(entire, "int | array | float", "λ".spun(*span)));
            }
        })
    }
}

#[derive(Clone, PartialEq)]
pub enum Val<'s> {
    Array(Array),
    Lambda(Λ<'s>),
    Int(i128),
    Float(f64),
}

impl<'s> Val<'s> {
    fn assert_array(self: Spanned<Val<'s>>, span: Span) -> Result<Spanned<Array>> {
        match self.inner {
            Self::Array(x) => Ok(x.spun(self.span)),
            x => Err(Error::ef(span, "array", x.ty().spun(self.span))),
        }
    }
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

// impl ConcreteVal {
//     fn val(self) -> Val<'static> {
//         match self {
//             ConcreteVal::Array(x) => {
//                 Val::Array(x.into_iter().map(ConcreteVal::val).collect::<Vec<_>>())
//             }
//             ConcreteVal::Int(x) => Val::Int(x),
//             ConcreteVal::Float(x) => Val::Float(x),
//         }
//     }
// }

impl<'s> Val<'s> {
    // pub fn concrete(self: Spanned<Self>, user: SimpleSpan) -> Result<Spanned<ConcreteVal>> {
    //     let (x, span) = self.raw();
    //     Ok(match x {
    //         Val::Array(x) => ConcreteVal::Array(
    //             x.into_iter()
    //                 .map(|x| x.spun(span).concrete(user).map(|x| x.inner))
    //                 .collect::<Result<Vec<_>, _>>()?,
    //         ),
    //         Val::Float(x) => ConcreteVal::Float(x),
    //         Val::Int(x) => ConcreteVal::Int(x),
    //         Val::Lambda(..) => {
    //             return Err(Error {
    //                 name: "value not concrete (λ)".into(),
    //                 message: "concrete value required here".to_string().spun(user),
    //                 labels: vec!["created here".to_string().spun(span)],
    //                 notes: vec![],
    //             });
    //         }
    //     }
    //     .spun(span))
    // }
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
    let mut s = Stack::new();
    crate::ui::display_execution(exec_lambda(x, &mut Context::default(), &mut s), code);
    println!("{s:?}");
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
    fn take(&mut self, take: usize) -> impl Iterator<Item = Spanned<Val<'s>>> {
        let n = self.len();
        self.drain(n - take..)
    }
    pub fn of(x: impl Iterator<Item = Spanned<Val<'s>>>) -> Self {
        Self(vec![x.collect()])
    }
    pub fn push(&mut self, x: Spanned<Val<'s>>) {
        self.curr().push(x);
    }
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

impl<'s> Deref for Stack<'s> {
    type Target = Vec<Spanned<Val<'s>>>;

    fn deref(&self) -> &Self::Target {
        self.0.last().unwrap()
    }
}
impl<'s> DerefMut for Stack<'s> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.curr()
    }
}

#[derive(Debug)]
pub struct Error {
    pub name: String,
    pub message: Spanned<String>,
    pub labels: Vec<Spanned<String>>,
    pub notes: Vec<String>,
}

impl Default for Error {
    fn default() -> Self {
        Self {
            name: Default::default(),
            message: String::default().spun(Span::new((), 0..0)),
            labels: Default::default(),
            notes: Default::default(),
        }
    }
}

impl Error {
    pub fn stack_empty(span: Span) -> Self {
        Error {
            name: "stack empty".into(),
            message: "empty stack".to_string().spun(span),
            labels: vec![],
            notes: vec![],
        }
    }

    pub fn ef(span: Span, expected: impl Display, found: Spanned<impl Display>) -> Self {
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

impl<T> Annotate for Result<T> {
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
    assert!(crate::parser::parse_s("0≥", crate::parser::top()).argc() == Argc::takes(1).into(1));
    assert_eq!(
        crate::parser::parse_s("'0'-^9≤🔓0 1¯⎦2🔒(10×+)⬇", crate::parser::top()).argc(),
        Argc::takes(1).into(1)
    );
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
        Mask | Group | Index | Sub | Add | Mul | Div | Xor | Mod | Pow | Eq | Ne | BitAnd | Or
        | Ge | Le | Lt | Gt => Argc::takes(2).into(1),
        Reduce(_) => Argc::takes(1).into(1),
        With(x) => Argc::takes(1).into(x.argc().output),
        Map(x) => Argc::takes(1 + (x.argc().input.saturating_sub(1))).into(1),
        Open | Neg | Sqrt | Not => Argc::takes(1).into(1),
        Flip => Argc::takes(2).into(2),
        Dup => Argc::takes(1).into(2),
        Zap => Argc::takes(1).into(0),
        Array(None) => {
            Argc::takes(5 /*all */).into(1)
        }
        Array(Some(NumberΛ::Number(x))) => Argc::takes(*x as _).into(1),
        // With => Argc::takes(1).into(),
        And(a, b) => {
            Argc::takes(a.argc().input.max(b.argc().input)).into(a.argc().output + b.argc().output)
        }
        Both(x) => Argc::takes(x.argc().input * 2).into(x.argc().output * 2),
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
fn normalize_index(x: i128, size: usize) -> usize {
    match x {
        ..0 => (size as i128 + x) as usize,
        _ => x as usize,
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
) -> Result<()> {
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
                            Val::Array(Array::Int(x.bytes().map(|x| x as i128).collect()))
                        }
                        Value::Lambda(x) => Val::Lambda(x),
                    }
                    .spun(span),
                );
            }
        }
    }
    Ok(())
}

fn pervasive_binop<'a>(
    span: SimpleSpan,
    a: &Spanned<Val<'a>>,
    b: &Spanned<Val<'a>>,
    map: impl Fn(&Val<'a>, &Val<'a>) -> Result<Val<'a>> + Copy,
) -> Result<Val<'a>> {
    match (&a.inner, &b.inner) {
        (Val::Array(x), Val::Array(y)) => {
            if x.len() != y.len() {
                return Err(Error {
                    name: "argument length mismatch".to_string(),
                    message: "for this function".to_string().spun(span),
                    labels: vec![],
                    notes: vec![],
                }
                .label("first argument".spun(a.span))
                .label("second argument".spun(b.span)));
            }
            if x.ty() != y.ty() {
                return Err(Error {
                    name: "array type mismatch".to_string(),
                    message: "for this function".to_string().spun(span),
                    ..Default::default()
                })
                .label(format!("first argument of type {}", x.ty()).spun(a.span))
                .label(format!("second argument of type {}", y.ty()).spun(b.span));
            }

            x.iter()
                .zip(y.iter())
                .map(|(x, y)| {
                    pervasive_binop(span, &x.spun(a.span), &y.spun(b.span), map)
                        .map(|x| x.spun(span))
                })
                .collect::<Result<_>>()
                .and_then(|x| Array::new(span, x))
                .map(Val::Array)
        }
        (Val::Array(x), y) | (y, Val::Array(x)) => x
            .iter()
            .map(|x| {
                pervasive_binop(span, &x.spun(a.span), &y.clone().spun(b.span), map)
                    .map(|x| x.spun(span))
            })
            .collect::<Result<_>>()
            .and_then(|x| Array::new(span, x))
            .map(Val::Array),
        (x, y) => map(x, y),
    }
}

fn pervasive_unop<'s>(
    Spanned { inner: x, span }: Spanned<Val<'s>>,
    f: impl Fn(Val<'s>) -> Result<Val<'s>> + Copy,
) -> Result<Val<'s>> {
    match x {
        Val::Array(x) => x
            .iter()
            .map(|x| {
                match x {
                    x @ Val::Array(_) => pervasive_unop(x.spun(span), f),
                    x => f(x),
                }
                .map(|x| x.spun(span))
            })
            .collect::<Result<_>>()
            .and_then(|x| Array::new(span, x))
            .map(Val::Array),
        x => f(x),
    }
}

impl<'s> Function<'s> {
    pub fn execute(self: Spanned<Self>, c: &Context<'s, '_>, stack: &mut Stack<'s>) -> Result<()> {
        let (x, span) = self.raw();
        macro_rules! pop {
            () => {
                stack.pop().ok_or(Error::stack_empty(span))?
            };
        }
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
                    span,
                    &a_, &b_, |a, b| {
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
                )?.spun(span));
            }};

        }

        macro_rules! number_ab {
            ($a:expr) => {{
                let a_ = pop!();
                let b_ = stack.pop().ok_or(
                    Error::stack_empty(span).label("got first argument from here".spun(a_.span())),
                )?;
                stack.push(
                    pervasive_binop(span, &a_, &b_, |a, b| match (a, b) {
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
                    })?
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
                let x = pop!();
                let xspan = x.span();
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
            Self::Eq => concrete_ab!(==),
            Self::Ne => concrete_ab!(!=),
            Self::Ge => concrete_ab!(>=),
            Self::Not => unary_num!(!),
            Self::Neg => unary!(-),
            Self::Sqrt => unary!(Help::sqrt),
            Self::Array(Some(x)) => {
                let r = match x {
                    NumberΛ::Number(x) => x as usize,
                    NumberΛ::Λ(x) => {
                        exec_lambda(x, &mut Context::inherits(c), stack)?;
                        let (y, yspan) = pop!().raw();
                        match y {
                            Val::Int(x) => x as usize,
                            z => {
                                return Err(Error::ef(span, "int", z.ty().spun(yspan)));
                            }
                        }
                    }
                };
                let r = stack.len() - r;
                let result = stack.split_off(r);
                stack.push(Val::Array(Array::new(span, result)?).spun(span))
            }
            Self::Array(None) => {
                let drained = Array::of(span, stack.drain(..))?;
                stack.push(Val::Array(drained).spun(span));
            }
            Self::Dup => {
                let x = pop!().clone();
                stack.push(x.clone());
                stack.push(x);
            }
            Self::Zap => drop(stack.pop()),
            Self::Flip => {
                let x = pop!();
                let y = pop!();
                stack.push(x);
                stack.push(y);
            }
            Self::And(x, y) => {
                let xargs = x.argc();
                let yargs = y.argc();
                let requires = yargs.input.max(xargs.input);

                let s = Stack::of(stack.take(requires));
                let mut a = s.clone();
                exec_lambda(x, &mut Context::inherits(c), &mut a)?;
                let x = a.take(xargs.output);

                let mut a = s.clone();
                exec_lambda(y, &mut Context::inherits(c), &mut a)?;
                let y = a.take(yargs.output);

                stack.extend(x);
                stack.extend(y);
            }
            Self::Both(λ) => {
                let xargs = λ.argc();
                let mut a = Stack::of(stack.take(xargs.input));
                exec_lambda(λ.clone(), &mut Context::inherits(c), &mut a)?;

                let mut b = Stack::of(stack.take(xargs.input));
                exec_lambda(λ, &mut Context::inherits(c), &mut b)?;

                stack.extend(b.take(xargs.output));
                stack.extend(a.take(xargs.output));
            }
            Self::Mask => {
                let mask = pop!().assert_array(span)?;
                let a = pop!().assert_array(span)?;
                let Array::Int(m) = mask.inner else {
                    return Err(Error::ef(span, "array[bit]", mask.ty().spun(mask.span)));
                };
                if a.len() != m.len() {
                    return Err(Error {
                        name: "argument length mismatch".to_string(),
                        message: "for this function".to_string().spun(span),
                        labels: vec![],
                        notes: vec![],
                    }
                    .label("first argument".spun(a.span))
                    .label("second argument".spun(mask.span)));
                }
                stack.push(
                    Val::Array(Array::new_unchecked(
                        a.iter().zip(m).filter(|(_, x)| *x == 1).map(|(x, _)| x),
                    ))
                    .spun(span),
                );
            }
            Self::With(λ) => {
                let array = pop!().assert_array(span)?;
                let mut a = Stack::of(array.iter().map(|x| x.spun(array.span)));
                exec_lambda(λ, &mut Context::inherits(c), &mut a)?;
                stack.extend(a.drain(..));
            }
            Self::Index => {
                let index = pop!().assert_array(span)?.assert_int(span)?;
                let array = pop!().assert_array(span)?;
                let out = each!(
                    array.inner,
                    |x| index
                        .iter()
                        .map(|y| x.get(normalize_index(*y, x.len())).cloned().ok_or_else(|| {
                            Error {
                                name: format!(
                                    "index ({y}) out of bounds for arra of length {}",
                                    x.len()
                                ),
                                message: "here".to_string().spun(span),
                                ..Default::default()
                            }
                            .label("index from".spun(index.span))
                            .label("array from".spun(array.span))
                        }))
                        .collect::<Result<Vec<_>>>().map(Array::from).map(Val::Array),
                    Vec<_> => Result<Val>
                )?;
                stack.push(out.spun(span));
            }
            Self::Open => {
                let x = pop!().assert_array(span)?.try_map(|x, s| match x {
                    Array::Int(x) => String::from_utf8(x.into_iter().map(|x| x as u8).collect())
                        .map_err(|e| {
                            Error::ef(span, "valid utf8", "invalid utf8".spun(s))
                                .note(e.utf8_error().to_string())
                        }),
                    x => Err(Error::ef(span, "array", x.ty().spun(s))),
                })?;
                stack.push(
                    Val::Array(Array::Int(
                        std::fs::read(&*x)
                            .map_err(|e| {
                                Error::ef(span, "valid file", "invalid file".spun(x.span))
                                    .note(e.to_string())
                            })?
                            .into_iter()
                            .map(|x| x as i128)
                            .collect(),
                    ))
                    .spun(x.span),
                );
            }
            // like mask but separating
            Self::Group => {
                let elem = pop!().assert_array(span)?.assert_int(span)?;
                let array = pop!().assert_array(span)?;
                if elem.len() != array.len() {
                    return Err(Error {
                        name: "argument length mismatch".to_string(),
                        message: "for this function".to_string().spun(span),
                        labels: vec![],
                        notes: vec![],
                    }
                    .label("first argument".spun(elem.span))
                    .label("second argument".spun(array.span)));
                }
                stack.push(
                    Val::Array(each!(array.inner, |a| {
                    let mut chunked = Vec::with_capacity(32);
                    let mut chunk = Vec::with_capacity(32);
                    for (e, x) in elem.iter().copied().zip(a) {
                        if e == 1 {
                            chunk.push(x);
                        } else if !chunk.is_empty() {
                            chunked.push(Array::from(take(&mut chunk)));
                        }
                    }
                    if !chunk.is_empty() {
                        chunked.push(Array::from(take(&mut chunk)));
                    }
                    Array::Array(chunked)
                }, Vec<_> => Array))
                    .spun(span),
                );
            }
            Self::Map(λ) => {
                if λ.argc().output != 1 {
                    return Err(Error {
                        name: "parameter to 🗺 does not return 1 value".to_string(),
                        message: λ.map(|λ| format!("return {}", λ.argc().output)),
                        ..Default::default()
                    });
                }
                let x = pop!().assert_array(span)?;
                let s = Stack::of(stack.take(λ.argc().input.saturating_sub(1)));
                stack.push(
                    Val::Array(Array::new(
                        span,
                        x.iter()
                            .map(|x| {
                                let mut stack = s.clone();
                                stack.push(x.spun(span));
                                exec_lambda(λ.clone(), &mut Context::inherits(c), &mut stack)
                                    .map(|()| stack.pop().expect("calculations failed"))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )?)
                    .spun(span),
                );
            }
            Self::Reduce(λ) => {
                let a = pop!().assert_array(span)?;
                assert!(λ.argc().output == 1);
                assert!(λ.argc().input >= 2);
                if λ.argc().input == 2 {
                    stack.push(
                        a.iter()
                            .map(|x| -> Val<'s> { x })
                            .try_reduce(|x, y| {
                                let mut s = Stack::of([x, y].into_iter().map(|y| y.spun(a.span)));
                                exec_lambda(λ.clone(), &mut Context::inherits(c), &mut s)
                                    .map(|()| s.pop().unwrap().inner)
                            })?
                            .unwrap()
                            .spun(span),
                    );
                }
                // vec![1, 2].iter().reduce(|x, y| {});
                // if λ.argc() !=
            }
            _ => (),
        }
        Ok(())
    }
}

fn annotate<'a, T>(f: impl FnOnce(&T) -> T) -> impl FnOnce(&T) -> T {
    f
}

fn annote<'a, T, U>(f: impl FnOnce(T) -> U) -> impl FnOnce(T) -> U {
    f
}
