use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::iter::once;
use std::mem::take;
use std::ops::{Add, Deref, DerefMut};

use chumsky::span::{SimpleSpan, Span as _};

use crate::parser::fun::Function;
use crate::parser::types::{Span, *};
use crate::parser::util::Spanner;
#[derive(Clone, Copy, PartialEq, Default, Eq, Hash, PartialOrd, Ord)]
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
impl Hash for Array {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Array::Array(items) => items.hash(state),
            Array::Int(items) => items.hash(state),
            Array::Float(items) => {
                items.len().hash(state);
                for x in items {
                    (x + 0.0).to_bits().hash(state);
                }
            }
        }
    }
}

impl Array {
    fn ty(&self) -> &'static str {
        match self {
            Array::Array(_) => "array",
            Array::Int(_) => "ℤ",
            Array::Float(_) => "ℝ",
        }
    }
    fn assert_int(
        self: Spanned<Array>,
        span: Span,
    ) -> Result<Spanned<Vec<i128>>> {
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
    fn sort(&mut self) {
        match self {
            Array::Int(x) => x.sort_unstable(),
            Array::Float(x) => x.sort_by_key(|x| unsafe {
                assert!(x == x && !x.is_infinite());
                umath::FF64::new(*x)
            }),
            Array::Array(_) => panic!(),
        };
    }
    fn len(&self) -> usize {
        each!(self, |x| x.len(), &Vec<_> => usize)
    }
    fn remove(&mut self, n: usize) {
        each!(self, |x| { x.remove(n); }, &mut Vec<_> => ())
    }
    fn iter(&self) -> Box<dyn Iterator<Item = Val<'static>> + '_> {
        match self {
            Array::Array(items) => {
                Box::new(items.iter().cloned().map(Val::Array))
            }
            Array::Int(items) => {
                Box::new(items.iter().copied().map(Val::Int))
            }
            Array::Float(items) => {
                Box::new(items.iter().copied().map(Val::Float))
            }
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
            Val::Map(_) | Val::Set(_) => panic!(),
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
    fn of<'s>(
        entire: Span,
        value: impl Iterator<Item = Spanned<Val<'s>>>,
    ) -> Result<Self> {
        let mut v = value.peekable();
        let Some(Spanned { inner, span }) = v.peek() else {
            return Ok(Array::Int(vec![]));
        };
        Ok(match inner {
            Val::Set(_) | Val::Map(_) => {
                return Err(Error::ef(
                    entire,
                    "array | ℝ | ℤ",
                    "container".spun(*span),
                ));
            }
            Val::Array(_) => Array::Array(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Array(x) => Ok(x),
                        _ => Err(Error::ef(
                            entire,
                            "array",
                            x.ty().spun(x.span),
                        )),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Float(_) => Array::Float(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Float(x) => Ok(x),
                        _ => Err(Error::ef(
                            entire,
                            "ℝ",
                            x.ty().spun(x.span),
                        )),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Int(_) => Array::Int(
                v.into_iter()
                    .map(|x| match x.inner {
                        Val::Int(x) => Ok(x),
                        _ => Err(Error::ef(
                            entire,
                            "ℤ",
                            x.ty().spun(x.span),
                        )),
                    })
                    .collect::<Result<_>>()?,
            ),
            Val::Lambda(_) => {
                return Err(Error::ef(
                    entire,
                    "ℤ | array | ℝ",
                    "λ".spun(*span),
                ));
            }
        })
    }
    fn new(entire: Span, value: Vec<Spanned<Val<'_>>>) -> Result<Self> {
        Self::of(entire, value.into_iter())
    }
}

#[derive(Clone, PartialEq)]
pub enum Val<'s> {
    Array(Array),
    Map(Map<'s>),
    Set(Set<'s>),
    Lambda(Λ<'s>),
    Int(i128),
    Float(f64),
}
type Map<'s> = HashMap<Val<'s>, Val<'s>>;
impl Eq for Val<'_> {}
impl Hash for Val<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Val::Map(x) => x.len().hash(state),
            Val::Float(x) => (x + 0.0).to_bits().hash(state),
            Val::Array(x) => x.hash(state),
            Val::Set(x) => x.len().hash(state),
            Val::Lambda(x) => x.hash(state),
            Val::Int(x) => x.hash(state),
        }
    }
}
type Set<'s> = HashSet<Val<'s>>;
impl<'s> Val<'s> {
    fn assert_array(
        self: Spanned<Val<'s>>,
        span: Span,
    ) -> Result<Spanned<Array>> {
        match self.inner {
            Self::Array(x) => Ok(x.spun(self.span)),
            x => Err(Error::ef(span, "array", x.ty().spun(self.span))),
        }
    }
    fn assert_map(
        self: Spanned<Val<'s>>,
        span: Span,
    ) -> Result<Spanned<Map<'s>>> {
        match self.inner {
            Self::Map(x) => Ok(x.spun(self.span)),
            x => Err(Error::ef(span, "map", x.ty().spun(self.span))),
        }
    }

    fn assert_set(
        self: Spanned<Val<'s>>,
        span: Span,
    ) -> Result<Spanned<Set<'s>>> {
        match self.inner {
            Self::Set(x) => Ok(x.spun(self.span)),
            x => Err(Error::ef(span, "set", x.ty().spun(self.span))),
        }
    }
    fn assert_int(
        self: Spanned<Val<'s>>,
        span: Span,
    ) -> Result<Spanned<i128>> {
        match self.inner {
            Self::Int(x) => Ok(x.spun(self.span)),
            x => Err(Error::ef(span, "ℤ", x.ty().spun(self.span))),
        }
    }

    fn ty(&self) -> &'static str {
        match self {
            Self::Map(_) => "map",
            Self::Set(_) => "set",
            Self::Array(_) => "array",
            Self::Float(_) => "ℝ",
            Self::Int(_) => "ℤ",
            Self::Lambda(..) => "λ",
        }
    }
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

impl std::fmt::Debug for Val<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Map(x) => x.fmt(f),
            Self::Set(x) => x.fmt(f),
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
    crate::ui::display_execution(
        exec_lambda(x, &mut Context::default(), &mut s),
        code,
    );
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
    fn pop_nth(&mut self, x: usize) -> Option<Spanned<Val<'s>>> {
        let n = self.len().checked_sub(x + 1)?;
        Some(self.remove(n))
    }
    #[track_caller]
    fn take(
        &mut self,
        take: usize,
    ) -> impl Iterator<Item = Spanned<Val<'s>>> {
        let n = self.len();
        self.drain(n - take..)
    }
    pub fn of(x: impl Iterator<Item = Spanned<Val<'s>>>) -> Self {
        Self(vec![x.collect()])
    }
    pub fn push(&mut self, x: Spanned<Val<'s>>) {
        self.curr().push(x);
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
    pub fn lazy(span: Span, message: impl Display) -> Self {
        Error {
            name: message.to_string(),
            message: "here".to_string().spun(span),
            labels: vec![],
            notes: vec![],
        }
    }

    pub fn stack_empty(span: Span) -> Self {
        Error {
            name: "stack empty".into(),
            message: "empty stack".to_string().spun(span),
            labels: vec![],
            notes: vec![],
        }
    }

    pub fn ef(
        span: Span,
        expected: impl Display,
        found: Spanned<impl Display>,
    ) -> Self {
        Error {
            name: "type mismatch".to_string(),
            labels: vec![
                format!("found {found}, not an {expected}")
                    .spun(found.span()),
            ],
            message: format!("expected {expected} found {found}")
                .spun(span),
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
        crate::parser::parse_s("5 + 1 2 ×", crate::parser::top()).argc()
            == Argc::takes(1).into(2)
    );
    assert!(
        crate::parser::parse_s("¯", crate::parser::top()).argc()
            == Argc::takes(1).into(1)
    );
    assert!(
        crate::parser::parse_s("0≥", crate::parser::top()).argc()
            == Argc::takes(1).into(1)
    );
    assert_eq!(
        crate::parser::parse_s(
            "'0'-^9≤🔓0 1¯⎦2🔒(10×+)⬇️",
            crate::parser::top()
        )
        .argc(),
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
        IndexHashMap | HashMap | Append | Length | Del | Fold(_)
        | Mask | Group | Index | Sub | Add | Mul | Div | Xor | Mod
        | Pow | Eq | Ne | BitAnd | Or | Ge | Le | Lt | Gt => {
            Argc::takes(2).into(1)
        }
        &Take(x) => Argc::takes(x as _).into(x as _),
        With(x) => Argc::takes(1).into(x.argc().output),
        Map(x) => {
            Argc::takes(1 + (x.argc().input.saturating_sub(1))).into(1)
        }
        Identity | Setify | Sort | Range | Reduce(_) | Open | Neg
        | Sqrt | Not => Argc::takes(1).into(1),
        Flip => Argc::takes(2).into(2),
        Dup => Argc::takes(1).into(2),
        Zap(None) | Zap(Some(0)) => Argc::takes(1).into(0),
        &Zap(Some(x)) => Argc::takes(x as _).into(x as usize - 1),
        Array(None) => {
            Argc::takes(5 /*all */).into(1)
        }
        Array(Some(x)) => Argc::takes(*x as _).into(1),
        // With => Argc::takes(1).into(),
        And(all) => {
            Argc::takes(all.iter().map(|x| x.argc().input).max().unwrap())
                .into(all.iter().map(|x| x.argc().output).sum())
        }
        Both(x, n) => {
            Argc::takes(x.argc().input * n).into(x.argc().output * n)
        }
        Ident(x) => Argc::takes(0).into(1),
        EmptySet => Argc::takes(0).into(1),
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
        x.iter().fold(Argc::takes(0).into(0), |acc, x| {
            acc + size_expr(&x.inner)
        })
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
                    let (x, span) = std::iter::successors(Some(&*c), |x| x.inherits)
                        .find_map(|c| c.variables.get(x))
                        .unwrap_or_else(|| {
                            println!("couldnt find definition for variable {x} at ast node {x:?}");
                            std::process::exit(1);
                        })
                        .clone()
                        .raw();
                    match x {
                        Val::Lambda(x) => exec_lambda(
                            x.spun(span),
                            &mut Context::inherits(c),
                            stack,
                        )?,
                        x => stack.push(x.spun(span)),
                    }
                }
                Function::Define(x) => {
                    c.variables.insert(
                        x,
                        stack.pop().ok_or(Error::stack_empty(span))?,
                    );
                }
                x => x.spun(span).execute(c, stack)?,
            },
            Expr::Value(x) => {
                stack.push(
                    match x {
                        Value::Int(x) => Val::Int(x as i128),
                        Value::Float(x) => Val::Float(x),
                        Value::String(x) => Val::Array(Array::Int(
                            x.bytes().map(|x| x as i128).collect(),
                        )),
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
                .label(
                    format!("first argument of type {}", x.ty())
                        .spun(a.span),
                )
                .label(
                    format!("second argument of type {}", y.ty())
                        .spun(b.span),
                );
            }

            x.iter()
                .zip(y.iter())
                .map(|(x, y)| {
                    pervasive_binop(
                        span,
                        &x.spun(a.span),
                        &y.spun(b.span),
                        map,
                    )
                    .map(|x| x.spun(span))
                })
                .collect::<Result<_>>()
                .and_then(|x| Array::new(span, x))
                .map(Val::Array)
        }
        (Val::Array(x), y) | (y, Val::Array(x)) => x
            .iter()
            .map(|x| {
                pervasive_binop(
                    span,
                    &x.spun(a.span),
                    &y.clone().spun(b.span),
                    map,
                )
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
    pub fn execute(
        self: Spanned<Self>,
        c: &Context<'s, '_>,
        stack: &mut Stack<'s>,
    ) -> Result<()> {
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
                let b_ =
                    stack.pop().ok_or(Error::stack_empty(span).label(
                        "got first argument from here".spun(a_.span()),
                    ))?;
                stack.push(
                    pervasive_binop(span, &a_, &b_, |a, b| {
                        match (a, b) {
                            (Val::Float(_), Val::Float(_)) => {
                                Err(Error::ef(
                                    span,
                                    "ℤ",
                                    "ℝ".spun(a_.span()),
                                )
                                .label("ℝ (not ℤ)".spun(b_.span())))
                            }
                            (Val::Int(x), Val::Int(y)) => {
                                Ok(Val::from(($a)(x, y)))
                            }
                            (x, Val::Int(_)) => Err(Error::ef(
                                span,
                                "expected ℤ",
                                x.ty().spun(a_.span()),
                            )),
                            (Val::Int(_), x) => Err(Error::ef(
                                span,
                                "expected ℤ",
                                x.ty().spun(b_.span()),
                            )),
                            _ => unreachable!(),
                        }
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
                    return Err(Error::ef(span, "ℤ", x.ty().spun(x.span())));
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
            Self::BitAnd => {
                number_ab!(|a, b| a & b)
            }
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
                let r = x as usize;
                let r = stack.len() - r;
                let result = stack.split_off(r);
                stack
                    .push(Val::Array(Array::new(span, result)?).spun(span))
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
            Self::Zap(None) => drop(stack.pop()),
            Self::Zap(Some(x)) => {
                stack.pop_nth(x as _).ok_or_else(|| {
                    Error::stack_empty(span).label(
                        format!("needed {x} had {}", stack.len())
                            .spun(span),
                    )
                })?;
            }
            Self::Flip => {
                let x = pop!();
                let y = pop!();
                stack.push(x);
                stack.push(y);
            }
            Self::And(λs) => {
                let x = size_fn(&Self::And(λs.clone()));
                let s = Stack::of(stack.take(x.input));
                for λ in λs {
                    let λargc = λ.argc();
                    let mut a = s.clone();
                    exec_lambda(λ, &mut Context::inherits(c), &mut a)?;
                    let x = a.take(λargc.output);
                    stack.extend(x);
                }
            }
            Self::Both(λ, n) => {
                let λargs = λ.argc();
                dbg!(λargs.input);
                let mut s = Stack::of(stack.take(λargs.input * n));
                for _ in 0..n {
                    let mut a = Stack::of(s.take(λargs.input));
                    exec_lambda(
                        λ.clone(),
                        &mut Context::inherits(c),
                        &mut a,
                    )?;
                    stack.extend(a.take(λargs.output));
                }
                assert!(s.is_empty());
            }
            Self::Mask => {
                let mask = pop!().assert_array(span)?;
                let a = pop!().assert_array(span)?;
                let Array::Int(m) = mask.inner else {
                    return Err(Error::ef(
                        span,
                        "array[bit]",
                        mask.ty().spun(mask.span),
                    ));
                };
                if a.len() != m.len() {
                    return Err(Error {
                        name: "argument length mismatch".to_string(),
                        message: "for this function"
                            .to_string()
                            .spun(span),
                        labels: vec![],
                        notes: vec![],
                    }
                    .label("first argument".spun(a.span))
                    .label("second argument".spun(mask.span)));
                }
                stack.push(
                    Val::Array(Array::new_unchecked(
                        a.iter()
                            .zip(m)
                            .filter(|(_, x)| *x == 1)
                            .map(|(x, _)| x),
                    ))
                    .spun(span),
                );
            }
            Self::With(λ) => {
                let array = pop!().assert_array(span)?;
                let mut a =
                    Stack::of(array.iter().map(|x| x.spun(array.span)));
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
                let x = pop!().assert_array(span)?.try_map(
                    |x, s| match x {
                        Array::Int(x) => String::from_utf8(
                            x.into_iter().map(|x| x as u8).collect(),
                        )
                        .map_err(|e| {
                            Error::ef(
                                span,
                                "valid utf8",
                                "invalid utf8".spun(s),
                            )
                            .note(e.utf8_error().to_string())
                        }),
                        x => Err(Error::ef(span, "array", x.ty().spun(s))),
                    },
                )?;
                stack.push(
                    Val::Array(Array::Int(
                        std::fs::read(&*x)
                            .map_err(|e| {
                                Error::ef(
                                    span,
                                    "valid file",
                                    "invalid file".spun(x.span),
                                )
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
                        message: "for this function"
                            .to_string()
                            .spun(span),
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
                        name: "parameter to 🗺 does not return 1 value"
                            .to_string(),
                        message: λ.map(|λ| {
                            format!("return {}", λ.argc().output)
                        }),
                        ..Default::default()
                    });
                }
                let x = pop!().assert_array(span)?;
                let s = Stack::of(
                    stack.take(λ.argc().input.saturating_sub(1)),
                );
                stack.push(
                    Val::Array(Array::new(
                        span,
                        x.iter()
                            .map(|x| {
                                let mut stack = s.clone();
                                stack.push(x.spun(span));
                                exec_lambda(
                                    λ.clone(),
                                    &mut Context::inherits(c),
                                    &mut stack,
                                )
                                .map(
                                    |()| {
                                        stack
                                            .pop()
                                            .expect("calculations failed")
                                    },
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )?)
                    .spun(span),
                );
            }
            Self::Range => {
                let n = pop!().assert_int(span)?;
                stack.push(
                    Val::Array(Array::Int((0..n.inner).collect()))
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
                                let mut s = Stack::of(
                                    [x, y]
                                        .into_iter()
                                        .map(|y| y.spun(a.span)),
                                );
                                exec_lambda(
                                    λ.clone(),
                                    &mut Context::inherits(c),
                                    &mut s,
                                )
                                .map(|()| s.pop().unwrap().inner)
                            })?
                            .unwrap()
                            .spun(span),
                    );
                }
                // vec![1, 2].iter().reduce(|x, y| {});
                // if λ.argc() !=
            }
            Self::Debug => {
                println!("stack: {:?} @ {span}", stack);
            }
            Self::Fold(λ) => {
                let a = pop!().assert_array(span)?;
                assert!(dbg!(λ.argc()).input >= 2);
                let input = λ.argc().input - 1;
                assert!(λ.argc().output == input);
                let accumulator = stack.take(input).collect::<Vec<_>>();
                stack.extend(a.iter().map(|x| -> Val<'s> { x }).try_fold(
                    accumulator,
                    |acc, x| {
                        let mut s = Stack::of(
                            acc.into_iter().chain(once(x.spun(span))),
                        );
                        // acc on bottom💽
                        exec_lambda(
                            λ.clone(),
                            &mut Context::inherits(c),
                            &mut s,
                        )
                        .map(|()| s.take(input).collect())
                    },
                )?)
            }
            Self::Take(n) => {
                let z = stack.len();
                stack.drain(..z - n as usize).for_each(drop);
            }
            Self::Del => {
                let n = pop!().assert_int(span)?;
                let mut a = pop!().assert_array(span)?;
                a.inner.remove(n.inner as usize);
                stack.push(a.map(Val::Array));
            }
            Self::Sort => {
                let mut a = pop!().assert_array(span)?;
                a.sort();
                stack.push(a.map(Val::Array));
            }
            Self::Setify => {
                let x = pop!().assert_array(span)?;
                stack.push(x.map(|x| {
                    Val::Set(Set::from_iter(
                        x.iter().map(|x| -> Val<'s> { x }),
                    ))
                }));
            }
            Self::EmptySet => stack.push(Val::Set(Set::new()).spun(span)),
            Self::Append => {
                let element = pop!();
                let container =
                    stack.last_mut().ok_or(Error::stack_empty(span))?;
                match &mut container.inner {
                    Val::Array(x) => {}
                    Val::Set(x) => drop(x.insert(element.inner)),
                    y => {
                        return Err(Error::ef(
                            span,
                            "array | set",
                            y.ty().spun(container.span),
                        ));
                    }
                }
            }
            Self::Length => {
                let x = stack.last().ok_or(Error::stack_empty(span))?;
                stack.push(
                    Val::Int(match &x.inner {
                        Val::Array(x) => x.len(),
                        Val::Set(x) => x.len(),
                        y => {
                            return Err(Error::ef(
                                span,
                                "array | set",
                                y.ty().spun(x.span),
                            ));
                        }
                    } as i128)
                    .spun(span),
                );
            }
            Self::HashMap => {
                let vals = pop!().assert_array(span)?;
                let keys = pop!().assert_array(span)?;
                if vals.len() != keys.len() {
                    return Err(Error::lazy(span, "bad"));
                }
                stack.push(
                    Val::Map(
                        keys.iter().zip(vals.iter()).collect::<Map>(),
                    )
                    .spun(span),
                );
            }
            Self::IndexHashMap => {
                let index = pop!();
                let map = pop!().assert_map(span)?;
                stack.push(
                    map.get(&index.inner)
                        .ok_or(Error::lazy(span, "indexfail"))?
                        .clone()
                        .spun(span),
                );
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
