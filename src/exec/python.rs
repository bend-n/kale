use std::ffi::CString;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::*;

use super::{Array, Context, Error, Stack, Val};
use crate::parser::types::Spanned;
use crate::parser::util::Spanner as _;
impl<'py> IntoPyObject<'py> for Array {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(
        self,
        p: Python<'py>,
    ) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Array(x) => PyList::new(p, x),
            Self::Int(x) => PyList::new(p, x),
            Self::Float(x) => PyList::new(p, x),
        }
    }
}

impl<'s, 'py> IntoPyObject<'py> for Val<'s> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(
        self,
        p: Python<'py>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Self::Float(x) => x.into_pyobject(p).map(Bound::into_any)?,
            Self::Int(x) => x.into_pyobject(p).map(Bound::into_any)?,
            Self::Array(x) => x.into_pyobject(p).map(Bound::into_any)?,
            Self::Set(x) => x.into_pyobject(p).map(Bound::into_any)?,
            Self::Map(x) => x.into_pyobject(p).map(Bound::into_any)?,
            Self::Lambda(_) => return Err(PyTypeError::new_err("λ")),
        })
    }
}

impl<'py, 's> FromPyObject<'py> for Val<'s> {
    fn extract_bound(x: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(match () {
            () if let Ok(x) = x.extract::<i128>() => Val::Int(x),
            () if let Ok(x) = x.extract::<f64>() => Val::Float(x),
            () if let Ok(x) = x.extract::<bool>() => Val::Int(x as i128),
            () if let Ok(x) = x.downcast::<PyList>() => {
                if let Ok(y) = x.get_item(0) {
                    match () {
                        () if y.is_instance_of::<PyFloat>() => {
                            Val::Array(Array::Float(
                                x.into_iter()
                                    .map(|x| x.extract::<f64>())
                                    .try_collect()?,
                            ))
                        }
                        () if y.is_instance_of::<PyInt>() => {
                            Val::Array(Array::Int(
                                x.into_iter()
                                    .map(|x| x.extract::<i128>())
                                    .try_collect()?,
                            ))
                        }
                        _ => {
                            return Err(PyTypeError::new_err(
                                "bad array types",
                            ));
                        }
                    }
                } else {
                    Val::Array(Array::Int(vec![]))
                }
            }
            () if let Ok(x) = x.downcast::<PySet>() => Val::Set(
                x.into_iter()
                    .map(|x| x.extract::<Val<'s>>())
                    .try_collect()?,
            ),
            _ => return Err(PyTypeError::new_err("bad types")),
        })
    }
}

pub fn exec<'s>(
    span: super::Span,
    code: Spanned<CString>,
    stack: &mut Stack<'s>,
    argc: super::Argc,
    context: &Context<'s, '_>,
) -> super::Result<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|g| {
        let locals = PyDict::new(g);
        context
            .all()
            .flat_map(|x| {
                x.variables.iter().map(|(x, y)| (x, y.inner.clone()))
            })
            .for_each(|(k, v)| {
                _ = locals.set_item(k, v);
            });
        let s = stack
            .take(argc.input)
            .map(|Spanned { inner, span }| {
                let t = inner.ty();
                inner.into_pyobject(g).map_err(|y| (y, t, span))
            })
            .try_collect::<Vec<_>>()
            .map_err(|(e, ty, span)| super::Error {
                name: format!("element ({ty}) → python ({e}) failure"),
                message: "here".to_string().spun(span),
                ..Default::default()
            })?;
        locals
            .set_item("s", PyList::new(g, s).unwrap())
            .map_err(|_| Error::lazy(span, "what is wrong with python"))?;
        g.run(&code, None, Some(&locals)).map_err(|x| {
            x.display(g);
            Error::lazy(code.span, "you wrote your 🐍 (󰌠 python) wrong")
        })?;
        let x = locals.get_item("s").unwrap().unwrap();
        let x = x.downcast::<PyList>().unwrap();
        let n = x.len();
        stack.extend(
            x.into_iter()
                .skip(n.saturating_sub(argc.output))
                .map(|x| x.extract::<Val<'_>>().map(|x| x.spun(span)))
                .try_collect::<Vec<_>>()
                .map_err(|_| Error::lazy(code.span, "nooo"))?,
        );
        Ok(())
    })
}
