use crate::parser::types::Error;
use chumsky::{error::RichReason, prelude::*};
use comat::cformat as cmt;

pub fn display<T>(result: Result<T, Vec<Error>>, code: &str) -> Result<T, ()> {
    let e = match result {
        Ok(x) => return Ok(x),
        Err(e) => e,
    };

    for e in e.into_iter().map(|e| e.map_token(|c| c.to_string())) {
        let mut o = lerr::Error::new(code);
        o.label((e.span().into_range(), "here"));
        match e.reason() {
            RichReason::Custom(x) => {
                o.message(cmt!("{red}error{reset}: {x}"));
            }
            RichReason::ExpectedFound { .. } => {
                o.message(cmt!("{red}error{reset}: {e}"));
            }
            RichReason::Many(x) => {
                match &x[..] {
                    [x, rest @ ..] => {
                        o.message(cmt!("{red}error{reset}: {x}"));
                        for elem in rest {
                            o.note(cmt!("{yellow}also{reset}: {elem}"));
                        }
                    }
                    _ => unreachable!(),
                };
            }
        }
        for (l, span) in e.contexts() {
            o.label((
                span.into_range(),
                cmt!("{yellow}while parsing this{reset}: {l}"),
            ));
        }
        eprintln!("{o}");
    }
    Err(())
}
