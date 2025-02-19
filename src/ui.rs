use crate::parser::types::Error;
use chumsky::{error::RichReason, prelude::*};
use codespan_reporting::diagnostic::LabelStyle::{Primary, Secondary};
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

use codespan_reporting::term::Chars;
use comat::cformat as cmt;
use itertools::Itertools;

pub fn display<T>(result: Result<T, Vec<Error>>, code: &str) -> Result<T, ()> {
    let e = match result {
        Ok(x) => return Ok(x),
        Err(e) => e,
    };

    let mut files = SimpleFiles::new();
    files.add("x.kale", code);

    for e in e.into_iter().map(|e| e.map_token(|c| c.to_string())) {
        let mut d = Diagnostic::<usize>::new(codespan_reporting::diagnostic::Severity::Error);
        // let mut o = lerr::Error::new(code);
        d = d.with_label(Label {
            style: Primary,
            file_id: 0,
            range: e.span().into_range(),
            message: "here".into(),
        });
        // o.label((e.span().into_range(), "here"));
        match e.reason() {
            RichReason::Custom(x) => {
                d = d.with_message(cmt!("{red}error{reset}: {x}"));
                // o.message(cmt!("{red}error{reset}: {x}"));
            }
            RichReason::ExpectedFound { .. } => {
                d = d.with_message(format!("{e}"));
                // o.message(cmt!("{red}error{reset}: {e}"));
            } // RichReason::Many(x) => {
              //     match &x[..] {
              //         [x, rest @ ..] => {
              //             o.message(cmt!("{red}error{reset}: {x}"));
              //             for elem in rest {
              //                 o.note(cmt!("{yellow}also{reset}: {elem}"));
              //             }
              //         }
              //         _ => unreachable!(),
              //     };
              // }
        }
        dbg!(e.contexts().collect::<Vec<_>>());

        for (l, span) in e.contexts() {
            d = d.with_label(Label {
                style: Secondary,
                file_id: 0,
                range: span.into_range(),
                message: cmt!("{yellow}while parsing this{reset}: {l}"),
            })
            // o.label((
            //     span.into_range(),
            //     cmt!("{yellow}while parsing this{reset}: {l}"),
            // ));
        }
        // eprintln!("{o}");

        let writer = StandardStream::stderr(ColorChoice::Always);
        let mut config = codespan_reporting::term::Config::default();
        config.chars = Chars::box_drawing();

        codespan_reporting::term::emit(&mut writer.lock(), &config, &files, &d).unwrap();
    }
    Err(())
}
