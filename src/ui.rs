use chumsky::error::RichReason;
use codespan_reporting::diagnostic::LabelStyle::{
    self, Primary, Secondary,
};
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term::Chars;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use comat::cformat as cmt;

use crate::exec::Error as ExecutionError;
use crate::parser::types::Error;

pub fn display_execution<T>(
    x: Result<T, ExecutionError>,
    code: &str,
) -> T {
    let e = match x {
        Ok(x) => return x,
        Err(e) => e,
    };
    let mut files = SimpleFiles::new();
    files.add("x.kale", code);
    let mut d = Diagnostic::<usize>::new(
        codespan_reporting::diagnostic::Severity::Error,
    )
    .with_message(e.name)
    .with_label(
        Label::new(LabelStyle::Primary, 0, e.message.span())
            .with_message(e.message.raw().0),
    );
    for label in e.labels {
        d = d.with_label(
            Label::new(LabelStyle::Secondary, 0, label.span())
                .with_message(label.raw().0),
        );
    }

    d = d.with_notes(e.notes);

    let writer = StandardStream::stderr(ColorChoice::Always);
    let mut config = codespan_reporting::term::Config::default();
    config.chars = Chars::box_drawing();
    codespan_reporting::term::emit(
        &mut writer.lock(),
        &config,
        &files,
        &d,
    )
    .unwrap();
    std::process::exit(2);
}

pub fn display<T>(
    result: Result<T, Vec<Error>>,
    code: &str,
) -> Result<T, ()> {
    let e = match result {
        Ok(x) => return Ok(x),
        Err(e) => e,
    };

    let mut files = SimpleFiles::new();
    files.add("x.kale", code);

    for e in e.into_iter().map(|e| e.map_token(|c| c.to_string())) {
        let mut d = Diagnostic::<usize>::new(
            codespan_reporting::diagnostic::Severity::Error,
        );
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
        codespan_reporting::term::emit(
            &mut writer.lock(),
            &config,
            &files,
            &d,
        )
        .unwrap();
    }
    Err(())
}
