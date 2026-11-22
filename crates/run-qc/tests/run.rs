//! End-to-end tests for `run-qc`, using `fixtures/test.qc` compiled with `fteqcc -O0`.

use std::{
    io::Cursor,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use qcvm::{ErasedAddrInfo, ErasedDefs, QcVm, Type, VmOptions};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Checks that `test.dat` isn't older than `test.qc`. This doesn't catch every case where the
/// bytecode doesn't match the source, but catches forgetting to recompile after editing it.
///
/// This allows equal modification times, as a fresh checkout gives both files (nearly) the same
/// one.
fn assert_bytecode_fresh() {
    let modified = |name| {
        std::fs::metadata(fixture(name))
            .and_then(|meta| meta.modified())
            .unwrap_or_else(|e| panic!("Couldn't get the modification time of {name}: {e}"))
    };

    assert!(
        modified("test.dat") >= modified("test.qc"),
        "tests/fixtures/test.dat is older than tests/fixtures/test.qc: recompile it by running \
         `fteqcc -O0` in tests/fixtures"
    );
}

/// Runs `run-qc` on the fixture.
fn run_qc(globals: Option<&str>, args: &[&str]) -> Output {
    assert_bytecode_fresh();

    let mut command = Command::new(env!("CARGO_BIN_EXE_run-qc"));
    if let Some(globals) = globals {
        command.args(["--globals", globals]);
    }
    command
        .arg(fixture("test.dat"))
        .args(args)
        .output()
        .expect("Couldn't run run-qc")
}

/// Runs `run-qc`, expecting it to succeed, and returns its output.
fn run_ok(globals: Option<&str>, args: &[&str]) -> String {
    let output = run_qc(globals, args);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "run-qc {args:?} failed:\n{stderr}");
    stdout.trim_end().to_owned()
}

/// Runs `run-qc`, expecting it to fail, and returns its error output.
fn run_err(globals: Option<&str>, args: &[&str]) -> String {
    let output = run_qc(globals, args);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !output.status.success(),
        "run-qc {args:?} succeeded with output {stdout:?}"
    );
    String::from_utf8(output.stderr).unwrap()
}

struct AnyCrc(i32);

impl ErasedDefs for AnyCrc {
    fn crc(&self) -> i32 {
        self.0
    }

    fn global(&self, _: u16, _: Type) -> Option<ErasedAddrInfo> {
        None
    }

    fn global_by_name(&self, _: &str) -> Option<ErasedAddrInfo> {
        None
    }

    fn globals(&self) -> Vec<ErasedAddrInfo> {
        Vec::new()
    }

    fn field(&self, _: u16, _: Type) -> Option<ErasedAddrInfo> {
        None
    }

    fn field_by_name(&self, _: &str) -> Option<ErasedAddrInfo> {
        None
    }

    fn builtin_name(&self, _: u32) -> Option<&'static str> {
        None
    }
}

/// Loads the fixture, to look up indices and offsets that change whenever it's recompiled.
fn load_fixture() -> QcVm {
    let bytes = std::fs::read(fixture("test.dat")).unwrap();
    let crc = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
    QcVm::load_with_defs(Cursor::new(bytes), &AnyCrc(crc), VmOptions::default()).unwrap()
}

fn function_index(name: &str) -> i32 {
    let name = std::ffi::CString::new(name).unwrap();
    load_fixture().functions().get_by_name(&name).unwrap().index
}

fn global_offset(name: &str) -> u16 {
    load_fixture()
        .global_defs()
        .iter()
        .find(|def| def.name.to_bytes() == name.as_bytes())
        .unwrap_or_else(|| panic!("No global named {name}"))
        .offset
}

#[test]
fn bytecode_is_fresh() {
    assert_bytecode_fresh();
}

#[test]
fn call_by_name() {
    assert_eq!(run_ok(None, &["add_scaled", "1", "2"]), "3");
    assert_eq!(run_ok(None, &["sum3", "1", "2", "-3.5"]), "-0.5");
    assert_eq!(run_ok(None, &["negate", "-4"]), "4");
}

#[test]
fn call_by_index() {
    let index = function_index("sum3");
    assert_eq!(run_ok(None, &[&format!("#{index}"), "1", "2", "3"]), "6");
}

#[test]
fn call_via_global() {
    assert_eq!(run_ok(None, &[".main", "1", "2"]), "3");

    let offset = global_offset("main");
    assert_eq!(run_ok(None, &[&format!(".{offset}"), "1", "2"]), "3");
}

#[test]
fn float_globals() {
    assert_eq!(
        run_ok(Some(r#"{ "scale": 2 }"#), &["add_scaled", "1", "2"]),
        "6"
    );

    let offset = global_offset("scale");
    assert_eq!(
        run_ok(
            Some(&format!(r#"{{ "{offset}": 3 }}"#)),
            &["add_scaled", "1", "2"]
        ),
        "9"
    );
}

#[test]
fn vector_globals() {
    assert_eq!(
        run_ok(Some(r#"{ "offset": [1, 2, 3.5] }"#), &["offset_sum"]),
        "6.5"
    );

    let offset = global_offset("offset");
    assert_eq!(
        run_ok(
            Some(&format!(r#"{{ "{offset}": [1, 1, 1] }}"#)),
            &["offset_sum"]
        ),
        "3"
    );
}

#[test]
fn string_globals() {
    assert_eq!(
        run_ok(
            Some(r#"{ "greeting": "Hello, world!" }"#),
            &["get_greeting"]
        ),
        "Hello, world!"
    );
}

#[test]
fn function_globals() {
    assert_eq!(
        run_ok(
            Some(r#"{ "callback": "add_scaled" }"#),
            &["call_callback", "1", "2"]
        ),
        "3"
    );

    let index = function_index("negate");
    assert_eq!(
        run_ok(
            Some(&format!(r##"{{ "callback": "#{index}" }}"##)),
            &["call_callback", "5", "0"]
        ),
        "-5"
    );
    assert_eq!(
        run_ok(
            Some(&format!(r#"{{ "main": {index} }}"#)),
            &[".main", "5", "0"]
        ),
        "-5"
    );
}

#[test]
fn errors() {
    assert!(run_err(None, &["nope"]).contains("No function named `nope`"));
    assert!(run_err(None, &["print"]).contains("is a builtin"));
    assert!(run_err(None, &["call_builtin"]).contains("doesn't support builtins"));
    assert!(run_err(None, &["vector_arg"]).contains("only float arguments are supported"));
    assert!(run_err(None, &["add_scaled", "1"]).contains("takes 2"));
    assert!(run_err(None, &[".callback", "1", "2"]).contains("null function"));
    assert!(run_err(None, &[".scale"]).contains("is not a function"));
    assert!(run_err(Some(r#"{ "nope": 1 }"#), &["negate", "1"]).contains("No global named `nope`"));
    assert!(run_err(Some("[1]"), &["negate", "1"]).contains("must be a JSON object"));
    assert!(run_err(Some(r#"{ "scale": "x" }"#), &["negate", "1"]).contains("Expected a number"));
}
