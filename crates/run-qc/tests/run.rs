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
    // Leading flags (e.g. `--print-globals`) go before the positional arguments.
    let flags = args.iter().take_while(|arg| arg.starts_with("--")).count();
    command
        .args(&args[..flags])
        .arg(fixture("test.dat"))
        .args(&args[flags..])
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
    assert_eq!(run_ok(None, &["add_scaled", "1", "2"]), "3.0");
    assert_eq!(run_ok(None, &["sum3", "1", "2", "-3.5"]), "-0.5");
    assert_eq!(run_ok(None, &["negate", "-4"]), "4.0");
}

#[test]
fn call_by_index() {
    let index = function_index("sum3");
    assert_eq!(run_ok(None, &[&format!("#{index}"), "1", "2", "3"]), "6.0");
}

#[test]
fn call_via_global() {
    assert_eq!(run_ok(None, &[".main", "1", "2"]), "3.0");

    let offset = global_offset("main");
    assert_eq!(run_ok(None, &[&format!(".{offset}"), "1", "2"]), "3.0");
}

#[test]
fn float_globals() {
    assert_eq!(
        run_ok(Some(r#"{ "scale": 2 }"#), &["add_scaled", "1", "2"]),
        "6.0"
    );

    let offset = global_offset("scale");
    assert_eq!(
        run_ok(
            Some(&format!(r#"{{ "{offset}": 3 }}"#)),
            &["add_scaled", "1", "2"]
        ),
        "9.0"
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
        "3.0"
    );
}

#[test]
fn string_globals() {
    assert_eq!(
        run_ok(
            Some(r#"{ "greeting": "Hello, world!" }"#),
            &["get_greeting"]
        ),
        r#""Hello, world!""#
    );
}

#[test]
fn function_globals() {
    assert_eq!(
        run_ok(
            Some(r#"{ "callback": "add_scaled" }"#),
            &["call_callback", "1", "2"]
        ),
        "3.0"
    );

    let index = function_index("negate");
    assert_eq!(
        run_ok(
            Some(&format!(r##"{{ "callback": "#{index}" }}"##)),
            &["call_callback", "5", "0"]
        ),
        "-5.0"
    );
    assert_eq!(
        run_ok(
            Some(&format!(r#"{{ "main": {index} }}"#)),
            &[".main", "5", "0"]
        ),
        "-5.0"
    );
}

#[test]
fn vector_args() {
    assert_eq!(run_ok(None, &["vector_sum", "[1, 2, 3.5]"]), "6.5");
    assert_eq!(run_ok(None, &["vector_arg", "[4, 5, 6]"]), "4.0");
}

#[test]
fn string_args() {
    assert_eq!(run_ok(None, &["string_identity", "hello"]), r#""hello""#);
    assert_eq!(
        run_ok(None, &["string_identity", "Hello, world!"]),
        r#""Hello, world!""#
    );
    // Explicit JSON strings, to pass strings that would otherwise be parsed as other types.
    assert_eq!(run_ok(None, &["string_identity", r#""0""#]), r#""0""#);
    assert_eq!(
        run_ok(None, &["string_identity", r#""[1, 2, 3]""#]),
        r#""[1, 2, 3]""#
    );
    assert_eq!(run_ok(None, &["string_is_null", r#""0""#]), "0.0");
    // Invalid JSON is a string.
    assert_eq!(run_ok(None, &["string_identity", "[1, 2"]), r#""[1, 2""#);
}

#[test]
fn null_args() {
    assert_eq!(run_ok(None, &["string_is_null", "null"]), "1.0");
    assert_eq!(run_ok(None, &["entity_is_null", "null"]), "1.0");
    assert_eq!(run_ok(None, &["negate", "null"]), "0.0");
}

#[test]
fn entity_args() {
    assert_eq!(run_ok(None, &["entity_is_null", "{}"]), "0.0");
    assert_eq!(run_ok(None, &["get_health", "{}"]), "0.0");
    assert_eq!(run_ok(None, &["get_health", r#"{ "health": 50 }"#]), "50.0");
    assert_eq!(
        run_ok(None, &["origin_sum", r#"{ "origin": [1, 2, 3] }"#]),
        "6.0"
    );
    assert_eq!(run_ok(None, &["origin_sum", r#"{ "origin_y": 5 }"#]), "5.0");
    assert_eq!(
        run_ok(None, &["get_netname", r#"{ "netname": "Ranger" }"#]),
        r#""Ranger""#
    );
    assert_eq!(
        run_ok(None, &["owner_health", r#"{ "owner": { "health": 7 } }"#]),
        "7.0"
    );
    assert_eq!(run_ok(None, &["heal", r#"{ "health": 50 }"#, "25"]), "75.0");

    let offset = load_fixture()
        .field_defs()
        .iter()
        .find(|def| def.name.to_bytes() == b"health")
        .unwrap()
        .offset;
    assert_eq!(
        run_ok(None, &["get_health", &format!(r#"{{ "{offset}": 12 }}"#)]),
        "12.0"
    );
}

#[test]
fn entity_globals() {
    assert_eq!(
        run_ok(
            Some(r#"{ "player": { "health": 99 } }"#),
            &["player_health"]
        ),
        "99.0"
    );
}

#[test]
fn entity_output() {
    assert_eq!(run_ok(None, &["entity_identity", "null"]), "null");
    assert_eq!(run_ok(None, &["entity_identity", "{}"]), "{}");
    // Fields are in offset order, and vector components are printed as the whole vector.
    assert_eq!(
        run_ok(
            None,
            &[
                "entity_identity",
                r#"{ "owner": { "netname": "Ranger" }, "origin_y": 2, "health": 5 }"#
            ]
        ),
        r#"{"health":5.0,"origin":[0.0,2.0,0.0],"owner":{"netname":"Ranger"}}"#
    );
    // `null` sets a field to its zero value, which still counts as set.
    assert_eq!(
        run_ok(None, &["entity_identity", r#"{ "health": null }"#]),
        r#"{"health":0.0}"#
    );
    assert!(run_err(None, &["make_cycle", "{}"]).contains("refers to itself"));
}

#[test]
fn void_output() {
    assert_eq!(run_ok(None, &["null_function"]), "null");
}

#[test]
fn print_globals() {
    let output = run_ok(
        Some(r#"{ "player": { "health": 99 }, "greeting": "hi", "callback": "negate" }"#),
        &["--print-globals", "hurt_player"],
    );
    let output: serde_json::Value = serde_json::from_str(&output).unwrap();

    assert_eq!(
        output,
        serde_json::json!({
            "globals": {
                "scale": 1.0,
                "offset": [0.0, 0.0, 0.0],
                "greeting": "hi",
                "print": "print",
                "main": "add_scaled",
                "callback": "negate",
                "player": { "health": 89.0 },
            },
            "return": null,
        })
    );

    let output = run_ok(None, &["--print-globals", "negate", "2"]);
    let output: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(output["return"], serde_json::json!(-2.0));
    assert_eq!(output["globals"]["player"], serde_json::Value::Null);
}

#[test]
fn errors() {
    assert!(run_err(None, &["nope"]).contains("No function named `nope`"));
    assert!(run_err(None, &["print"]).contains("is a builtin"));
    assert!(run_err(None, &["call_builtin"]).contains("doesn't support builtins"));
    assert!(run_err(None, &["vector_arg", "1"]).contains("is a vector, but `1` isn't"));
    assert!(run_err(None, &["negate", "[1, 2, 3]"]).contains("is a scalar"));
    assert!(
        run_err(None, &["get_health", r#"{ "nope": 1 }"#]).contains("No entity field named `nope`")
    );
    assert!(run_err(None, &["get_health", r#"{ "health": "x" }"#]).contains("Expected a number"));
    assert!(run_err(None, &["owner_health", r#"{ "owner": null }"#]).contains("null entity"));
    assert!(run_err(None, &["add_scaled", "1"]).contains("takes 2"));
    assert!(run_err(None, &[".callback", "1", "2"]).contains("null function"));
    assert!(run_err(None, &[".scale"]).contains("is not a function"));
    assert!(run_err(Some(r#"{ "nope": 1 }"#), &["negate", "1"]).contains("No global named `nope`"));
    assert!(run_err(Some("[1]"), &["negate", "1"]).contains("must be a JSON object"));
    assert!(run_err(Some(r#"{ "scale": "x" }"#), &["negate", "1"]).contains("Expected a number"));
}
