//! Runs a function from a compiled QuakeC `progs.dat`.
//!
//! ```sh
//! run-qc --globals '{ "foo": 1, "bar": "Hello, world!" }' my-progs.dat my_func 1 2 3
//! ```
//!
//! See `run-qc --help` for the ways that functions and globals can be specified.

use std::{ffi::CString, io::Cursor, path::PathBuf, sync::Arc};

use anyhow::{Context as _, bail, format_err};
use clap::Parser;
use qcvm::{
    BuiltinDef, EntityRef, EntityTypeDef, ErasedAddrInfo, ErasedDefs, Function, FunctionDef,
    FunctionIndex, GlobalDef, QcVm, Type, Value, VmScalarType, VmType,
    userdata::{AddrError, ErasedContext, ErasedFunction},
};

/// Run a function from a compiled QuakeC `progs.dat`.
///
/// Builtins aren't supported, so calling a function that (directly or indirectly) calls a builtin
/// is an error.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Globals to set before running the function, as a JSON object.
    ///
    /// Keys are global names or offsets (e.g. `{ "foo": 1, "40": "Hello" }`). Values are converted
    /// according to the global's type: numbers for floats, `[x, y, z]` for vectors, strings for
    /// strings, and function names, `"#N"` or numbers for functions. Globals without a type (e.g.
    /// offsets without a definition) get a type based on the JSON value.
    #[arg(long, short)]
    globals: Option<String>,

    /// The compiled QuakeC bytecode.
    progs: PathBuf,

    /// The function to run: a function name (`my_func`), a function index (`#4`, which must be
    /// quoted in most shells), or a global holding a function, by name (`.main`) or by offset
    /// (`.40`).
    function: String,

    /// The arguments to pass to the function. Only floats are supported for now.
    #[arg(allow_negative_numbers = true)]
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<f32>,
}

/// A function specified on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FuncSelector {
    /// `my_func`
    Name(String),
    /// `#4`
    Index(i32),
    /// `.main` or `.40`
    Global(GlobalKey),
}

/// A global specified by name or offset.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GlobalKey {
    Name(String),
    Offset(u16),
}

impl GlobalKey {
    fn parse(s: &str) -> anyhow::Result<Self> {
        if s.is_empty() {
            bail!("Empty global name");
        }

        Ok(match s.parse::<u16>() {
            Ok(offset) => Self::Offset(offset),
            Err(_) if s.starts_with(|c: char| c.is_ascii_digit()) => {
                bail!("Invalid global offset `{s}`")
            }
            Err(_) => Self::Name(s.to_owned()),
        })
    }
}

impl FuncSelector {
    fn parse(s: &str) -> anyhow::Result<Self> {
        if let Some(index) = s.strip_prefix('#') {
            let index = index
                .parse()
                .with_context(|| format!("Invalid function index `{s}`"))?;
            Ok(Self::Index(index))
        } else if let Some(global) = s.strip_prefix('.') {
            Ok(Self::Global(GlobalKey::parse(global)?))
        } else if s.is_empty() {
            bail!("Empty function name")
        } else {
            Ok(Self::Name(s.to_owned()))
        }
    }
}

/// Defs that don't know about any globals, fields or builtins, and accept `progs.dat`s with CRC
/// `crc` (i.e. any `progs.dat`, since we read the CRC from the file itself).
struct NoDefs {
    crc: i32,
}

impl ErasedDefs for NoDefs {
    fn crc(&self) -> i32 {
        self.crc
    }

    fn global(&self, _offset: u16, _ty: Type) -> Option<ErasedAddrInfo> {
        None
    }

    fn global_by_name(&self, _name: &str) -> Option<ErasedAddrInfo> {
        None
    }

    fn globals(&self) -> Vec<ErasedAddrInfo> {
        Vec::new()
    }

    fn field(&self, _offset: u16, _ty: Type) -> Option<ErasedAddrInfo> {
        None
    }

    fn field_by_name(&self, _name: &str) -> Option<ErasedAddrInfo> {
        None
    }

    fn builtin_name(&self, _number: u32) -> Option<&'static str> {
        None
    }
}

/// A host that provides nothing: no builtins, entities or host globals.
struct NoHost;

impl ErasedContext for NoHost {
    fn dyn_builtin(&self, def: &BuiltinDef) -> anyhow::Result<Arc<dyn ErasedFunction>> {
        let name = def.name.to_string_lossy();
        match def.builtin_number() {
            Some(number) => {
                bail!("Builtin #{number} `{name}` was called, but run-qc doesn't support builtins")
            }
            None => bail!("Builtin `{name}` was called, but run-qc doesn't support builtins"),
        }
    }

    fn dyn_entity_get(
        &self,
        _erased_ent: EntityRef,
        _field: u16,
        _ty: Type,
        _entity_def: &EntityTypeDef,
    ) -> anyhow::Result<Value, AddrError<anyhow::Error>> {
        Err(format_err!("run-qc doesn't support entities").into())
    }

    fn dyn_entity_set(
        &mut self,
        _erased_ent: EntityRef,
        _field: u16,
        _value: Value,
        _entity_def: &EntityTypeDef,
    ) -> anyhow::Result<(), AddrError<anyhow::Error>> {
        Err(format_err!("run-qc doesn't support entities").into())
    }

    fn dyn_global(&self, _def: u16, _ty: Type) -> Result<Value, AddrError<anyhow::Error>> {
        Err(AddrError::OutOfRange)
    }

    fn dyn_set_global(&mut self, _def: u16, _value: Value) -> Result<(), AddrError<anyhow::Error>> {
        Err(AddrError::OutOfRange)
    }

    fn dyn_global_type(&self, _def: u16) -> Option<Type> {
        None
    }
}

/// Reads the CRC from the header of a `progs.dat` (after the version).
fn read_crc(bytes: &[u8]) -> anyhow::Result<i32> {
    let crc = bytes
        .get(4..8)
        .ok_or_else(|| format_err!("File is too short to be a progs.dat"))?;
    Ok(i32::from_le_bytes(crc.try_into().unwrap()))
}

/// Whether a def is typed, i.e. tells us how to interpret the global.
fn is_typed(def: &GlobalDef) -> bool {
    def.type_ != VmType::Scalar(VmScalarType::Void)
}

/// Finds the offset and (if known) the definition of a global. If several defs have the same
/// offset (e.g. a vector and its `_x` component), `prefer_vector` chooses between them.
fn resolve_global<'a>(
    vm: &'a QcVm,
    key: &GlobalKey,
    prefer_vector: bool,
) -> anyhow::Result<(u16, Option<&'a GlobalDef>)> {
    let pick = |defs: Vec<&'a GlobalDef>| {
        let typed = || defs.iter().copied().filter(|def| is_typed(def));
        typed()
            .find(|def| (def.type_ == VmType::Vector) == prefer_vector)
            .or_else(|| typed().next())
            .or_else(|| defs.first().copied())
    };

    match key {
        GlobalKey::Offset(offset) => {
            let defs = vm
                .global_defs()
                .iter()
                .filter(|def| def.offset == *offset)
                .collect();
            // Check that the offset exists, as `set_global` can't tell us which global was wrong.
            vm.global(*offset)
                .with_context(|| format!("No global at offset {offset}"))?;
            Ok((*offset, pick(defs)))
        }
        GlobalKey::Name(name) => {
            let defs = vm
                .global_defs()
                .iter()
                .filter(|def| def.name.to_bytes() == name.as_bytes())
                .collect();
            let def = pick(defs).ok_or_else(|| format_err!("No global named `{name}`"))?;
            Ok((def.offset, Some(def)))
        }
    }
}

/// Finds a function by name or index (but not via a global).
fn function_by_name_or_index<'a>(
    vm: &'a QcVm,
    selector: &FuncSelector,
) -> anyhow::Result<&'a FunctionDef> {
    match selector {
        FuncSelector::Name(name) => {
            let name = CString::new(name.as_str())?;
            vm.functions()
                .get_by_name(&name)
                .with_context(|| format!("No function named `{}`", name.to_string_lossy()))
        }
        FuncSelector::Index(index) => vm
            .functions()
            .get_by_index(*index)
            .with_context(|| format!("No function with index {index}")),
        FuncSelector::Global(_) => bail!("Functions can't be specified via globals here"),
    }
}

/// Finds the function to run.
fn resolve_function<'a>(vm: &'a QcVm, selector: &FuncSelector) -> anyhow::Result<&'a FunctionDef> {
    let FuncSelector::Global(key) = selector else {
        return function_by_name_or_index(vm, selector);
    };

    let (offset, def) = resolve_global(vm, key, false)?;
    if let Some(def) = def.filter(|def| is_typed(def))
        && def.type_ != VmType::Scalar(VmScalarType::Function)
    {
        bail!("Global {def} is not a function");
    }

    let index = match vm.global(offset)? {
        Value::Function(Function::Index(FunctionIndex(index))) => index,
        // Untyped globals are read as floats that preserve the bits.
        Value::Float(bits) => bits.to_bits() as i32,
        Value::Function(Function::Host(_)) => bail!("Global {offset} holds a host function"),
        other => bail!("Global {offset} holds {other:?}, not a function"),
    };

    if index == 0 {
        bail!("Global {offset} holds the null function");
    }

    vm.functions()
        .get_by_index(index)
        .with_context(|| format!("Global {offset} holds an invalid function index ({index})"))
}

/// Converts a JSON value to a function, for function globals.
fn json_to_function(vm: &QcVm, json: &serde_json::Value) -> anyhow::Result<Value> {
    let index = match json {
        serde_json::Value::Number(n) => {
            let index = n
                .as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| format_err!("Invalid function index {n}"))?;
            // Index 0 is the null function, which can't be looked up but is a valid value.
            if index != 0 {
                vm.functions().get_by_index(index)?;
            }
            index
        }
        serde_json::Value::String(s) => {
            let selector = FuncSelector::parse(s)?;
            function_by_name_or_index(vm, &selector)?.index
        }
        other => bail!("Expected a function name or index, found {other}"),
    };

    Ok(FunctionIndex(index).to_value())
}

fn json_to_float(json: &serde_json::Value) -> anyhow::Result<f32> {
    match json {
        serde_json::Value::Number(n) => n
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| format_err!("Invalid number {n}")),
        serde_json::Value::Bool(b) => Ok(if *b { 1. } else { 0. }),
        other => bail!("Expected a number, found {other}"),
    }
}

fn json_to_vector(json: &serde_json::Value) -> anyhow::Result<Value> {
    match json {
        serde_json::Value::Array(items) if items.len() == 3 => {
            let [x, y, z] = [&items[0], &items[1], &items[2]].map(json_to_float);
            Ok(Value::Vector([x?, y?, z?].into()))
        }
        other => bail!("Expected a vector (`[x, y, z]`), found {other}"),
    }
}

fn json_to_string(json: &serde_json::Value) -> anyhow::Result<Value> {
    match json {
        serde_json::Value::String(s) => Value::try_from(s.as_str()),
        other => bail!("Expected a string, found {other}"),
    }
}

/// Converts a JSON value to a value for a global of type `ty`, or a type inferred from the JSON if
/// the global is untyped.
fn json_to_value(vm: &QcVm, ty: Option<VmType>, json: &serde_json::Value) -> anyhow::Result<Value> {
    use serde_json::Value as Json;

    match ty {
        Some(VmType::Vector) => json_to_vector(json),
        Some(VmType::Scalar(VmScalarType::Float)) => json_to_float(json).map(Value::Float),
        Some(VmType::Scalar(VmScalarType::String)) => json_to_string(json),
        Some(VmType::Scalar(VmScalarType::Function)) => json_to_function(vm, json),
        Some(VmType::Scalar(VmScalarType::Void)) | None => match json {
            Json::Number(_) | Json::Bool(_) => json_to_float(json).map(Value::Float),
            Json::Array(_) => json_to_vector(json),
            Json::String(_) => json_to_string(json),
            other => bail!("Can't infer a type for {other}"),
        },
        Some(other) => bail!("Globals of type {other} aren't supported yet"),
    }
}

/// Sets the globals specified by the JSON object `json`.
fn apply_globals(vm: &QcVm, json: &str) -> anyhow::Result<()> {
    let json: serde_json::Value = serde_json::from_str(json).context("Invalid globals JSON")?;
    let serde_json::Value::Object(globals) = json else {
        bail!("Globals must be a JSON object, found {json}");
    };

    for (key, json) in &globals {
        (|| {
            let key = GlobalKey::parse(key)?;
            let (offset, def) = resolve_global(vm, &key, json.is_array())?;
            let value = json_to_value(vm, def.map(|def| def.type_), json)?;
            vm.set_global(offset, value)
        })()
        .with_context(|| format!("Couldn't set global `{key}`"))?;
    }

    Ok(())
}

/// Formats a value returned from QuakeC.
fn format_value(value: &Value) -> String {
    match value {
        Value::Void => "void".to_owned(),
        Value::Float(f) => f.to_string(),
        Value::Vector(v) => format!("'{} {} {}'", v.x, v.y, v.z),
        Value::String(s) => s.to_string_lossy().into_owned(),
        Value::Function(Function::Index(FunctionIndex(index))) => format!("#{index}"),
        other => format!("{other:?}"),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let bytes = std::fs::read(&args.progs)
        .with_context(|| format!("Couldn't read {}", args.progs.display()))?;
    let defs = NoDefs {
        crc: read_crc(&bytes)?,
    };
    let vm = QcVm::load_with_defs(Cursor::new(bytes), &defs, Default::default())
        .with_context(|| format!("Couldn't load {}", args.progs.display()))?;

    if let Some(globals) = &args.globals {
        apply_globals(&vm, globals)?;
    }

    let selector = FuncSelector::parse(&args.function)?;
    let function = resolve_function(&vm, &selector)?;
    let name = function.name.to_string_lossy();

    if function.is_builtin() {
        bail!("`{name}` is a builtin, which run-qc doesn't support");
    }

    if let Some(i) = function
        .args
        .iter()
        .position(|ty| !matches!(ty, Type::Float | Type::AnyScalar))
    {
        let invalid_type = function.args[i];

        bail!(
            "Argument {i} of `{name}` is {invalid_type}, but only float arguments are supported for now"
        );
    }

    let params = args
        .args
        .iter()
        .copied()
        .map(Value::Float)
        .collect::<Vec<_>>();
    let ret = vm.run(&mut NoHost, function.index, params)?;

    println!("{}", format_value(&ret.scalar()?));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_selectors() {
        assert_eq!(
            FuncSelector::parse("my_func").unwrap(),
            FuncSelector::Name("my_func".into())
        );
        assert_eq!(FuncSelector::parse("#4").unwrap(), FuncSelector::Index(4));
        assert_eq!(
            FuncSelector::parse(".main").unwrap(),
            FuncSelector::Global(GlobalKey::Name("main".into()))
        );
        assert_eq!(
            FuncSelector::parse(".40").unwrap(),
            FuncSelector::Global(GlobalKey::Offset(40))
        );
        assert!(FuncSelector::parse("#").is_err());
        assert!(FuncSelector::parse("#foo").is_err());
        assert!(FuncSelector::parse(".").is_err());
        assert!(FuncSelector::parse(".4x").is_err());
        assert!(FuncSelector::parse("").is_err());
    }

    #[test]
    fn parse_global_keys() {
        assert_eq!(GlobalKey::parse("4").unwrap(), GlobalKey::Offset(4));
        assert_eq!(
            GlobalKey::parse("foo").unwrap(),
            GlobalKey::Name("foo".into())
        );
        assert!(GlobalKey::parse("99999").is_err());
    }

    #[test]
    fn read_crc_from_header() {
        let mut header = 6i32.to_le_bytes().to_vec();
        header.extend(1234i32.to_le_bytes());
        assert_eq!(read_crc(&header).unwrap(), 1234);
        assert!(read_crc(&header[..6]).is_err());
    }

    #[test]
    fn json_conversions() {
        use serde_json::json;

        assert_eq!(json_to_float(&json!(1.5)).unwrap(), 1.5);
        assert_eq!(json_to_float(&json!(true)).unwrap(), 1.);
        assert!(json_to_float(&json!("1")).is_err());

        assert_eq!(
            json_to_vector(&json!([1, 2, 3])).unwrap(),
            Value::Vector([1., 2., 3.].into())
        );
        assert!(json_to_vector(&json!([1, 2])).is_err());
        assert!(json_to_vector(&json!(1)).is_err());

        assert_eq!(
            json_to_string(&json!("hi")).unwrap(),
            Value::try_from("hi").unwrap()
        );
        assert!(json_to_string(&json!(1)).is_err());
    }
}
