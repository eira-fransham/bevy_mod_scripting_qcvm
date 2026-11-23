//! Runs a function from a compiled QuakeC `progs.dat`.
//!
//! ```sh
//! run-qc --globals '{ "foo": 1, "bar": "Hello, world!" }' my-progs.dat my_func 1 2 3
//! ```
//!
//! See `run-qc --help` for the ways that functions and globals can be specified.

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    io::Cursor,
    num::NonZeroU64,
    path::PathBuf,
    sync::Arc,
};

use anyhow::{Context as _, bail, format_err};
use clap::Parser;
use qcvm::{
    BuiltinDef, EntityRef, EntityTypeDef, ErasedAddrInfo, ErasedDefs, ErasedEntityHandle, FieldDef,
    Function, FunctionDef, FunctionIndex, GlobalDef, QcVm, Type, Value, VmScalarType, VmType,
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

    /// Print the globals after running the function, as `{ "globals": ..., "return": ... }`.
    ///
    /// Globals are keyed by name, or by offset if they have no name or their name is ambiguous.
    /// Locals and immediates are left out.
    #[arg(long)]
    print_globals: bool,

    /// The compiled QuakeC bytecode.
    progs: PathBuf,

    /// The function to run: a function name (`my_func`), a function index (`#4`, which must be
    /// quoted in most shells), or a global holding a function, by name (`.main`) or by offset
    /// (`.40`).
    function: String,

    /// The arguments to pass to the function, each as JSON.
    ///
    /// Numbers (and bools) are floats, `[x, y, z]` is a vector, `null` is `void` (e.g. a null
    /// entity or string), and an object is a new entity with the given fields, with the same system
    /// for specifying keys as for globals, e.g. `{ "foo": 1, "40", "Hello" }` for the field named "foo"
    /// and the field at offset 40, respectively. Anything that doesn't parse as JSON is interpreted
    /// as a string. If you want to pass something that would be interpreted as JSON as a string
    /// instead, wrap it in quotes to turn it into a JSON string, e.g. `run-qc foo.dat some_fn '"0"'`
    /// to pass `"0"` as a string (it would otherwise be interpreted as a float).
    #[arg(allow_negative_numbers = true)]
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

/// A function specified on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FuncSelector {
    /// `my_func`
    Name(String),
    /// `#4`
    Index(i32),
    /// `.main` or `.40`
    Global(DefKey),
}

/// A global or entity field specified by name or offset.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DefKey {
    Name(String),
    Offset(u16),
}

impl DefKey {
    fn parse(s: &str) -> anyhow::Result<Self> {
        if s.is_empty() {
            bail!("Empty name");
        }

        Ok(match s.parse::<u16>() {
            Ok(offset) => Self::Offset(offset),
            Err(_) if s.starts_with(|c: char| c.is_ascii_digit()) => {
                bail!("Invalid offset `{s}`")
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
            Ok(Self::Global(DefKey::parse(global)?))
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

/// The fields of an entity, by offset. Vector fields are stored as their three components.
type EntityFields = HashMap<u16, Value>;

/// A host that provides no builtins or host globals, and stores entities created from the
/// command line.
#[derive(Default)]
struct Host {
    /// Entity `n` has handle `n + 1`, as handle 0 is the null entity.
    entities: Vec<EntityFields>,
}

impl Host {
    /// Creates an entity with no fields set.
    fn spawn(&mut self) -> ErasedEntityHandle {
        self.entities.push(EntityFields::default());
        NonZeroU64::new(self.entities.len() as u64).unwrap()
    }

    fn fields(&self, ent: EntityRef) -> anyhow::Result<&EntityFields> {
        let ent = ent.ok_or_else(|| format_err!("Accessed a field of the null entity"))?;
        self.entities
            .get(ent.get() as usize - 1)
            .ok_or_else(|| format_err!("Invalid entity {ent}"))
    }

    fn fields_mut(&mut self, ent: EntityRef) -> anyhow::Result<&mut EntityFields> {
        let ent = ent.ok_or_else(|| format_err!("Accessed a field of the null entity"))?;
        self.entities
            .get_mut(ent.get() as usize - 1)
            .ok_or_else(|| format_err!("Invalid entity {ent}"))
    }

    /// Reads a single scalar of an entity, which is zero if it hasn't been set.
    fn get_scalar(
        fields: &EntityFields,
        offset: u16,
        entity_def: &EntityTypeDef,
    ) -> Result<Value, AddrError<anyhow::Error>> {
        let def = entity_def.get(offset).ok_or(AddrError::OutOfRange)?;
        Ok(fields
            .get(&offset)
            .cloned()
            .unwrap_or_else(|| Value::zero(Type::try_from(def.type_).unwrap_or(Type::Void))))
    }

    fn entity_get(
        &self,
        ent: EntityRef,
        offset: u16,
        ty: Type,
        entity_def: &EntityTypeDef,
    ) -> Result<Value, AddrError<anyhow::Error>> {
        let fields = self.fields(ent)?;

        if ty != Type::Vector {
            return Self::get_scalar(fields, offset, entity_def);
        }

        let mut components = [0.; 3];
        for (i, component) in components.iter_mut().enumerate() {
            let value = Self::get_scalar(fields, offset + i as u16, entity_def)?;
            *component = value
                .or_zero(Type::Float)
                .try_into()
                .map_err(|e: anyhow::Error| e.context(format!("Field {offset} isn't a vector")))?;
        }

        Ok(Value::Vector(components.into()))
    }

    fn entity_set(
        &mut self,
        ent: EntityRef,
        offset: u16,
        value: Value,
        entity_def: &EntityTypeDef,
    ) -> Result<(), AddrError<anyhow::Error>> {
        let scalars = match value {
            Value::Vector(vec) => vec.to_array().map(Value::Float).to_vec(),
            value => vec![value],
        };

        let fields = self.fields_mut(ent)?;
        for (i, value) in scalars.into_iter().enumerate() {
            let offset = offset + i as u16;
            entity_def.get(offset).ok_or(AddrError::OutOfRange)?;
            match value {
                // `void` is the zero value of whatever it is stored into.
                Value::Void => fields.remove(&offset),
                value => fields.insert(offset, value),
            };
        }

        Ok(())
    }
}

impl ErasedContext for Host {
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
        erased_ent: EntityRef,
        field: u16,
        ty: Type,
        entity_def: &EntityTypeDef,
    ) -> anyhow::Result<Value, AddrError<anyhow::Error>> {
        self.entity_get(erased_ent, field, ty, entity_def)
    }

    fn dyn_entity_set(
        &mut self,
        erased_ent: EntityRef,
        field: u16,
        value: Value,
        entity_def: &EntityTypeDef,
    ) -> anyhow::Result<(), AddrError<anyhow::Error>> {
        self.entity_set(erased_ent, field, value, entity_def)
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

/// A global or entity field definition.
trait Def {
    fn type_(&self) -> VmType;
    fn offset(&self) -> u16;
    fn name(&self) -> &CStr;

    /// Whether the def is typed, i.e. tells us how to interpret the value.
    fn is_typed(&self) -> bool {
        self.type_() != VmType::Scalar(VmScalarType::Void)
    }

    fn same_as(&self, other: &Self) -> bool {
        self.offset() == other.offset()
            && self.type_() == other.type_()
            && self.name() == other.name()
    }
}

impl Def for GlobalDef {
    fn type_(&self) -> VmType {
        self.type_
    }

    fn offset(&self) -> u16 {
        self.offset
    }

    fn name(&self) -> &CStr {
        &self.name
    }
}

impl Def for FieldDef {
    fn type_(&self) -> VmType {
        self.type_
    }

    fn offset(&self) -> u16 {
        self.offset
    }

    fn name(&self) -> &CStr {
        &self.name
    }
}

/// Finds the definition matching `key`. If several defs match (e.g. a vector and its `_x`
/// component share an offset), `prefer_vector` chooses between them.
fn find_def<'a, D: Def>(defs: &'a [D], key: &DefKey, prefer_vector: bool) -> Option<&'a D> {
    let matching = || {
        defs.iter().filter(move |def| match key {
            DefKey::Offset(offset) => def.offset() == *offset,
            DefKey::Name(name) => def.name().to_bytes() == name.as_bytes(),
        })
    };
    let typed = || matching().filter(|def| def.is_typed());

    typed()
        .find(|def| (def.type_() == VmType::Vector) == prefer_vector)
        .or_else(|| typed().next())
        .or_else(|| matching().next())
}

/// Finds the offset and (if known) the definition of a global (see [`find_def`]).
fn resolve_global<'a>(
    vm: &'a QcVm,
    key: &DefKey,
    prefer_vector: bool,
) -> anyhow::Result<(u16, Option<&'a GlobalDef>)> {
    let def = find_def(vm.global_defs(), key, prefer_vector);

    match key {
        DefKey::Offset(offset) => {
            // Check that the offset exists, as `set_global` can't tell us which global was wrong.
            vm.global(*offset)
                .with_context(|| format!("No global at offset {offset}"))?;
            Ok((*offset, def))
        }
        DefKey::Name(name) => {
            let def = def.ok_or_else(|| format_err!("No global named `{name}`"))?;
            Ok((def.offset, Some(def)))
        }
    }
}

/// Finds the definition of an entity field (see [`find_def`]).
fn resolve_field<'a>(
    vm: &'a QcVm,
    key: &DefKey,
    prefer_vector: bool,
) -> anyhow::Result<&'a FieldDef> {
    find_def(vm.field_defs(), key, prefer_vector).ok_or_else(|| match key {
        DefKey::Offset(offset) => format_err!("No entity field at offset {offset}"),
        DefKey::Name(name) => format_err!("No entity field named `{name}`"),
    })
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
    if let Some(def) = def.filter(|def| def.is_typed())
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

/// Creates an entity with the fields specified by the JSON object `fields`.
fn json_to_entity(
    vm: &QcVm,
    host: &mut Host,
    fields: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<Value> {
    let ent = host.spawn();

    for (key, json) in fields {
        (|| {
            let def = resolve_field(vm, &DefKey::parse(key)?, json.is_array())?;
            let value = json_to_value(vm, host, Some(def.type_), json)?;
            host.entity_set(Some(ent), def.offset, value, vm.entity_def())
                .map_err(|e| match e {
                    AddrError::OutOfRange => format_err!("Field {} is out of range", def.offset),
                    AddrError::Other { error } => error,
                })
        })()
        .with_context(|| format!("Couldn't set entity field `{key}`"))?;
    }

    Ok(Value::Entity(Some(ent)))
}

/// Converts a JSON value to a value of type `ty`, or a type inferred from the JSON if `ty` is
/// unknown or untyped.
///
/// `null` is the zero value of `ty`, or `void` if the type isn't known.
fn json_to_value(
    vm: &QcVm,
    host: &mut Host,
    ty: Option<VmType>,
    json: &serde_json::Value,
) -> anyhow::Result<Value> {
    use serde_json::Value as Json;

    let ty = ty.filter(|ty| *ty != VmType::Scalar(VmScalarType::Void));

    if json.is_null() {
        return match ty {
            Some(ty) => Ok(Value::zero(Type::try_from(ty)?)),
            None => Ok(Value::Void),
        };
    }

    match ty {
        Some(VmType::Vector) => json_to_vector(json),
        Some(VmType::Scalar(VmScalarType::Float)) => json_to_float(json).map(Value::Float),
        Some(VmType::Scalar(VmScalarType::String)) => json_to_string(json),
        Some(VmType::Scalar(VmScalarType::Function)) => json_to_function(vm, json),
        Some(VmType::Scalar(VmScalarType::Entity)) => match json {
            Json::Object(fields) => json_to_entity(vm, host, fields),
            other => bail!("Expected an entity (a JSON object), found {other}"),
        },
        Some(other) => bail!("Values of type {other} aren't supported yet"),
        None => match json {
            Json::Null => Ok(Value::Void),
            Json::Number(_) | Json::Bool(_) => json_to_float(json).map(Value::Float),
            Json::Array(_) => json_to_vector(json),
            Json::String(_) => json_to_string(json),
            Json::Object(fields) => json_to_entity(vm, host, fields),
        },
    }
}

/// Parses a command-line argument as JSON, or as a string if it isn't valid JSON.
fn parse_arg(arg: &str) -> serde_json::Value {
    serde_json::from_str(arg).unwrap_or_else(|_| serde_json::Value::String(arg.to_owned()))
}

/// Sets the globals specified by the JSON object `json`.
fn apply_globals(vm: &QcVm, host: &mut Host, json: &str) -> anyhow::Result<()> {
    let json: serde_json::Value = serde_json::from_str(json).context("Invalid globals JSON")?;
    let serde_json::Value::Object(globals) = json else {
        bail!("Globals must be a JSON object, found {json}");
    };

    for (key, json) in &globals {
        (|| {
            let key = DefKey::parse(key)?;
            let (offset, def) = resolve_global(vm, &key, json.is_array())?;
            let value = json_to_value(vm, host, def.map(|def| def.type_), json)?;
            vm.set_global(offset, value)
        })()
        .with_context(|| format!("Couldn't set global `{key}`"))?;
    }

    Ok(())
}

/// The JSON key for `def`, following the same rules as parsing keys: its name if that refers back
/// to `def`, and otherwise its offset.
fn def_key<D: Def>(defs: &[D], def: &D) -> String {
    let name = def.name().to_string_lossy();
    let resolves_to_def = matches!(DefKey::parse(&name), Ok(DefKey::Name(_)))
        && find_def(
            defs,
            &DefKey::Name(name.clone().into_owned()),
            def.type_() == VmType::Vector,
        )
        .is_some_and(|found| found.same_as(def));

    if resolves_to_def {
        name.into_owned()
    } else {
        def.offset().to_string()
    }
}

/// Converts a float to JSON, using the shortest representation that round-trips as an `f32`.
/// JSON has no NaN or infinity, so those are strings (e.g. `"NaN"`, `"inf"`).
fn float_to_json(f: f32) -> serde_json::Value {
    f.to_string()
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map(serde_json::Value::Number)
        .unwrap_or_else(|| serde_json::Value::String(f.to_string()))
}

/// Converts the fields of an entity to a JSON object, in the same format as entity arguments.
///
/// Only fields that have been set are included, as the rest are zero. `stack` is the entities
/// currently being converted, to detect cycles.
fn entity_to_json(
    vm: &QcVm,
    host: &Host,
    ent: ErasedEntityHandle,
    stack: &mut Vec<ErasedEntityHandle>,
) -> anyhow::Result<serde_json::Value> {
    if stack.contains(&ent) {
        bail!("Entity {ent} (indirectly) refers to itself, which can't be printed yet");
    }

    let entity_def = vm.entity_def();
    let fields = host.fields(Some(ent))?;

    // The whole fields (rather than vector components) that have been set, in offset order.
    let mut defs = fields
        .keys()
        .filter_map(|offset| entity_def.get(*offset))
        .map(|scalar| &scalar.def)
        .collect::<Vec<_>>();
    defs.sort_by_key(|def| def.offset);
    defs.dedup_by(|a, b| a.same_as(b));

    stack.push(ent);
    let mut out = serde_json::Map::new();
    for def in defs {
        let ty = match def.type_ {
            VmType::Vector => Type::Vector,
            _ => Type::AnyScalar,
        };
        let value = host
            .entity_get(Some(ent), def.offset, ty, entity_def)
            .map_err(|e| match e {
                AddrError::OutOfRange => format_err!("Field {} is out of range", def.offset),
                AddrError::Other { error } => error,
            })?;
        let json = value_to_json(vm, host, &value, stack)
            .with_context(|| format!("Couldn't print field {def}"))?;
        out.insert(def_key(vm.field_defs(), def), json);
    }
    stack.pop();

    Ok(serde_json::Value::Object(out))
}

/// Converts a value to JSON, in the same format as arguments and globals are given in.
fn value_to_json(
    vm: &QcVm,
    host: &Host,
    value: &Value,
    stack: &mut Vec<ErasedEntityHandle>,
) -> anyhow::Result<serde_json::Value> {
    use serde_json::Value as Json;

    Ok(match value {
        Value::Void | Value::Entity(None) => Json::Null,
        Value::Float(f) => float_to_json(*f),
        Value::Vector(v) => Json::Array(v.to_array().map(float_to_json).into()),
        Value::String(s) => Json::String(s.to_string_lossy().into_owned()),
        Value::Entity(Some(ent)) => entity_to_json(vm, host, *ent, stack)?,
        Value::Function(Function::Index(FunctionIndex(0))) => Json::Null,
        Value::Function(Function::Index(FunctionIndex(index))) => {
            // Use the name if it refers back to this function, like `json_to_function` accepts.
            let name = vm
                .functions()
                .get_by_index(*index)
                .ok()
                .map(|def| def.name.clone())
                .filter(|name| {
                    !name.is_empty()
                        && vm
                            .functions()
                            .get_by_name(name)
                            .is_ok_and(|def| def.index == *index)
                });
            match name {
                Some(name) => Json::String(name.to_string_lossy().into_owned()),
                None => Json::String(format!("#{index}")),
            }
        }
        Value::Function(Function::Host(_)) => bail!("Host functions can't be printed"),
        Value::Field(_) => bail!("Field references can't be printed yet"),
    })
}

/// The globals to print with `--print-globals`, and their keys.
///
/// This leaves out locals (which are stored per call, so their global slots are meaningless),
/// immediates, field definitions, vector components and untyped slots.
fn printed_globals(vm: &QcVm) -> Vec<(String, &GlobalDef)> {
    let locals = vm
        .functions()
        .iter()
        .filter_map(|def| {
            let body = def.body.clone().try_into_qc().ok()?;
            let args_size: usize = def
                .args
                .iter()
                .map(|ty| if *ty == Type::Vector { 3 } else { 1 })
                .sum();
            Some(body.locals.start..body.locals.end.max(body.locals.start + args_size))
        })
        .collect::<Vec<_>>();
    let vector_components = vm
        .global_defs()
        .iter()
        .filter(|def| def.type_ == VmType::Vector)
        .flat_map(|def| [def.offset + 1, def.offset + 2])
        .collect::<Vec<_>>();

    vm.global_defs()
        .iter()
        .filter(|def| {
            def.is_typed()
                && !matches!(
                    def.type_,
                    VmType::Scalar(
                        VmScalarType::FieldRef
                            | VmScalarType::GlobalRef
                            | VmScalarType::EntityFieldRef
                    )
                )
                && def.name.to_bytes() != b"IMMEDIATE"
                && !vector_components.contains(&def.offset)
                && !locals
                    .iter()
                    .any(|range| range.contains(&usize::from(def.offset)))
        })
        // A vector shares its offset with its `_x` component.
        .filter(|def| {
            find_def(vm.global_defs(), &DefKey::Offset(def.offset), true)
                .is_some_and(|found| found.same_as(def))
        })
        .map(|def| (def_key(vm.global_defs(), def), def))
        .collect()
}

/// Converts the globals to JSON for `--print-globals`.
fn globals_to_json(vm: &QcVm, host: &Host) -> anyhow::Result<serde_json::Value> {
    let mut out = serde_json::Map::new();
    for (key, def) in printed_globals(vm) {
        let value = match def.type_ {
            VmType::Vector => Value::Vector(vm.global_vector(def.offset)?),
            _ => vm.global(def.offset)?,
        };
        let json = value_to_json(vm, host, &value, &mut Vec::new())
            .with_context(|| format!("Couldn't print global {def}"))?;
        out.insert(key, json);
    }

    Ok(serde_json::Value::Object(out))
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

    let mut host = Host::default();

    if let Some(globals) = &args.globals {
        apply_globals(&vm, &mut host, globals)?;
    }

    let selector = FuncSelector::parse(&args.function)?;
    let function = resolve_function(&vm, &selector)?;
    let name = function.name.to_string_lossy();

    if function.is_builtin() {
        bail!("`{name}` is a builtin, which run-qc doesn't support");
    }

    let mut params = Vec::with_capacity(args.args.len());
    for (i, arg) in args.args.iter().enumerate() {
        let value = json_to_value(&vm, &mut host, None, &parse_arg(arg))
            .with_context(|| format!("Invalid argument {i} (`{arg}`)"))?;

        // Only the size of parameters is known, so this is the only check we can do.
        match (function.args.get(i), &value) {
            (Some(Type::Vector), Value::Vector(_) | Value::Void) => {}
            (Some(Type::Vector), _) => {
                bail!("Argument {i} of `{name}` is a vector, but `{arg}` isn't")
            }
            (Some(_), Value::Vector(_)) => {
                bail!("Argument {i} of `{name}` is a scalar, but `{arg}` is a vector")
            }
            _ => {}
        }

        params.push(value);
    }

    let ret = vm.run(&mut host, function.index, params)?;

    let ret = ret.vector().map(Value::Vector).or_else(|_| ret.scalar())?;

    let ret = value_to_json(&vm, &host, &ret, &mut Vec::new())
        .context("Couldn't print the return value")?;

    if args.print_globals {
        let globals = globals_to_json(&vm, &host)?;
        println!(
            "{}",
            serde_json::json!({ "globals": globals, "return": ret })
        );
    } else {
        println!("{ret}");
    }

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
            FuncSelector::Global(DefKey::Name("main".into()))
        );
        assert_eq!(
            FuncSelector::parse(".40").unwrap(),
            FuncSelector::Global(DefKey::Offset(40))
        );
        assert!(FuncSelector::parse("#").is_err());
        assert!(FuncSelector::parse("#foo").is_err());
        assert!(FuncSelector::parse(".").is_err());
        assert!(FuncSelector::parse(".4x").is_err());
        assert!(FuncSelector::parse("").is_err());
    }

    #[test]
    fn parse_global_keys() {
        assert_eq!(DefKey::parse("4").unwrap(), DefKey::Offset(4));
        assert_eq!(DefKey::parse("foo").unwrap(), DefKey::Name("foo".into()));
        assert!(DefKey::parse("99999").is_err());
    }

    #[test]
    fn read_crc_from_header() {
        let mut header = 6i32.to_le_bytes().to_vec();
        header.extend(1234i32.to_le_bytes());
        assert_eq!(read_crc(&header).unwrap(), 1234);
        assert!(read_crc(&header[..6]).is_err());
    }

    #[test]
    fn parse_args() {
        use serde_json::json;

        assert_eq!(parse_arg("1"), json!(1));
        assert_eq!(parse_arg("-1.5"), json!(-1.5));
        assert_eq!(parse_arg("[1, 2, 3]"), json!([1, 2, 3]));
        assert_eq!(parse_arg("null"), json!(null));
        assert_eq!(parse_arg(r#"{ "health": 1 }"#), json!({ "health": 1 }));
        assert_eq!(parse_arg(r#""0""#), json!("0"));
        assert_eq!(parse_arg("hello"), json!("hello"));
        assert_eq!(parse_arg("[1, 2"), json!("[1, 2"));
    }

    #[test]
    fn float_output() {
        use serde_json::json;

        assert_eq!(float_to_json(1.), json!(1.0));
        assert_eq!(float_to_json(0.1), json!(0.1));
        assert_eq!(float_to_json(-2.5), json!(-2.5));
        assert_eq!(float_to_json(f32::NAN), json!("NaN"));
        assert_eq!(float_to_json(f32::INFINITY), json!("inf"));
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
