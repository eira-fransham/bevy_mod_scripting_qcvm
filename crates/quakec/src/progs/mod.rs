pub mod functions;
pub mod globals;

use std::{
    backtrace::Backtrace,
    cmp::Ordering,
    ffi::CStr,
    fmt,
    io::{Read, Seek, SeekFrom},
    ops::{Deref, Range},
    sync::Arc,
};

use arc_slice::ArcSlice;
use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::HashMap;
use num::FromPrimitive;
use num_derive::FromPrimitive;
use snafu::Snafu;
use tracing::debug;

use crate::{
    entity::{EntityError, EntityTypeDef, ScalarFieldInfo},
    progs::{
        functions::{ArgSize, FunctionRegistry, MAX_ARGS, QuakeCFunctionDef, Statement},
        globals::{GlobalRegistry, GlobalsError},
    },
    userdata::{ErasedBuiltin, ErasedEntity},
};

const VERSION: i32 = 6;
const CRC: i32 = 5927;
const LUMP_COUNT: usize = 6;
const SAVE_GLOBAL: u16 = 1 << 15;

// the on-disk size of a bytecode statement
const STATEMENT_SIZE: usize = 8;

// the on-disk size of a function declaration
const FUNCTION_SIZE: usize = 36;

// the on-disk size of a global or field definition
const DEF_SIZE: usize = 8;

#[derive(Clone, Debug)]
pub struct StringTable {
    /// Interned string data. In the `progs.dat` all the strings are concatenated together,
    /// delineated by `\0`, but since we handle strings using `Arc` we still need to map
    /// the raw byte offset in the lump to something we can cheaply clone.
    strings: HashMap<usize, Arc<CStr>>,
}

impl StringTable {
    pub fn new<D: AsRef<[u8]>>(data: D) -> anyhow::Result<StringTable> {
        let data = data.as_ref();
        let mut offset = 0;
        let mut strings = HashMap::new();

        while !data.is_empty() {
            // TODO: Error handling.
            let next = CStr::from_bytes_until_nul(&data[offset..])?;
            let next_len = next.count_bytes();
            strings.insert(offset, next.into());
            offset += next_len;
        }

        Ok(Self { strings })
    }

    pub fn get<I>(&self, id: I) -> anyhow::Result<Arc<CStr>>
    where
        I: Into<StringRef>,
    {
        match id.into() {
            StringRef::Id(id) => {
                let id: usize = id.try_into()?;
                self.strings
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| anyhow::Error::msg(format!("No string with id {id}")))
            }
            StringRef::Temp(lit) => Ok(lit.clone()),
        }
    }
}
#[derive(Snafu, Debug)]
pub enum ProgsError {
    #[snafu(context(false), display("{source:?}"))]
    Io {
        source: ::std::io::Error,
        backtrace: Backtrace,
    },
    #[snafu(context(false))]
    Globals {
        source: GlobalsError,
        backtrace: Backtrace,
    },
    #[snafu(context(false))]
    Entity {
        source: EntityError,
        backtrace: Backtrace,
    },
    CallStackOverflow {
        backtrace: Backtrace,
    },
    LocalStackOverflow {
        backtrace: Backtrace,
    },
    #[snafu(display("{message}"))]
    Other {
        message: String,
        backtrace: Backtrace,
    },
}

impl ProgsError {
    pub fn with_msg<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        ProgsError::Other {
            message: msg.into(),
            backtrace: Backtrace::capture(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ptr(pub(crate) i32);

impl fmt::Display for Ptr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "*{}", self.0)
    }
}

impl Default for Ptr {
    fn default() -> Self {
        Self(0)
    }
}

impl Ptr {
    pub const NULL: Self = Self(0);

    pub fn is_null(&self) -> bool {
        *self == Self::NULL
    }
}

impl TryFrom<usize> for Ptr {
    type Error = <usize as TryInto<i32>>::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

impl From<i32> for Ptr {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

impl From<i16> for Ptr {
    fn from(value: i16) -> Self {
        Self(value.into())
    }
}

enum LumpId {
    Statements = 0,
    GlobalDefs = 1,
    Fielddefs = 2,
    Functions = 3,
    Strings = 4,
    Globals = 5,
}

#[derive(Copy, Clone, Debug, FromPrimitive, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[repr(u8)]
pub enum ScalarType {
    Void = 0,
    String = 1,
    Float = 2,
    // Vector = 3, see `Type`.
    Entity = 4,

    FieldRef = 5,
    Function = 6,
    GlobalRef = 7,
}

const VECTOR_TAG: u8 = 3;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[repr(u8)]
pub enum Type {
    Scalar(ScalarType),
    Vector = 3,
}

impl Type {
    pub fn num_elements(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector => 3,
        }
    }
}

impl TryFrom<Type> for ScalarType {
    type Error = [(ScalarType, FieldOffset); 3];

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Scalar(scalar) => Ok(scalar),
            Type::Vector => Err(FieldOffset::FIELDS.map(|fld| (ScalarType::Float, fld))),
        }
    }
}

impl FromPrimitive for Type {
    fn from_i64(n: i64) -> Option<Self> {
        Self::from_u8(n.try_into().ok()?)
    }

    fn from_u64(n: u64) -> Option<Self> {
        Self::from_u8(n.try_into().ok()?)
    }
    fn from_u8(n: u8) -> Option<Self> {
        if n == VECTOR_TAG {
            Some(Self::Vector)
        } else {
            ScalarType::from_u8(n).map(Self::Scalar)
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::String => write!(f, "string"),
            Self::Float => write!(f, "float"),
            Self::Entity => write!(f, "entity"),
            Self::FieldRef => write!(f, "field"),
            Self::Function => write!(f, "function"),
            Self::GlobalRef => write!(f, "pointer"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Vector => write!(f, "vector"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Lump {
    offset: usize,
    count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, FromPrimitive)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
pub enum FieldOffset {
    X = 0,
    Y = 1,
    Z = 2,
}

impl FieldOffset {
    /// All fields of a vector.
    pub const FIELDS: [Self; 3] = [FieldOffset::X, FieldOffset::Y, FieldOffset::Z];
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldName {
    pub name: Arc<CStr>,
    /// For vectors, we want to take the "raw" global definition
    /// and make each field addressable individually, so we only
    /// have to store scalars.
    pub offset: Option<FieldOffset>,
}

impl From<Arc<CStr>> for FieldName {
    fn from(name: Arc<CStr>) -> Self {
        Self { name, offset: None }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalDef {
    // TODO: Implement this
    _save: bool,
    type_: Type,
    offset: u16,
    name: Arc<CStr>,
}

impl fmt::Display for GlobalDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} {} = {}",
            self.type_,
            self.name.to_string_lossy(),
            self.offset
        )
    }
}

/// An entity field definition.
///
/// These definitions can be used to look up entity fields by name. This is
/// required for custom fields defined in QuakeC code; their offsets are not
/// known at compile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
    pub type_: Type,
    pub offset: u16,
    pub name: Arc<CStr>,
}

impl FieldDef {
    pub fn to_scalar(&self) -> Result<ScalarFieldInfo, [ScalarFieldInfo; 3]> {
        match ScalarType::try_from(self.type_) {
            Err(fields) => Err(fields.map(|(type_, fld)| ScalarFieldInfo {
                name: FieldName {
                    name: self.name.clone(),
                    offset: Some(fld),
                },
                offset: fld as _,
                type_,
            })),
            Ok(type_) => Ok(ScalarFieldInfo {
                name: FieldName {
                    name: self.name.clone(),
                    offset: None,
                },
                offset: self.offset,
                type_,
            }),
        }
    }
}

impl PartialOrd for FieldDef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FieldDef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Put vector fields first, since we will deduplicate and vector fields
        // require fewer calls to host code.
        self.offset
            .cmp(&other.offset)
            .then(match (self.type_, other.type_) {
                (Type::Vector, Type::Vector) => Ordering::Equal,
                (Type::Vector, _) => Ordering::Less,
                (_, Type::Vector) => Ordering::Greater,
                _ => Ordering::Equal,
            })
            .then(self.name.cmp(&other.name))
    }
}

impl fmt::Display for FieldDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} = {}",
            self.type_,
            self.name.to_string_lossy(),
            self.offset
        )
    }
}

/// The values returned by loading a `progs.dat` file.
pub struct LoadProgs {
    pub globals: GlobalRegistry,
    pub entity_def: EntityTypeDef,
    /// We don't use string IDs internally any more, but we store the string table so that
    /// progs can load them.
    pub string_table: StringTable,
    pub function_defs: FunctionRegistry,
}

struct LoadFn {
    offset: i32,
    name: Arc<CStr>,
    source: Arc<CStr>,
    locals: Range<usize>,
    args: ArrayVec<ArgSize, MAX_ARGS>,
}

/// Loads all data from a `progs.dat` file.
///
/// This returns objects representing the necessary context to execute QuakeC bytecode.
pub fn load<R>(mut src: R) -> anyhow::Result<LoadProgs>
where
    R: Read + Seek,
{
    assert!(src.read_i32::<LittleEndian>()? == VERSION);
    assert!(src.read_i32::<LittleEndian>()? == CRC);

    let mut lumps = [Lump {
        offset: 0,
        count: 0,
    }; LUMP_COUNT];

    for (i, lump) in lumps.iter_mut().enumerate() {
        *lump = Lump {
            offset: src.read_i32::<LittleEndian>()? as usize,
            count: src.read_i32::<LittleEndian>()? as usize,
        };

        debug!("{:?}: {:?}", i, lump);
    }

    let ent_addr_count = src.read_i32::<LittleEndian>()? as usize;
    debug!("Field count: {}", ent_addr_count);

    // Read string data and construct StringTable

    let string_lump = &lumps[LumpId::Strings as usize];
    src.seek(SeekFrom::Start(string_lump.offset as u64))?;
    let mut strings = Vec::new();
    (&mut src)
        .take(string_lump.count as u64)
        .read_to_end(&mut strings)?;
    let string_table = StringTable::new(strings)?;

    assert_eq!(
        src.stream_position()?,
        src.seek(SeekFrom::Start(
            (string_lump.offset + string_lump.count) as u64,
        ))?
    );

    // Read function definitions and statements and construct Functions

    let statement_lump = &lumps[LumpId::Statements as usize];
    src.seek(SeekFrom::Start(statement_lump.offset as u64))?;
    let mut statements = Vec::with_capacity(statement_lump.count);
    for _ in 0..statement_lump.count {
        statements.push(Statement::new(
            src.read_i16::<LittleEndian>()?,
            src.read_i16::<LittleEndian>()?,
            src.read_i16::<LittleEndian>()?,
            src.read_i16::<LittleEndian>()?,
        )?);
    }

    assert_eq!(
        src.stream_position()?,
        src.seek(SeekFrom::Start(
            (statement_lump.offset + statement_lump.count * STATEMENT_SIZE) as u64,
        ))?
    );

    let statements: ArcSlice<[Statement]> = statements.into();

    let function_lump = &lumps[LumpId::Functions as usize];
    src.seek(SeekFrom::Start(function_lump.offset as u64))?;
    let mut load_functions = Vec::with_capacity(function_lump.count);

    for _ in 0..function_lump.count {
        let offset = src.read_i32::<LittleEndian>()?;

        let arg_start = usize::try_from(src.read_i32::<LittleEndian>()?)?;
        let locals = usize::try_from(src.read_i32::<LittleEndian>()?)?;

        // This is always 0
        let _ = src.read_i32::<LittleEndian>()?;

        let name_id = src.read_i32::<LittleEndian>()?;
        let srcfile_id = src.read_i32::<LittleEndian>()?;

        let name = string_table.get(name_id)?;
        let source = string_table.get(srcfile_id)?;

        let argc = src.read_i32::<LittleEndian>()?;
        let mut arg_size_buf = [0; MAX_ARGS];
        src.read_exact(&mut arg_size_buf)?;

        let mut args = ArrayVec::<ArgSize, MAX_ARGS>::new();

        for byte in &arg_size_buf[..argc as usize] {
            args.push(ArgSize::from_u8(*byte).unwrap());
        }

        load_functions.push(LoadFn {
            offset,
            name,
            locals: arg_start..arg_start + locals,
            source,
            args,
        });
    }

    assert_eq!(
        src.stream_position()?,
        src.seek(SeekFrom::Start(
            (function_lump.offset + function_lump.count * FUNCTION_SIZE) as u64,
        ))?
    );

    load_functions.sort_unstable_by_key(|def| def.offset);

    let function_defs = FunctionRegistry::new(statements, load_functions)?;

    let globaldef_lump = &lumps[LumpId::GlobalDefs as usize];
    src.seek(SeekFrom::Start(globaldef_lump.offset as u64))?;
    let mut global_defs = Vec::new();
    for _ in 0..globaldef_lump.count {
        let type_ = src.read_u16::<LittleEndian>()?;
        let offset = src.read_u16::<LittleEndian>()?;
        let name_id = src.read_i32::<LittleEndian>()?;
        let name = string_table.get(name_id)?;

        global_defs.push(GlobalDef {
            _save: type_ & SAVE_GLOBAL != 0,
            type_: Type::from_u16(type_ & !SAVE_GLOBAL).unwrap(),
            offset,
            name,
        });
    }

    global_defs.sort_by_key(|def| def.offset);

    assert_eq!(
        src.stream_position()?,
        src.seek(SeekFrom::Start(
            (globaldef_lump.offset + globaldef_lump.count * DEF_SIZE) as u64,
        ))?
    );

    let fielddef_lump = &lumps[LumpId::Fielddefs as usize];
    src.seek(SeekFrom::Start(fielddef_lump.offset as u64))?;
    let mut field_defs = Vec::new();
    for _ in 0..fielddef_lump.count {
        let type_ = src.read_u16::<LittleEndian>()?;
        let offset = src.read_u16::<LittleEndian>()?;
        let name_id = src.read_i32::<LittleEndian>()?;

        let name = string_table.get(name_id)?;

        if type_ & SAVE_GLOBAL != 0 {
            return Err(anyhow::Error::msg(
                "Save flag not allowed in field definitions",
            ));
        }
        field_defs.push(FieldDef {
            type_: Type::from_u16(type_).unwrap(),
            offset,
            name,
        });
    }

    assert_eq!(
        src.stream_position()?,
        src.seek(SeekFrom::Start(
            (fielddef_lump.offset + fielddef_lump.count * DEF_SIZE) as u64,
        ))?
    );

    let globals_lump = &lumps[LumpId::Globals as usize];
    src.seek(SeekFrom::Start(globals_lump.offset as u64))?;

    // if globals_lump.count < GLOBAL_STATIC_COUNT {
    //     return Err(ProgsError::with_msg(
    //         "Global count lower than static global count",
    //     ));
    // }

    let mut global_values = vec![0; globals_lump.count * 4];
    src.read_exact(&mut global_values)?;

    let globals = GlobalRegistry::new(global_defs, &global_values)?;

    let entity_def = EntityTypeDef::new(field_defs.into_boxed_slice())?;

    Ok(LoadProgs {
        globals,
        entity_def,
        string_table,
        function_defs,
    })
}

/// Abstraction around `bevy_ecs::entity::Entity` that allows us to impl `Default` without
/// world access.
#[derive(Debug, Default, Clone)]
pub enum EntityRef {
    #[default]
    Worldspawn,
    /// We use `Entity` rather than an index here so entities that aren't managed by the VM
    /// can still be passed to QuakeC functions.
    Entity(Arc<dyn ErasedEntity>),
}

impl PartialEq for EntityRef {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Worldspawn, Self::Worldspawn) => true,
            (Self::Entity(lhs), Self::Entity(rhs)) => lhs.dyn_eq(&**rhs),
            _ => false,
        }
    }
}

impl EntityRef {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Worldspawn)
    }
}

#[derive(Debug, Clone)]
pub enum FunctionRef {
    /// A reference to a function that is statically known by QuakeC
    Ptr(Ptr),
    /// An inline reference to an external function.
    Extern(Arc<dyn ErasedBuiltin>),
}

impl PartialEq for FunctionRef {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ptr(lhs), Self::Ptr(rhs)) => lhs == rhs,
            (Self::Extern(lhs), Self::Extern(rhs)) => lhs.dyn_eq(&**rhs),
            _ => false,
        }
    }
}

pub enum FunctionKind {
    QuakeC(QuakeCFunctionDef),
    Extern(Arc<dyn ErasedBuiltin>),
}

impl FunctionRef {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Ptr(Ptr::NULL))
    }
}

impl Default for FunctionRef {
    fn default() -> Self {
        Self::Ptr(Default::default())
    }
}

/// Separate type from `Ref` in order to prevent them being confused, as the `Arc<CStr>` variant
/// of `StringRef` is not the name, it's the actual value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StringRef {
    Id(i32),
    Temp(Arc<CStr>),
}

impl From<i32> for StringRef {
    fn from(value: i32) -> Self {
        Self::Id(value)
    }
}

impl From<Arc<CStr>> for StringRef {
    fn from(value: Arc<CStr>) -> Self {
        Self::Temp(value)
    }
}

impl Default for StringRef {
    fn default() -> Self {
        Self::Id(0)
    }
}

impl StringRef {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Id(0))
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct GlobalPtr(pub Ptr);

impl Deref for GlobalPtr {
    type Target = Ptr;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct FieldPtr(pub Ptr);

impl Deref for FieldPtr {
    type Target = Ptr;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub enum Scalar {
    /// This can be converted to any of the other values, as it is just a general "0".
    #[default]
    Void,
    /// For all other variants,
    Float(f32),
    Entity(EntityRef),
    String(StringRef),
    Function(FunctionRef),
    Global(Ptr),
    Field(Ptr),
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        if value {
            Self::Float(1.)
        } else {
            Self::Float(0.)
        }
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

impl TryFrom<Scalar> for f32 {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::Float(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: ScalarType::Float,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<Scalar> for FunctionRef {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::Function(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: ScalarType::Function,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<Scalar> for EntityRef {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::Entity(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: ScalarType::Entity,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<Scalar> for StringRef {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::String(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: ScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<Scalar> for GlobalPtr {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::Global(p) => Ok(GlobalPtr(p)),
            _ => Err(ScalarCastError {
                expected: ScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<Scalar> for FieldPtr {
    type Error = ScalarCastError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        match value {
            Scalar::Void => Ok(Default::default()),
            Scalar::Field(p) => Ok(FieldPtr(p)),
            _ => Err(ScalarCastError {
                expected: ScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

const _: [(); 24] = [(); std::mem::size_of::<Scalar>()];

impl Scalar {
    pub fn is_null(&self) -> bool {
        match self {
            Scalar::Void => true,
            Scalar::Float(f) => *f != 0.,
            Scalar::Entity(entity_ref) => entity_ref.is_null(),
            Scalar::String(string_ref) => string_ref.is_null(),
            Scalar::Function(function_ref) => function_ref.is_null(),
            Scalar::Global(ptr) => ptr.is_null(),
            Scalar::Field(ptr) => ptr.is_null(),
        }
    }

    pub fn type_(&self) -> ScalarType {
        match self {
            Scalar::Void => ScalarType::Void,
            Scalar::Float(_) => ScalarType::Float,
            Scalar::Entity(_) => ScalarType::Entity,
            Scalar::String(_) => ScalarType::String,
            Scalar::Function(_) => ScalarType::Function,
            Scalar::Global(_) => ScalarType::GlobalRef,
            Scalar::Field(_) => ScalarType::FieldRef,
        }
    }

    pub fn try_from_bytes(ty: ScalarType, bytes: [u8; 4]) -> anyhow::Result<Self> {
        match ty {
            ScalarType::Void => {
                if bytes == [0; 4] {
                    Ok(Scalar::Void)
                } else {
                    Err(anyhow::Error::msg("`void` can only be initialized with 0"))
                }
            }
            ScalarType::Float => Ok(Scalar::Float(f32::from_le_bytes(bytes))),
            ScalarType::String => Ok(Scalar::String(StringRef::Id(i32::from_le_bytes(bytes)))),
            ScalarType::Entity => {
                if bytes == [0; 4] {
                    Ok(Scalar::Entity(EntityRef::Worldspawn))
                } else {
                    Err(anyhow::Error::msg(
                        "Cannot literally initialise an entity to any value other than worldspawn (no entities have been spawned at load-time)",
                    ))
                }
            }
            ScalarType::Function => Ok(Scalar::Function(FunctionRef::Ptr(Ptr(
                i32::from_le_bytes(bytes),
            )))),

            ScalarType::FieldRef => Ok(Scalar::Field(Ptr(i32::from_le_bytes(bytes).try_into()?))),
            ScalarType::GlobalRef => Ok(Scalar::Global(Ptr(i32::from_le_bytes(bytes).try_into()?))),
        }
    }
}

pub enum Value {
    Scalar(Scalar),
    /// The only value in QuakeC that can be more than 32 bits (logically 32 bits, `Scalar` is larger
    /// because it's not just indices) is a vector of floats. This can only exist "ephemerally", as
    /// all values stored to/from stack or globals are ultimately scalars.
    Vector([f32; 3]),
}

impl From<[f32; 3]> for Value {
    fn from(value: [f32; 3]) -> Self {
        Self::Vector(value)
    }
}

impl<T> From<T> for Value
where
    T: Into<Scalar>,
{
    fn from(value: T) -> Self {
        Self::Scalar(value.into())
    }
}

impl Value {
    pub fn type_(&self) -> Type {
        match self {
            Self::Scalar(scalar) => Type::Scalar(scalar.type_()),
            Self::Vector(_) => Type::Vector,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, snafu::Snafu)]
pub struct ScalarCastError {
    pub expected: ScalarType,
    pub found: ScalarType,
}
