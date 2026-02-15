pub mod functions;
pub mod globals;

use std::{cmp::Ordering, ffi::CStr, fmt, ops::Deref, sync::Arc};

use crate::{
    HashMap,
    userdata::{DynEq as _, EntityHandle},
};
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use glam::Vec3;
use num::FromPrimitive;
use num_derive::FromPrimitive;

use crate::{
    entity::ScalarFieldDef,
    userdata::{ErasedEntityHandle, ErasedFunction},
};

#[derive(Clone, Debug)]
pub struct StringTable {
    /// Interned string data. In the `progs.dat` all the strings are concatenated together,
    /// delineated by `\0`, but since we handle strings using `Arc` we still need to map
    /// the raw byte offset in the lump to something we can cheaply clone.
    strings: HashMap<i32, Arc<CStr>>,
}

impl StringTable {
    pub fn new<D: AsRef<[u8]>>(data: D) -> anyhow::Result<StringTable> {
        let data = data.as_ref();
        let mut offset = 0;
        let mut strings = HashMap::default();

        while !data.is_empty() {
            // TODO: Error handling.
            let next = CStr::from_bytes_until_nul(&data[offset..])?;
            let next_len = next.count_bytes();
            strings.insert(offset.try_into()?, next.into());
            offset += next_len;
        }

        Ok(Self { strings })
    }

    pub fn get<I>(&self, id: I) -> anyhow::Result<Arc<CStr>>
    where
        I: Into<StringRef>,
    {
        match id.into() {
            StringRef::Id(id) => self
                .strings
                .get(&id)
                .cloned()
                .ok_or_else(|| anyhow::Error::msg(format!("No string with id {id}"))),
            StringRef::Temp(lit) => Ok(lit.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Ptr(pub(crate) i32);

impl fmt::Display for Ptr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "*{}", self.0)
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[repr(u8)]
pub enum VmScalarType {
    Void = 0,
    String = 1,
    Float = 2,
    // Vector = 3, see `Type`.
    Entity = 4,

    FieldRef = 5,
    Function = 6,
    GlobalRef = 7,

    /// Fake type, only used internally
    EntityFieldRef = 255,
}

impl FromPrimitive for VmScalarType {
    fn from_i64(n: i64) -> Option<Self> {
        match n {
            0 => Some(Self::Void),
            1 => Some(Self::String),
            2 => Some(Self::Float),
            4 => Some(Self::Entity),

            5 => Some(Self::FieldRef),
            6 => Some(Self::Function),
            7 => Some(Self::GlobalRef),
            // `EntityFieldRef` invalid when converting from byte
            _ => None,
        }
    }

    fn from_u64(n: u64) -> Option<Self> {
        match n {
            0 => Some(Self::Void),
            1 => Some(Self::String),
            2 => Some(Self::Float),
            4 => Some(Self::Entity),

            5 => Some(Self::FieldRef),
            6 => Some(Self::Function),
            7 => Some(Self::GlobalRef),
            // `EntityFieldRef` invalid when converting from byte
            _ => None,
        }
    }
}

const VECTOR_TAG: u8 = 3;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[repr(u8)]
pub enum VmType {
    Scalar(VmScalarType),
    Vector = VECTOR_TAG,
}

impl From<VmScalarType> for VmType {
    fn from(value: VmScalarType) -> Self {
        Self::Scalar(value)
    }
}

impl VmType {
    pub fn num_elements(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector => 3,
        }
    }
}

impl TryFrom<VmType> for VmScalarType {
    type Error = [(VmScalarType, VectorField); 3];

    fn try_from(value: VmType) -> Result<Self, Self::Error> {
        match value {
            VmType::Scalar(scalar) => Ok(scalar),
            VmType::Vector => Err(VectorField::FIELDS.map(|fld| (VmScalarType::Float, fld))),
        }
    }
}

impl FromPrimitive for VmType {
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
            VmScalarType::from_u8(n).map(Self::Scalar)
        }
    }
}

impl fmt::Display for VmScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::String => write!(f, "string"),
            Self::Float => write!(f, "float"),
            Self::Entity => write!(f, "entity"),
            Self::FieldRef => write!(f, "field"),
            Self::Function => write!(f, "function"),
            Self::GlobalRef => write!(f, "pointer"),
            Self::EntityFieldRef => write!(f, "entityfield"),
        }
    }
}

impl fmt::Display for VmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Vector => write!(f, "vector"),
        }
    }
}

/// An individual field of a vector.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, FromPrimitive)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
pub enum VectorField {
    /// The x component.
    #[default]
    X = 0,
    /// The y component.
    Y = 1,
    /// The z component.
    Z = 2,
}

impl VectorField {
    /// All fields of a vector.
    pub const FIELDS: [Self; 3] = [VectorField::X, VectorField::Y, VectorField::Z];
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldName {
    // TODO: `CStr` is a pain in the ass to use, we should also support regular strs even if we internally store CStr
    pub name: Arc<CStr>,
    /// For vectors, we want to take the "raw" global definition
    /// and make each field addressable individually, so we only
    /// have to store scalars.
    pub offset: Option<VectorField>,
}

impl From<Arc<CStr>> for FieldName {
    fn from(name: Arc<CStr>) -> Self {
        Self { name, offset: None }
    }
}

/// A definition for a global in the `progs.dat`
#[derive(Debug, Clone)]
pub struct GlobalDef {
    /// Whether the global should be saved in a savegame
    // TODO: Implement this (should we conflate "save" with "persists between frames"?)
    pub save: bool,
    /// The type of the global
    pub type_: VmType,
    /// The offset of the global
    pub offset: u16,
    /// The name of the global
    pub name: Arc<CStr>,
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
    /// The defined type of the field
    pub type_: VmType,
    /// The byte offset of the field (usually not used in host bindings)
    pub offset: u16,
    /// The name of the field
    pub name: Arc<CStr>,
}

impl FieldDef {
    /// Extract a scalar field def from a field def that is either a scalar or a vector
    pub fn to_scalar(&self) -> Result<ScalarFieldDef, [ScalarFieldDef; 3]> {
        match VmScalarType::try_from(self.type_) {
            Err(fields) => Err(fields.map(|(type_, field)| ScalarFieldDef {
                field: Some(field),
                def: self.clone(),
                type_,
            })),
            Ok(type_) => Ok(ScalarFieldDef {
                field: None,
                def: self.clone(),
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
                (VmType::Vector, VmType::Vector) => Ordering::Equal,
                (VmType::Vector, _) => Ordering::Less,
                (_, VmType::Vector) => Ordering::Greater,
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

/// Abstraction around `bevy_ecs::entity::Entity` that allows us to impl `Default` without
/// world access.
#[derive(Debug, Default, Clone)]
pub enum EntityRef {
    /// The zero value for an entity reference - it is invalid to read or write fields on
    /// this type (TODO: is this true?), but it can be useful to pass it as an argument to builtins.
    #[default]
    Worldspawn,
    /// We use `Entity` rather than an index here so entities that aren't managed by the VM
    /// can still be passed to QuakeC functions.
    Entity(ErasedEntityHandle),
}

impl EntityRef {
    /// Create an `EntityRef` from a `T: EntityHandle`.
    pub fn new<T: EntityHandle>(value: T) -> Self {
        Self::Entity(ErasedEntityHandle(value.to_erased()))
    }

    /// If the entity reference is not `Worldspawn`, return the inner entity.
    pub fn non_null(self) -> anyhow::Result<ErasedEntityHandle> {
        match self {
            Self::Worldspawn => anyhow::bail!("Tried to access fields on worldspawn entity"),
            Self::Entity(ent) => Ok(ent),
        }
    }

    /// If the reference is `Worldspawn`, then `true`. Otherwise false.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Worldspawn)
    }
}

impl PartialEq for EntityRef {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Worldspawn, Self::Worldspawn) => true,
            (Self::Entity(lhs), Self::Entity(rhs)) => lhs.dyn_eq(rhs),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum VmFunctionRef {
    /// A reference to a function that is statically known by QuakeC
    Ptr(Ptr),
    /// An inline reference to an external function.
    ///
    /// > TODO: This shouldn't be in the actual `VmScalar` type, it should be elsewhere
    Extern(Arc<dyn ErasedFunction>),
}

impl PartialEq for VmFunctionRef {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ptr(lhs), Self::Ptr(rhs)) => lhs == rhs,
            (Self::Extern(lhs), Self::Extern(rhs)) => lhs.dyn_eq(&**rhs),
            _ => false,
        }
    }
}

impl VmFunctionRef {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Ptr(Ptr::NULL))
    }
}

impl Default for VmFunctionRef {
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

// TODO: This could be very easily expressed with NaN-boxing
//
// Suggested scheme:
//
// 3 bit tag, 20 bit payload. 1 bit of 24-bit mantissa reserved for quiet bit, to ensure it
// doesn't get overwritten (as the quiet bit is allowed by the spec to be overwritten at
// any point)
//
// Tag:
// 0 - entity
// 1 - string (static)
// 2 - string (temp)
// 3 - function (global)
// 4 - function (extern)
// 5 - global (always inline)
// 6 - field (always inline)
// 7 - entity field
//
// Payload:
// static string, global function, global/field ref: see `Ref`.
// entity, temp string, extern function: index into per-type vector stored on VM context
//   TODO: This needs some kind of garbage collection system, which makes everything way more complicated.
// entity field: these can only be temps, so we can have a 4-bit index into a 16-element list which is cleared when the function
//               returns and a 16 bit field reference
#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) enum VmScalar {
    /// This can be converted to any of the other values, as it is just a general "0".
    #[default]
    Void,
    /// For all other variants,
    Float(f32),
    Entity(EntityRef),
    String(StringRef),
    Function(VmFunctionRef),
    Global(Ptr),
    Field(Ptr),
    /// Internal representation for `OP_ADDR`.
    ///
    /// For some reason this doesn't seem to bloat this type at all,
    /// it's 24 bytes either way. The inner values can't be extracted
    /// to a separate struct without increasing size even further,
    /// though.
    ///
    /// > TODO: NaN-boxing with SoA-style extra storage for larger variants
    EntityField(EntityRef, Ptr),
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) struct EntityField {
    pub entity: EntityRef,
    pub field: Ptr,
}

impl TryFrom<VmScalar> for EntityField {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::EntityField(entity, field) => Ok(EntityField { entity, field }),
            // TODO: `FieldRef` isn't really right for this.
            other => Err(ScalarCastError {
                expected: VmScalarType::FieldRef,
                found: other.type_(),
            }),
        }
    }
}

const _: [(); 24] = [(); std::mem::size_of::<VmScalar>()];

impl From<bool> for VmScalar {
    fn from(value: bool) -> Self {
        if value {
            Self::Float(1.)
        } else {
            Self::Float(0.)
        }
    }
}

impl From<f32> for VmScalar {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

impl From<EntityRef> for VmScalar {
    fn from(value: EntityRef) -> Self {
        Self::Entity(value)
    }
}

impl From<VmFunctionRef> for VmScalar {
    fn from(value: VmFunctionRef) -> Self {
        Self::Function(value)
    }
}

impl From<StringRef> for VmScalar {
    fn from(value: StringRef) -> Self {
        Self::String(value)
    }
}

impl From<FieldPtr> for VmScalar {
    fn from(value: FieldPtr) -> Self {
        Self::Field(value.0)
    }
}

impl From<GlobalPtr> for VmScalar {
    fn from(value: GlobalPtr) -> Self {
        Self::Global(value.0)
    }
}

impl TryFrom<VmScalar> for f32 {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::Float(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: VmScalarType::Float,
                found: value.type_(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, snafu::Snafu)]
pub enum IntoEntityError {
    WorldspawnAccessForbidden,
    TypeMismatch { types: ScalarCastError },
}

impl From<ScalarCastError> for IntoEntityError {
    fn from(value: ScalarCastError) -> Self {
        Self::TypeMismatch { types: value }
    }
}

impl TryFrom<VmScalar> for VmFunctionRef {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::Function(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: VmScalarType::Function,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<VmScalar> for EntityRef {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::Entity(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: VmScalarType::Entity,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<VmScalar> for StringRef {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::String(f) => Ok(f),
            _ => Err(ScalarCastError {
                expected: VmScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<VmScalar> for GlobalPtr {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::Global(p) => Ok(GlobalPtr(p)),
            _ => Err(ScalarCastError {
                expected: VmScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

impl TryFrom<VmScalar> for FieldPtr {
    type Error = ScalarCastError;

    fn try_from(value: VmScalar) -> Result<Self, Self::Error> {
        match value {
            VmScalar::Void => Ok(Default::default()),
            VmScalar::Field(p) => Ok(FieldPtr(p)),
            _ => Err(ScalarCastError {
                expected: VmScalarType::String,
                found: value.type_(),
            }),
        }
    }
}

const _: [(); 24] = [(); std::mem::size_of::<VmScalar>()];

impl VmScalar {
    pub fn is_null(&self) -> bool {
        match self {
            VmScalar::Void => true,
            VmScalar::Float(f) => *f != 0.,
            VmScalar::Entity(entity_ref) => entity_ref.is_null(),
            VmScalar::String(string_ref) => string_ref.is_null(),
            VmScalar::Function(function_ref) => function_ref.is_null(),
            VmScalar::Global(ptr) => ptr.is_null(),
            VmScalar::Field(ptr) => ptr.is_null(),
            VmScalar::EntityField(ent, fld) => ent.is_null() && fld.is_null(),
        }
    }

    pub fn type_(&self) -> VmScalarType {
        match self {
            VmScalar::Void => VmScalarType::Void,
            VmScalar::Float(_) => VmScalarType::Float,
            VmScalar::Entity(_) => VmScalarType::Entity,
            VmScalar::String(_) => VmScalarType::String,
            VmScalar::Function(_) => VmScalarType::Function,
            VmScalar::Global(_) => VmScalarType::GlobalRef,
            VmScalar::Field(_) => VmScalarType::FieldRef,
            VmScalar::EntityField(..) => VmScalarType::EntityFieldRef,
        }
    }

    pub fn try_from_bytes(ty: VmScalarType, bytes: [u8; 4]) -> anyhow::Result<Self> {
        match ty {
            VmScalarType::Void => {
                if bytes == [0; 4] {
                    Ok(VmScalar::Void)
                } else {
                    Err(anyhow::Error::msg("`void` can only be initialized with 0"))
                }
            }
            VmScalarType::Float => Ok(VmScalar::Float(f32::from_le_bytes(bytes))),
            VmScalarType::String => Ok(VmScalar::String(StringRef::Id(i32::from_le_bytes(bytes)))),
            VmScalarType::Entity => {
                if bytes == [0; 4] {
                    Ok(VmScalar::Entity(EntityRef::Worldspawn))
                } else {
                    Err(anyhow::Error::msg(
                        "Cannot literally initialise an entity to any value other than worldspawn (no entities have been spawned at load-time)",
                    ))
                }
            }
            VmScalarType::Function => Ok(VmScalar::Function(VmFunctionRef::Ptr(Ptr(
                i32::from_le_bytes(bytes),
            )))),

            VmScalarType::FieldRef => Ok(VmScalar::Field(Ptr(i32::from_le_bytes(bytes)))),
            VmScalarType::GlobalRef => Ok(VmScalar::Global(Ptr(i32::from_le_bytes(bytes)))),

            VmScalarType::EntityFieldRef => Err(anyhow::Error::msg(
                "`entityfield` is invalid in literal instantiation",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum VmValue {
    Scalar(VmScalar),
    /// The only value in QuakeC that can be more than 32 bits (logically 32 bits, `Scalar` is larger
    /// because it's not just indices) is a vector of floats. This can only exist "ephemerally", as
    /// all values stored to/from stack or globals are ultimately scalars.
    Vector([f32; 3]),
}

impl Default for VmValue {
    fn default() -> Self {
        Self::Scalar(Default::default())
    }
}

impl From<VmValue> for [VmScalar; 3] {
    fn from(value: VmValue) -> Self {
        match value {
            VmValue::Scalar(scalar) => [scalar, VmScalar::Void, VmScalar::Void],
            VmValue::Vector(floats) => floats.map(Into::into),
        }
    }
}

impl From<[f32; 3]> for VmValue {
    fn from(value: [f32; 3]) -> Self {
        Self::Vector(value)
    }
}

impl From<Vec3> for VmValue {
    fn from(value: Vec3) -> Self {
        Self::Vector(value.into())
    }
}

impl TryFrom<[VmScalar; 3]> for VmValue {
    type Error = ScalarCastError;

    fn try_from(value: [VmScalar; 3]) -> Result<Self, Self::Error> {
        match value {
            [scalar, VmScalar::Void, VmScalar::Void] => Ok(VmValue::Scalar(scalar)),
            [x, y, z] => Ok(VmValue::Vector([
                x.try_into()?,
                y.try_into()?,
                z.try_into()?,
            ])),
        }
    }
}

impl<T> From<T> for VmValue
where
    T: Into<VmScalar>,
{
    fn from(value: T) -> Self {
        Self::Scalar(value.into())
    }
}

impl VmValue {
    pub fn type_(&self) -> VmType {
        match self {
            Self::Scalar(scalar) => VmType::Scalar(scalar.type_()),
            Self::Vector(_) => VmType::Vector,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, snafu::Snafu)]
pub(crate) struct ScalarCastError {
    pub expected: VmScalarType,
    pub found: VmScalarType,
}
