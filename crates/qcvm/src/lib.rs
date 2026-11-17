#![deny(missing_docs)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

use std::{
    any::Any,
    cell::Cell,
    collections::VecDeque,
    ffi::{CStr, CString},
    fmt,
    num::{NonZeroIsize, NonZeroU64},
    ops::{ControlFlow, Range},
    str::FromStr,
    sync::Arc,
};

use anyhow::Context as _;
use arrayvec::ArrayVec;
use bump_scope::{Bump, BumpScope};
use glam::Vec3;
use num::FromPrimitive as _;
use snafu::Snafu;
use strum::VariantArray;

use crate::{
    load::Progs,
    progs::{
        StringRef, StringTable, VmFunctionRef, VmScalar, VmValue,
        functions::{ArgSize, FunctionExecutionCtx, QCFunctionDef, Statement},
        globals::GlobalRegistry,
    },
    userdata::{AddrError, ErasedContext, ErasedFunction, FnCall, QCType},
};

mod entity;
mod load;
mod ops;
mod progs;
pub mod userdata;

#[cfg(test)]
mod tests;

pub use arrayvec;
pub use strum;

pub use anyhow::{self, Error};

pub use crate::{
    entity::{EntityTypeDef, ScalarFieldDef},
    progs::{
        EntityRef, FieldDef, GlobalDef, VectorField, VmScalarType, VmType,
        functions::{Builtin, BuiltinDef, FunctionDef, FunctionRegistry, MAX_ARGS},
        is_null_string, null_string,
    },
};

#[cfg(feature = "quake1")]
pub mod quake1;

/// A type-erased entity handle.
pub type ErasedEntityHandle = NonZeroU64;

/// Trait for global or field addresses
pub trait Address: strum::VariantArray + Clone + Send + Sync + Sized + 'static {
    /// For vector globals, returns the component fields. For scalars, just returns `self`.
    fn fields(&self) -> Option<[(Self, VectorField); 3]> {
        fn float_field<A: Address>(addr: u16, field: VectorField) -> Option<(A, VectorField)> {
            if let Some(addr) = A::from_u16_typed(addr + field as u16, Type::Float) {
                Some((addr, field))
            } else {
                None
            }
        }

        match self.type_() {
            Type::Vector => {
                let addr = self.to_u16();
                let [x, y, z] = VectorField::FIELDS;
                let [Some(x), Some(y), Some(z)] = [
                    float_field::<Self>(addr, x),
                    float_field::<Self>(addr, y),
                    float_field::<Self>(addr, z),
                ] else {
                    return None;
                };

                Some([x, y, z])
            }
            _ => None,
        }
    }

    /// For addresses that are a component of a vector, this returns the address of the vector itself along with the field of the
    /// vector that it relates to (x, y, or z). For other fields, this is a no-op.
    fn vector_field_or_scalar(&self) -> (Self, VectorField);

    /// Convert a raw offset into the relevant `FieldAddr`, given the type. Certain
    /// field offsets are specified to overlap in the `progdefs.qc` in order to have
    /// quick access to fields of vectors, and this can distinguish between `vector foo`
    /// and `float foo_x`.
    fn from_u16_typed(val: u16, ty: Type) -> Option<Self> {
        Self::VARIANTS
            .iter()
            .cloned()
            .find(|variant| variant.to_u16() == val && variant.type_().typeck(&ty))
    }

    /// Get the offset to this global.
    fn to_u16(&self) -> u16;

    /// Get the name of this global.
    fn name(&self) -> &'static str;

    /// Given a name, get the global address the name corresponds to.
    fn from_name(name: &str) -> Option<Self> {
        Self::VARIANTS
            .iter()
            .cloned()
            .find(|variant| variant.name() == name)
    }

    /// Get the type of values at this address
    fn type_(&self) -> Type;
}

/// Dummy address for when there are no external globals
#[derive(VariantArray, PartialEq, Hash, Clone)]
pub enum EmptyAddress {}

impl Address for EmptyAddress {
    fn type_(&self) -> Type {
        unreachable!()
    }

    fn vector_field_or_scalar(&self) -> (Self, VectorField) {
        unreachable!()
    }

    fn from_u16_typed(_: u16, _: Type) -> Option<Self> {
        None
    }

    fn to_u16(&self) -> u16 {
        unreachable!()
    }

    fn name(&self) -> &'static str {
        unreachable!()
    }
}

/// Special arguments for certain functions
///
/// > TODO: Make this generic instead of just "self" and "other" - some qcdefs may use different "special params"
#[derive(Default, Debug, Copy, Clone)]
pub struct SpecialArgs {
    /// `self` type used for e.g. think and touch functions
    pub self_: Option<ErasedEntityHandle>,
    /// `other` type used for touch functions
    pub other: Option<ErasedEntityHandle>,
}

/// The arguments passed to a QuakeC function.
#[derive(Debug)]
struct CallArgs<T> {
    /// Regular function parameters
    params: T,
}

type HashMap<K, V> = hashbrown::HashMap<K, V, std::hash::BuildHasherDefault<hash32::FnvHasher>>;

/// The type of individual values passed in and out of the QuakeC runtime.
/// Note that, in most cases, `Void` will be converted to the default value
/// for the type.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Corresponds to QuakeC's `void`. Should rarely be used, and is mostly
    /// for the return value of functions that do not return any sensible value.
    Void,
    /// An entity reference. See documentation for `EntityRef` for more details.
    Entity(EntityRef),
    /// A function pointer. This requires some context to call, see [`userdata::Function`].
    Function(Arc<dyn ErasedFunction>),
    /// A single scalar float.
    Float(f32),
    /// An `{x, y, z}` vector value.
    Vector(Vec3),
    /// A refcounted string pointer. Quake strings are not strictly ascii, but cannot have internal
    /// `NUL`, so `CStr` is used.
    String(Arc<CStr>),
}

#[cfg(test)]
mod test_value {
    /// Regression test for `Value` size.
    #[test]
    fn value_size() {
        assert_eq!(std::mem::size_of::<super::Value>(), 24);
    }
}

/// Errors that can occur when using [`Value::get`].
#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
pub enum TryScalarError {
    /// Tried to get a field of a scalar (other than `.x`)
    FieldOfScalar {
        /// The field that we attempted to get (see documentation for [`VectorField`]).
        field: VectorField,
    },
    /// Tried to get an entire vector as a scalar
    ///
    /// > TODO: Potentially remove this and just use `VectorField::X` for scalars
    CannotGetVectorAsScalar,
}

/// Errors that can occur when using [`Value::set`].
#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
pub enum SetValueError {
    /// Tried to set a vector field, but the value was not a vector.
    NoSuchField {
        /// The field that we attempted to set (see documentation for [`VectorField`]).
        field: VectorField,
    },
    /// Tried to set a vector field to a value other than float.
    TypeError {
        /// The expected type. For now, always [`Type::Float`], but stricter type-checking
        /// may be implemented in the future.
        expected: Type,
        /// The type of the value that was specfied as the source.
        found: Type,
    },
}

impl Value {
    /// The zero value of a type, which is what QuakeC's `void` (e.g. an uninitialised value)
    /// means when stored somewhere of that type.
    pub fn zero(ty: Type) -> Self {
        match ty {
            Type::AnyScalar | Type::Void => Value::Void,
            Type::Entity => Value::Entity(None),
            Type::Function => FunctionIndex::NULL.to_value(),
            Type::Vector => Value::Vector(Vec3::ZERO),
            Type::Float => Value::Float(0.),
            Type::String => Value::String(null_string()),
        }
    }

    /// Replaces [`Value::Void`] with the zero value of `ty`.
    pub fn or_zero(self, ty: Type) -> Self {
        match self {
            Value::Void => Self::zero(ty),
            other => other,
        }
    }

    /// The [`Type`] of this value.
    pub const fn type_(&self) -> Type {
        match self {
            Value::Void => Type::Void,
            Value::Entity(_) => Type::Entity,
            Value::Function(_) => Type::Function,
            Value::Float(_) => Type::Float,
            Value::Vector(_) => Type::Vector,
            Value::String(_) => Type::String,
        }
    }

    /// Get a field of the value, if it's a vector. Passing `None` will just clone the value
    /// (this is useful for generic code using `VectorField` to optionally convert a type into
    /// a scalar).
    pub(crate) fn try_scalar(&self, field: VectorField) -> Result<VmScalar, TryScalarError> {
        self.field(field)?
            .try_into()
            .map_err(|_| TryScalarError::CannotGetVectorAsScalar)
    }

    /// Get a field of a value.
    pub fn field(&self, field: VectorField) -> Result<Value, TryScalarError> {
        Ok(match (self, field.into()) {
            (Value::Vector(v), offset) => match offset {
                VectorField::XOrScalar => v.x.into(),
                VectorField::Y => v.y.into(),
                VectorField::Z => v.z.into(),
            },
            (_, VectorField::XOrScalar) => self.clone(),
            (_, offset @ (VectorField::Y | VectorField::Z)) => {
                return Err(TryScalarError::FieldOfScalar { field: offset });
            }
        })
    }

    /// Clone the source value to this value, with an optional field reference. Useful to implement
    /// [`userdata::EntityHandle::set`], as sometimes QuakeC only wants to set a single field of a
    /// vector.
    pub fn set(&mut self, field: VectorField, value: Value) -> Result<(), SetValueError> {
        match (self, field.into(), value) {
            (this, VectorField::XOrScalar, value) => *this = value,
            (Value::Vector(v), offset, Value::Float(f)) => match offset {
                VectorField::XOrScalar => v.x = f,
                VectorField::Y => v.y = f,
                VectorField::Z => v.z = f,
            },
            (_, offset @ (VectorField::Y | VectorField::Z), _) => {
                return Err(SetValueError::NoSuchField { field: offset });
            }
        }

        Ok(())
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        if value { 1. } else { 0. }.into()
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

macro_rules! impl_from_by_casting_to_float {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for Value {
                fn from(value: $t) -> Self {
                    Self::Float(value as _)
                }
            }
        )*
    };
}

impl_from_by_casting_to_float!(
    f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl TryFrom<&str> for Value {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> anyhow::Result<Self> {
        Ok(Self::String(CString::from_str(value)?.into()))
    }
}

impl TryFrom<Value> for ErasedEntityHandle {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Entity(Some(ent)) => Ok(ent),
            _ => anyhow::bail!("Expected entity, found {}", value.type_()),
        }
    }
}

impl From<Arc<dyn ErasedFunction>> for Value {
    fn from(value: Arc<dyn ErasedFunction>) -> Self {
        Self::Function(value)
    }
}

impl From<EntityRef> for Value {
    fn from(value: EntityRef) -> Self {
        Self::Entity(value)
    }
}

impl From<ErasedEntityHandle> for Value {
    fn from(value: ErasedEntityHandle) -> Self {
        Self::Entity(Some(value))
    }
}

impl From<Vec3> for Value {
    fn from(value: Vec3) -> Self {
        Self::Vector(value)
    }
}

impl From<[f32; 3]> for Value {
    fn from(value: [f32; 3]) -> Self {
        Self::Vector(value.into())
    }
}

impl From<Arc<CStr>> for Value {
    fn from(value: Arc<CStr>) -> Self {
        Self::String(value)
    }
}

impl TryFrom<[VmScalar; 3]> for Value {
    type Error = <VmScalar as TryInto<f32>>::Error;

    fn try_from(value: [VmScalar; 3]) -> Result<Self, Self::Error> {
        let [x, y, z]: [Result<f32, _>; 3] = value.map(TryInto::try_into);
        let floats = [x?, y?, z?];

        Ok(floats.into())
    }
}

#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
struct NotScalarError;

impl TryFrom<Value> for VmScalar {
    type Error = NotScalarError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(match value {
            Value::Void => VmScalar::Void,
            Value::Entity(entity_ref) => VmScalar::Entity(entity_ref),
            Value::Function(erased_function) => {
                VmScalar::Function(VmFunctionRef::Extern(erased_function).normalize())
            }
            Value::Float(float) => VmScalar::Float(float),
            // TODO: Just read x value?
            Value::Vector(_) => return Err(NotScalarError),
            Value::String(cstr) => VmScalar::String(StringRef::Temp(cstr)),
        })
    }
}

impl TryFrom<Value> for EntityRef {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Entity(entity_ref) => Ok(entity_ref),
            _ => anyhow::bail!("Expected {}, found {}", Type::Entity, value.type_()),
        }
    }
}

impl TryFrom<Value> for Arc<dyn ErasedFunction> {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Function(func) => Ok(func),
            _ => anyhow::bail!("Expected {}, found {}", Type::Function, value.type_()),
        }
    }
}

impl TryFrom<Value> for f32 {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(f) => Ok(f),
            _ => anyhow::bail!("Expected {}, found {}", Type::Float, value.type_()),
        }
    }
}

impl TryFrom<Value> for Vec3 {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Vector(vec) => Ok(vec),
            _ => anyhow::bail!("Expected {}, found {}", Type::Vector, value.type_()),
        }
    }
}

impl TryFrom<Value> for Arc<CStr> {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(str) => Ok(str),
            _ => anyhow::bail!("Expected {}, found {}", Type::String, value.type_()),
        }
    }
}

impl From<Value> for VmValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Void => VmValue::Scalar(VmScalar::Void),
            Value::Entity(entity_ref) => entity_ref.into(),
            Value::Function(erased_function) => VmValue::Scalar(VmScalar::Function(
                VmFunctionRef::Extern(erased_function).normalize(),
            )),
            Value::Float(f) => f.into(),
            Value::Vector(vec3) => vec3.into(),
            Value::String(cstr) => VmValue::Scalar(VmScalar::String(StringRef::Temp(cstr))),
        }
    }
}

/// The type of values that can be used as arguments to a QuakeC function. Implemented
/// for tuples up to 8 elements (the maximum number of arguments supported by the
/// vanilla Quake engine).
pub trait QCParams: fmt::Debug {
    /// The error returned by [`QCParams::nth`].
    type Error: Into<anyhow::Error>;

    /// Get the number of parameters available
    fn count(&self) -> usize;

    /// Get the nth argument, specified by `index`.
    fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>>;
}

/// Errors that can happen when parsing an argument list.
#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArgError<E> {
    /// An argument index was specified that was out of range for the number of supported arguments.
    ArgOutOfRange {
        /// The number of available arguments
        len: usize,
        /// The index we tried to get
        index: usize,
    },
    /// Some other kind of error occurred (usually in conversion to a [`Value`])
    Other {
        /// The underlying cause.
        error: E,
    },
}

impl<E> From<E> for ArgError<E> {
    fn from(other: E) -> Self {
        Self::Other { error: other }
    }
}

macro_rules! impl_memory_tuple {
    ($first:ident $(, $rest:ident)*) => {
        impl<$first, $($rest),*> crate::QCParams for ($first, $($rest),*)
        where
        $first: Clone + fmt::Debug + TryInto<Value>,
        $first::Error: Into<anyhow::Error>,
        $(
            $rest: Clone + fmt::Debug + TryInto<Value>,
            $rest::Error: Into<anyhow::Error>,
        )*
        {
            type Error = anyhow::Error;

            #[expect(non_snake_case)]
            fn count(&self) -> usize {
                let $first = 1usize;
                $(
                    let $rest = 1usize;
                )*

                #[allow(unused_mut)]
                let mut out = $first;

                $(
                    out += $rest;
                )*

                out
            }

            #[expect(non_snake_case)]
            fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
                let ($first, $($rest),*) = self;

                let count = self.count();
                Ok(impl_memory_tuple!(@arg_match count index $first $($rest)*))
            }
        }

        impl_memory_tuple!($($rest),*);
    };
    (@arg_match $count:ident $match_name:ident $a0:ident $($a1:ident $($a2:ident $($a3:ident $($a4:ident $($a5:ident $($a6:ident $($a7:ident)?)?)?)?)?)?)?) => {
        match $match_name {
            0 => $a0.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(1 => $a1.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(2 => $a2.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(3 => $a3.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(4 => $a4.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(5 => $a5.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(6 => $a6.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,
            $(7 => $a7.clone().try_into().map_err(|e| -> anyhow::Error { e.into() })?,)?)?)?)?)?)?)?
            _ => return Err(ArgError::ArgOutOfRange { len: $count, index: $match_name }),
        }
    };
    () => {
        impl crate::QCParams for () {
            type Error = std::convert::Infallible;

            fn count(&self) -> usize {
                0
            }

            fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
                Err(ArgError::ArgOutOfRange { len: self.count(), index })
            }
        }

    }
}

impl_memory_tuple!(A, B, C, D, E, F, G, H);

impl<T, const COUNT: usize> crate::QCParams for ArrayVec<T, COUNT>
where
    T: Clone + fmt::Debug + TryInto<Value>,
    <T as TryInto<Value>>::Error: Into<anyhow::Error>,
{
    type Error = <T as TryInto<Value>>::Error;

    fn count(&self) -> usize {
        self.len()
    }

    fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
        self.get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })
            .and_then(|val| {
                val.clone()
                    .try_into()
                    .map_err(|error| ArgError::Other { error })
            })
    }
}

impl<T> crate::QCParams for Vec<T>
where
    T: Clone + fmt::Debug + TryInto<Value>,
    <T as TryInto<Value>>::Error: Into<anyhow::Error>,
{
    type Error = <T as TryInto<Value>>::Error;

    fn count(&self) -> usize {
        self.len()
    }

    fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
        self.get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })
            .and_then(|val| {
                val.clone()
                    .try_into()
                    .map_err(|error| ArgError::Other { error })
            })
    }
}

impl<T> crate::QCParams for VecDeque<T>
where
    T: Clone + fmt::Debug + TryInto<Value>,
    <T as TryInto<Value>>::Error: Into<anyhow::Error>,
{
    type Error = <T as TryInto<Value>>::Error;

    fn count(&self) -> usize {
        self.len()
    }

    fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
        self.get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })
            .and_then(|val| {
                val.clone()
                    .try_into()
                    .map_err(|error| ArgError::Other { error })
            })
    }
}

impl<T> crate::QCParams for bump_scope::FixedBumpVec<'_, T>
where
    T: Clone + fmt::Debug + TryInto<Value>,
    <T as TryInto<Value>>::Error: Into<anyhow::Error>,
{
    type Error = <T as TryInto<Value>>::Error;

    fn count(&self) -> usize {
        self.len()
    }

    fn nth(&self, index: usize) -> Result<Value, ArgError<Self::Error>> {
        self.get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })
            .and_then(|val| {
                val.clone()
                    .try_into()
                    .map_err(|error| ArgError::Other { error })
            })
    }
}

impl<T> QCMemory for CallArgs<T>
where
    T: QCParams,
{
    type Scalar = Option<VmScalar>;

    fn num_args(&self) -> Option<usize> {
        Some(self.params.count())
    }

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
        let Ok(index) = u16::try_from(index) else {
            return Ok(None);
        };

        match function_args().find_map(|arg| arg.try_match_scalar(index)) {
            Some((arg_index, ofs)) => match self.params.nth(arg_index) {
                Ok(val) => Ok(Some(val.try_scalar(ofs).unwrap_or(VmScalar::Void))),
                Err(ArgError::ArgOutOfRange { .. }) => Ok(None),
                Err(ArgError::Other { error: e }) => Err(e.into()),
            },
            None => Ok(None),
        }
    }
}

/// The arguments to a builtin, as raw scalars copied from the caller's parameters.
#[derive(Debug)]
pub(crate) struct RawArgs(pub ArrayVec<[VmScalar; 3], MAX_ARGS>);

impl QCMemory for RawArgs {
    type Scalar = Option<VmScalar>;

    fn num_args(&self) -> Option<usize> {
        Some(self.0.len())
    }

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
        let Ok(index) = u16::try_from(index) else {
            return Ok(None);
        };

        // Parameters that weren't passed (e.g. the unused arguments of variadic builtins) are
        // `void`, rather than whatever was left in the parameter slots.
        Ok(function_args()
            .find_map(|arg| arg.try_match_scalar(index))
            .map(|(arg_index, ofs)| {
                self.0
                    .get(arg_index)
                    .map(|arg| arg[ofs as usize].clone())
                    .unwrap_or_default()
            }))
    }
}

impl QCType for QCFunctionDef {
    fn type_(&self) -> Type {
        Type::Function
    }

    fn is_null(&self) -> bool {
        self.index == 0
    }
}

/// The possible types of values exposed to the host.
///
/// > NOTE: These do not represent all possible types of values internal to the QuakeC
/// > runtime. QuakeC has "entity field" and "pointer" types - which are relative pointers
/// > within entities and absolute pointers to globals respectively - but these are implementation
/// > details due to some quirks about how QuakeC's opcodes are defined. The values exposed to the
/// > host are deliberately limited to values that are exposed when writing QuakeC code.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Any scalar value. If a function wants to be generic in its arguments, it can specify
    /// [`Type::AnyScalar`] in its arguments. The only non-scalar type is vector, and for now
    /// it is not possible to be generic over both scalars and vectors in the same argument.
    #[default]
    AnyScalar,
    /// Corresponds to QuakeC's `void`. Should rarely be used, and is mostly
    /// for the return value of functions that do not return any sensible value.
    // TODO: Should we ever expose `Void` to the caller?
    Void,
    /// An entity reference. See documentation for `EntityRef` for more details.
    Entity,
    /// A function pointer. This requires some context to call, see [`userdata::Function`].
    Function,
    /// An `{x, y, z}` vector value.
    Vector,
    /// A single scalar float.
    Float,
    /// A refcounted string pointer. Quake strings are not strictly ascii, but cannot have internal
    /// `NUL`, so `CStr` is used.
    String,
}

impl Type {
    /// Returns true if this is a scalar type, false if it is [`Type::Vector`].
    pub const fn is_scalar(&self) -> bool {
        !matches!(self, Self::Vector)
    }

    pub(crate) const fn typeck(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Type::AnyScalar
                | Type::Void
                | Type::Entity
                | Type::Function
                | Type::Float
                | Type::String,
                Type::AnyScalar,
            ) => true,
            (Type::Void, Type::Void) => true,
            (Type::Entity, Type::Entity) => true,
            (Type::Function, Type::Function) => true,
            (Type::Vector, Type::Vector) => true,
            (Type::Float, Type::Float) => true,
            (Type::String, Type::String) => true,
            _ => false,
        }
    }
}

impl TryFrom<progs::VmType> for Type {
    type Error = anyhow::Error;

    fn try_from(value: progs::VmType) -> Result<Self, Self::Error> {
        Ok(match value {
            progs::VmType::Scalar(scalar_type) => scalar_type.try_into()?,
            progs::VmType::Vector => Type::Vector,
        })
    }
}

impl TryFrom<VmScalarType> for Type {
    type Error = anyhow::Error;

    fn try_from(value: VmScalarType) -> Result<Self, Self::Error> {
        Ok(match value {
            VmScalarType::Void => Type::Void,
            VmScalarType::String => Type::String,
            VmScalarType::Float => Type::Float,
            VmScalarType::Entity => Type::Entity,
            VmScalarType::Function => Type::Function,
            VmScalarType::FieldRef | VmScalarType::GlobalRef | VmScalarType::EntityFieldRef => {
                anyhow::bail!("We don't support `{value}` in host bindings")
            }
        })
    }
}

impl Type {
    pub(crate) fn arg_size(&self) -> ArgSize {
        match self {
            Self::Vector => ArgSize::Vector,
            _ => ArgSize::Scalar,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::AnyScalar => write!(f, "<any>"),
            Type::Entity => write!(f, "entity"),
            Type::Function => write!(f, "function"),
            Type::Float => write!(f, "float"),
            Type::Vector => write!(f, "vector"),
            Type::String => write!(f, "string"),
        }
    }
}

/// A reference to a function in the `progs.dat` function table, by index. Index 0 is the null
/// function.
///
/// This is how functions defined by the `progs.dat` (both QuakeC functions and builtins) are passed
/// to the host, e.g. when QuakeC sets an entity's `think`. Calling it from the host runs the
/// function it refers to.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct FunctionIndex(pub i32);

impl FunctionIndex {
    /// The null function.
    pub const NULL: Self = Self(0);

    /// Wraps this as a [`Value`].
    pub fn to_value(self) -> Value {
        Value::Function(Arc::new(self))
    }
}

impl QCType for FunctionIndex {
    fn type_(&self) -> Type {
        Type::Function
    }

    fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl ErasedFunction for FunctionIndex {
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<Type, MAX_ARGS>> {
        anyhow::bail!(
            "The signature of function {} is only known to the VM",
            self.0
        )
    }

    fn dyn_call(&self, mut context: FnCall) -> anyhow::Result<Value> {
        let def = context.execution.functions.get_by_index(self.0)?.clone();
        let num_args = def.args.len();
        let vm_value = match def.try_into_qc() {
            Ok(quakec) => {
                if let Some(passed) = context.execution.memory.local.num_args() {
                    context.execution.check_num_args(
                        &quakec,
                        passed,
                        CallSite::Host(quakec.index),
                    )?;
                }
                context.execution.execute_def(&quakec)?
            }
            Err(builtin) => {
                let func = context.execution.context.dyn_builtin(&builtin)?;
                context
                    .execution
                    .enter_builtin(&builtin.name, &*func, num_args)?
            }
        };

        context.execution.to_value(vm_value.try_into()?)
    }
}

/// The maximum number of statements executed by a single call into the VM before it is considered
/// an infinite loop (see `PR_ExecuteProgram` in Quake).
pub const MAX_OPS_SINGLE_CALL: u32 = 100_000;

/// The maximum depth of nested QuakeC function calls (see `MAX_STACK_DEPTH` in Quake).
pub const MAX_CALL_DEPTH: usize = 32;

/// Options chosen when the VM is initialised, see [`QCVm::load_with_options`].
///
/// The defaults are strict, so that bugs in QuakeC are errors. Some of the original QuakeC
/// (e.g. Quake's `id1/progs.dat`) is buggy and only works because of quirks of Quake's VM, so
/// hosts that need to run it unmodified can opt into those quirks here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VmOptions {
    /// What happens when a QuakeC function is called with fewer arguments than it takes.
    pub missing_arguments: MissingArguments,
}

/// What happens when a QuakeC function is called with fewer arguments than it takes. See
/// [`VmOptions::missing_arguments`].
///
/// This doesn't apply to builtins, which (like Quake's `pr_argc`) see how many arguments they were
/// passed, and get `void` for the rest. Passing more arguments than a function takes is always
/// allowed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MissingArguments {
    /// Calling a QuakeC function with fewer arguments than it takes is an error.
    #[default]
    Error,
    /// Quake's behaviour: missing arguments are whatever was left in the parameter slots.
    ///
    /// Parameter slots are global memory, so this is the value of that argument in the last call
    /// (to any function) that passed it. For example, `ogre_smash12` in Quake calls `ai_charge()`
    /// without its argument. A warning is logged the first time each call site does this.
    Stale,
}

/// Where a function was called from, used to only warn once about each call with missing
/// arguments.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum CallSite {
    /// A call statement, by its index in the `progs.dat`.
    Statement(usize),
    /// A call from the host, by the index of the function called.
    Host(i32),
}

/// [`VmOptions`], and the state needed to implement them.
#[derive(Debug, Default)]
struct Compat {
    options: VmOptions,
    /// The call sites that have already been warned about missing arguments.
    warned_missing_arguments: parking_lot::Mutex<hashbrown::HashSet<CallSite>>,
}

/// Per-call (rather than per-frame) execution state.
#[derive(Debug)]
struct Limits {
    /// How many more statements may be executed before the call is considered a runaway loop.
    ops_left: Cell<u32>,
    /// The current depth of nested QuakeC calls.
    depth: Cell<usize>,
    /// Whether the QuakeC backtrace has been printed for the current error.
    printed_backtrace: Cell<bool>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            ops_left: Cell::new(MAX_OPS_SINGLE_CALL),
            depth: Cell::new(0),
            printed_backtrace: Cell::new(false),
        }
    }
}

/// The core QuakeC runtime.
#[derive(Debug)]
pub struct QCVm {
    progs: Progs,
    compat: Compat,
}

/// A reference to a function stored in a `QuakeCVm`.
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionRef {
    /// A function specified by its index in the function table. This is the value QuakeC uses for
    /// function pointers.
    Index(i32),
    /// A function specified by name.
    Name(Arc<CStr>),
}

impl From<i32> for FunctionRef {
    fn from(value: i32) -> Self {
        Self::Index(value)
    }
}

impl From<FunctionIndex> for FunctionRef {
    fn from(value: FunctionIndex) -> Self {
        Self::Index(value.0)
    }
}

impl From<Arc<CStr>> for FunctionRef {
    fn from(value: Arc<CStr>) -> Self {
        Self::Name(value)
    }
}

impl From<&CStr> for FunctionRef {
    fn from(value: &CStr) -> Self {
        value.to_owned().into()
    }
}

impl From<CString> for FunctionRef {
    fn from(value: CString) -> Self {
        Arc::<CStr>::from(value).into()
    }
}

impl TryFrom<String> for FunctionRef {
    type Error = std::ffi::NulError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Ok(CString::new(value)?.into())
    }
}

impl TryFrom<&str> for FunctionRef {
    type Error = std::ffi::NulError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.to_owned().try_into()
    }
}

impl QCVm {
    /// Load the runtime from a `progs.dat` file, with the default (strict) [`VmOptions`].
    pub fn load<R>(bytes: R) -> anyhow::Result<Self>
    where
        R: std::io::Read + std::io::Seek,
    {
        Self::load_with_options(bytes, VmOptions::default())
    }

    /// Load the runtime from a `progs.dat` file, with the given options.
    pub fn load_with_options<R>(bytes: R, options: VmOptions) -> anyhow::Result<Self>
    where
        R: std::io::Read + std::io::Seek,
    {
        Ok(Self::from_progs(Progs::load(bytes)?, options))
    }

    fn from_progs(progs: Progs, options: VmOptions) -> Self {
        Self {
            progs,
            compat: Compat {
                options,
                ..Default::default()
            },
        }
    }

    /// The options that the VM was loaded with.
    pub fn options(&self) -> &VmOptions {
        &self.compat.options
    }

    /// The global definitions in the `progs.dat`, sorted by offset.
    pub fn global_defs(&self) -> &[GlobalDef] {
        self.progs.globals.defs()
    }

    /// The entity field definitions in the `progs.dat`.
    pub fn field_defs(&self) -> &[FieldDef] {
        self.progs.entity_def.field_defs()
    }

    /// The entity field layout defined by the `progs.dat`.
    pub fn entity_def(&self) -> &EntityTypeDef {
        &self.progs.entity_def
    }

    /// The functions defined in the `progs.dat`.
    pub fn functions(&self) -> &FunctionRegistry {
        &self.progs.functions
    }

    /// The value of the global at `addr` as specified in the `progs.dat`. Hosts that store globals
    /// themselves (i.e. those defined by `progdefs.h`, such as `StartFrame`) should initialise them
    /// with this.
    pub fn initial_global(&self, addr: u16) -> anyhow::Result<Value> {
        let scalar = self.progs.globals.initial(addr.into())?.clone();
        self.scalar_to_value(scalar)
    }

    /// The initial value of a vector global (see [`QCVm::initial_global`]).
    pub fn initial_global_vector(&self, addr: u16) -> anyhow::Result<Vec3> {
        let addr = usize::from(addr);
        let [x, y, z] = [addr, addr + 1, addr + 2].map(|addr| {
            self.progs
                .globals
                .initial(addr)
                .and_then(|scalar| Ok(f32::try_from(scalar.clone())?))
        });

        Ok(Vec3::new(x?, y?, z?))
    }

    /// Resets the globals that are stored in the VM to their values from the `progs.dat`.
    pub fn reset_globals(&self) {
        self.progs.globals.reset();
    }

    /// The value of a global stored in the VM (i.e. not provided by the host).
    pub fn global(&self, addr: u16) -> anyhow::Result<Value> {
        let scalar = self.progs.globals.get_value(addr.into())?;
        self.scalar_to_value(scalar)
    }

    fn scalar_to_value(&self, scalar: VmScalar) -> anyhow::Result<Value> {
        Ok(match scalar {
            VmScalar::Void => Value::Void,
            VmScalar::Float(f) => Value::Float(f),
            VmScalar::Entity(ent) => Value::Entity(ent),
            VmScalar::String(s) => Value::String(self.progs.string_table.get(s)?),
            VmScalar::Function(VmFunctionRef::Ptr(ptr)) => FunctionIndex(ptr.0).to_value(),
            VmScalar::Function(VmFunctionRef::Extern(func)) => Value::Function(func),
            VmScalar::Field(ptr) => Value::Float(ptr.0 as f32),
            other @ (VmScalar::Global(_) | VmScalar::EntityField(..)) => {
                anyhow::bail!(
                    "Values of type {} can't be passed to the host",
                    other.type_()
                )
            }
        })
    }

    fn execution_ctx<'a, T>(
        &'a self,
        context: &'a mut dyn ErasedContext,
        limits: &'a Limits,
        params: T,
    ) -> ExecutionCtx<'a, dyn ErasedContext, Bump, CallArgs<T>> {
        ExecutionCtx {
            alloc: Bump::new(),
            memory: ExecutionMemory {
                local: CallArgs { params },
                global: &self.progs.globals,
                last_ret: None,
            },
            backtrace: Default::default(),
            context,
            limits,
            compat: &self.compat,
            entity_def: &self.progs.entity_def,
            string_table: &self.progs.string_table,
            functions: &self.progs.functions,
        }
    }

    /// Run the specified function with the specified context and arguments.
    ///
    /// Each call gets its own limits on the number of statements executed and the call depth, so
    /// hosts that call QuakeC from inside a builtin should call this again rather than nesting
    /// through [`userdata::FnCall`].
    pub fn run<'a, T, C, F>(
        &'a self,
        context: &'a mut C,
        function: F,
        params: T,
    ) -> anyhow::Result<Value>
    where
        F: Into<FunctionRef>,
        T: QCParams,
        C: ErasedContext,
    {
        let limits = Limits::default();
        let mut ctx = self.execution_ctx(context, &limits, params);

        ctx.execute(function)
    }

    /// Run a dynamic function
    pub fn run_dynamic<'a, T, C>(
        &'a self,
        context: &'a mut C,
        function: &'a dyn ErasedFunction,
        params: T,
    ) -> anyhow::Result<Value>
    where
        T: QCParams,
        C: ErasedContext,
    {
        let num_params = params.count();
        let limits = Limits::default();
        let mut ctx = self.execution_ctx(context, &limits, params);

        ctx.enter_builtin(c"{anonymous}", function, num_params)
            .and_then(|vals| vals.try_into().map_err(anyhow::Error::new))
            .and_then(|vals| ctx.to_value(vals))
    }
}

#[derive(Default, Debug)]
struct BacktraceFrame<'a>(Option<(&'a CStr, &'a BacktraceFrame<'a>)>);

impl fmt::Display for BacktraceFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((function, prev)) = self.0 {
            write!(f, "{}", function.to_string_lossy())?;

            prev.fmt(f)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
struct ExecutionMemory<'a, Caller = FunctionExecutionCtx<'a>> {
    local: Caller,
    global: &'a GlobalRegistry,
    /// Technically, every QuakeC function returns 3 scalars of arbitrary types.
    /// Only one return value is available at once, if another function is called then the previous return
    /// value is lost unless it was saved to a local. We store it here in order to allow us to access it in `get` etc.
    last_ret: Option<[VmScalar; 3]>,
}

impl ExecutionMemory<'_> {
    fn instr(&self, pc: usize) -> anyhow::Result<Statement> {
        self.local.instr(pc)
    }
}

trait QCMemory {
    type Scalar;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar>;

    /// If this is the arguments to a call from the host, the number of arguments passed.
    fn num_args(&self) -> Option<usize> {
        None
    }

    fn get_vector(&self, index: usize) -> anyhow::Result<[Self::Scalar; 3]> {
        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }
}

trait QCMemoryMut: QCMemory {
    fn set(&mut self, index: usize, value: Self::Scalar) -> anyhow::Result<()>;
}

impl<M> QCMemory for ExecutionMemory<'_, M>
where
    M: QCMemory,
    <M as QCMemory>::Scalar: Into<Option<VmScalar>>,
{
    type Scalar = VmScalar;

    fn get(&self, index: usize) -> anyhow::Result<VmScalar> {
        if RETURN_ADDRS.contains(&index) {
            // Like Quake, reading the return value before anything has been returned gives zero.
            return Ok(self
                .last_ret
                .as_ref()
                .and_then(|val| val.get(index.checked_sub(RETURN_ADDRS.start)?).cloned())
                .unwrap_or_default());
        }

        match self.local.get(index)?.into() {
            Some(val) => Ok(val),
            None => self.global.get_value(index),
        }
    }
}

impl QCMemoryMut for ExecutionMemory<'_> {
    fn set(&mut self, index: usize, value: VmScalar) -> anyhow::Result<()> {
        match self.local.try_set(index, value) {
            Ok(()) => Ok(()),
            Err(value) => self.global.set_value(index, value),
        }
    }
}

#[derive(Debug)]
struct ExecutionCtx<
    'a,
    Ctx = dyn ErasedContext,
    Alloc = BumpScope<'a>,
    Caller = FunctionExecutionCtx<'a>,
> where
    Ctx: ?Sized,
{
    alloc: Alloc,
    memory: ExecutionMemory<'a, Caller>,
    backtrace: BacktraceFrame<'a>,
    context: &'a mut Ctx,
    limits: &'a Limits,
    compat: &'a Compat,
    entity_def: &'a EntityTypeDef,

    string_table: &'a StringTable,

    /// Function definitions and bodies.
    functions: &'a FunctionRegistry,
}

enum OpResult {
    Jump(NonZeroIsize),
    Ret([VmScalar; 3]),
    Continue,
}

impl From<OpResult> for ControlFlow<[VmScalar; 3], isize> {
    fn from(value: OpResult) -> Self {
        match value {
            OpResult::Jump(ofs) => ControlFlow::Continue(ofs.get()),
            OpResult::Continue => ControlFlow::Continue(1),
            OpResult::Ret(ret) => ControlFlow::Break(ret),
        }
    }
}

impl From<()> for OpResult {
    fn from(_: ()) -> Self {
        Self::Continue
    }
}

impl From<[VmScalar; 3]> for OpResult {
    fn from(value: [VmScalar; 3]) -> Self {
        Self::Ret(value)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ArgAddr {
    pub addr: u16,
    pub index: usize,
}

impl ArgAddr {
    pub(crate) fn vector_field_addrs(&self) -> impl Iterator<Item = u16> {
        VectorField::FIELDS
            .map(|field| self.addr + field as u16)
            .into_iter()
    }
}

impl ArgAddr {
    pub(crate) fn try_match_scalar(&self, idx: u16) -> Option<(usize, VectorField)> {
        if (self.addr..self.addr + VectorField::FIELDS.len() as u16).contains(&idx) {
            Some((self.index, VectorField::from_u16(idx - self.addr)?))
        } else {
            None
        }
    }
}

pub(crate) fn function_args() -> impl ExactSizeIterator<Item = ArgAddr> {
    // TODO: Make this generic so contexts can use different arg addrs

    [
        ArgAddr { addr: 4, index: 0 },
        ArgAddr { addr: 7, index: 1 },
        ArgAddr { addr: 10, index: 2 },
        ArgAddr { addr: 13, index: 3 },
        ArgAddr { addr: 16, index: 4 },
        ArgAddr { addr: 19, index: 5 },
        ArgAddr { addr: 22, index: 6 },
        ArgAddr { addr: 25, index: 7 },
    ]
    .into_iter()
}

const RETURN_ADDRS: Range<usize> = 1..4;

/// Frustratingly, the `bump-scope` crate doesn't have a way to be generic over `Bump` or `BumpScope`.
trait ScopedAlloc {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O;
}

impl ScopedAlloc for Bump {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        self.scoped(func)
    }
}

impl ScopedAlloc for BumpScope<'_> {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        self.scoped(func)
    }
}

impl ScopedAlloc for &'_ mut BumpScope<'_> {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        (**self).scoped(func)
    }
}

impl<Ctx, Alloc> ExecutionCtx<'_, Ctx, Alloc>
where
    Ctx: ?Sized,
    Alloc: ScopedAlloc,
{
    pub fn instr(&self, pc: usize) -> anyhow::Result<Statement> {
        self.memory.instr(pc)
    }
}

impl<'a, Ctx, Alloc, Caller> ExecutionCtx<'a, Ctx, Alloc, Caller>
where
    Ctx: ?Sized + AsErasedContext,
    Alloc: ScopedAlloc,
{
    fn with_args<F, A, O>(&mut self, name: &CStr, params: A, func: F) -> O
    where
        F: FnOnce(ExecutionCtx<'_, dyn ErasedContext, BumpScope<'_>, CallArgs<A>>) -> O,
        A: QCParams,
    {
        self.with_memory(name, CallArgs { params }, func)
    }

    /// Runs `func` in a new frame (named `name`) whose memory is `local`.
    fn with_memory<F, M, O>(&mut self, name: &CStr, local: M, func: F) -> O
    where
        F: FnOnce(ExecutionCtx<'_, dyn ErasedContext, BumpScope<'_>, M>) -> O,
    {
        let ExecutionCtx {
            alloc,
            memory: ExecutionMemory { global, .. },
            backtrace,
            context,
            limits,
            compat,
            entity_def,
            string_table,
            functions,
        } = self;

        alloc.scoped(|alloc| {
            func(ExecutionCtx {
                alloc,
                memory: ExecutionMemory {
                    local,
                    global,
                    last_ret: None,
                },
                backtrace: BacktraceFrame(Some((name, backtrace))),
                context: context.as_erased_mut(),
                limits,
                compat,
                entity_def,
                string_table,
                functions,
            })
        })
    }
}

impl<'a, Alloc, Caller> ExecutionCtx<'a, dyn ErasedContext, Alloc, Caller> {
    fn downcast<T>(self) -> Option<ExecutionCtx<'a, T, Alloc, Caller>>
    where
        T: Any,
    {
        let ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context,
            limits,
            compat,
            entity_def,
            string_table,
            functions,
        } = self;

        Some(ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context: (context as &mut dyn Any).downcast_mut()?,
            limits,
            compat,
            entity_def,
            string_table,
            functions,
        })
    }
}

/// Annoyingly, we can't generically handle `&mut T` where `T: ErasedContext` and _also_
/// `&mut dyn ErasedContext` without introducing this ugly trait.
///
/// > TODO: Split up the execution context so that we handle memory and execution separately,
/// > since the memory is the only thing really stopping us from making this more ergonomic.
pub trait AsErasedContext {
    /// Convert to a mutable type-erased context.
    fn as_erased_mut(&mut self) -> &mut dyn ErasedContext;
    /// Convert to an immutable type-erased context.
    fn as_erased(&self) -> &dyn ErasedContext;
}

impl<T> AsErasedContext for T
where
    T: ErasedContext,
{
    fn as_erased_mut(&mut self) -> &mut dyn ErasedContext {
        self
    }

    fn as_erased(&self) -> &dyn ErasedContext {
        self
    }
}

impl AsErasedContext for dyn ErasedContext {
    fn as_erased_mut(&mut self) -> &mut dyn ErasedContext {
        self
    }

    fn as_erased(&self) -> &dyn ErasedContext {
        self
    }
}

impl<Ctx, Alloc, Caller> ExecutionCtx<'_, Ctx, Alloc, Caller>
where
    Ctx: ?Sized,
{
    /// Checks the number of arguments that a QuakeC function was called with, see
    /// [`VmOptions::missing_arguments`].
    fn check_num_args(
        &self,
        function: &QCFunctionDef,
        num_args: usize,
        site: CallSite,
    ) -> anyhow::Result<()> {
        let takes = function.args.len();
        let name = function.name.to_string_lossy();

        match num_args.cmp(&takes) {
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                tracing::debug!("{name} called with {num_args} args, but it takes {takes}");
            }
            std::cmp::Ordering::Less => match self.compat.options.missing_arguments {
                MissingArguments::Error => {
                    anyhow::bail!("{name} called with {num_args} args, but it takes {takes}")
                }
                MissingArguments::Stale => {
                    if self.compat.warned_missing_arguments.lock().insert(site) {
                        tracing::warn!(
                            "{name} called with {num_args} args, but it takes {takes}: using the \
                             values left in the parameter slots ({site:?})"
                        );
                    }
                }
            },
        }

        Ok(())
    }
}

impl<'a, Ctx, Alloc, Caller> ExecutionCtx<'a, Ctx, Alloc, Caller>
where
    Ctx: ?Sized + AsErasedContext,
{
    fn to_value(&self, vm_value: VmValue) -> anyhow::Result<Value> {
        match vm_value {
            VmValue::Vector(vec) => Ok(Value::Vector(vec.into())),
            VmValue::Scalar(VmScalar::Float(f)) => Ok(Value::Float(f)),
            VmValue::Scalar(VmScalar::Void) => Ok(Value::Void),
            VmValue::Scalar(VmScalar::Entity(ent)) => Ok(Value::Entity(ent)),
            VmValue::Scalar(VmScalar::String(str)) => {
                Ok(Value::String(self.string_table.get(str)?))
            }
            VmValue::Scalar(VmScalar::Function(VmFunctionRef::Ptr(ptr))) => {
                Ok(FunctionIndex(ptr.0).to_value())
            }
            VmValue::Scalar(VmScalar::Function(VmFunctionRef::Extern(func))) => {
                Ok(Value::Function(func))
            }
            // Field references are passed to the host as their offset, as Quake treats them as
            // integers. This is how e.g. `find` gets the field to search.
            VmValue::Scalar(VmScalar::Field(ptr)) => Ok(Value::Float(ptr.0 as f32)),
            VmValue::Scalar(VmScalar::EntityField(..) | VmScalar::Global(_)) => anyhow::bail!(
                "Values of type {} are unsupported as arguments to builtins",
                vm_value.type_()
            ),
        }
    }
}

impl<Alloc, Caller> ExecutionCtx<'_, dyn ErasedContext, Alloc, Caller>
where
    Alloc: ScopedAlloc,
    Caller: fmt::Debug + QCMemory<Scalar = Option<VmScalar>>,
{
    pub fn execute<F>(&mut self, function: F) -> anyhow::Result<Value>
    where
        F: Into<FunctionRef>,
    {
        let def = self.functions.get(function)?.clone();
        let num_args = def.args.len();

        let out = match def.try_into_qc() {
            Ok(quakec) => {
                if let Some(passed) = self.memory.local.num_args() {
                    self.check_num_args(&quakec, passed, CallSite::Host(quakec.index))?;
                }
                self.execute_def(&quakec)?
            }
            // Running a builtin directly is allowed, it just runs the host's function.
            Err(builtin) => {
                let func = self.context.dyn_builtin(&builtin)?;
                self.enter_builtin(&builtin.name, &*func, num_args)?
            }
        };
        let value = VmValue::try_from(out)?;

        self.to_value(value)
    }

    /// Runs a QuakeC function. Its arguments are copied from the current frame's parameters, as the
    /// `OP_CALL*` opcodes do.
    // TODO: We can use the unsafe checkpoint API if just recursing becomes too slow.
    pub fn execute_def(&mut self, function: &QCFunctionDef) -> anyhow::Result<[VmScalar; 3]> {
        let Self {
            alloc,
            memory,
            backtrace,
            context,
            limits,
            compat,
            entity_def,
            string_table,
            functions,
        } = self;

        let limits: &Limits = limits;
        let depth = limits.depth.get();
        if depth + 1 >= MAX_CALL_DEPTH {
            anyhow::bail!(
                "Call stack overflow calling {}: more than {MAX_CALL_DEPTH} nested calls",
                function.name.to_string_lossy()
            );
        }

        let initial_locals = memory.global.initial_range(function.body.locals.clone())?;

        tracing::debug!(target: "qcvm::calls", "Calling QuakeC function {}", function.name.to_string_lossy());

        limits.depth.set(depth + 1);

        let result = alloc.scoped(move |alloc| {
            let mut new_memory = ExecutionMemory {
                local: function.ctx(&alloc, initial_locals),
                global: memory.global,
                last_ret: None,
            };

            // Arguments are copied to consecutive addresses starting at the first local. Like
            // Quake, this doesn't check that they fit in the locals: `qcc` sometimes says that
            // functions have no locals even though they have arguments, in which case they are
            // copied to (and read from) global memory.
            let mut dst = function.body.locals.start;

            for (src, dst_size) in function_args().zip(&function.args) {
                match dst_size.arg_size() {
                    ArgSize::Scalar => {
                        new_memory.set(dst, memory.get(src.addr as usize)?)?;
                        dst += 1;
                    }

                    ArgSize::Vector => {
                        for src in src.vector_field_addrs() {
                            new_memory.set(dst, memory.get(src as usize)?)?;
                            dst += 1;
                        }
                    }
                }
            }

            let mut out = ExecutionCtx {
                memory: new_memory,
                alloc,
                backtrace: BacktraceFrame(Some((&function.name, &*backtrace))),
                context: &mut **context,
                limits,
                compat,
                entity_def,
                string_table,
                functions,
            };

            out.execute_internal()
        });

        limits.depth.set(depth);

        result.with_context(|| format!("in {}", function.name.to_string_lossy()))
    }

    /// The names of the functions currently being executed, innermost first.
    pub fn backtrace(&self) -> impl Iterator<Item = &'_ CStr> + '_ {
        std::iter::successors(self.backtrace.0.as_ref(), |(_, prev)| prev.0.as_ref())
            .map(|(name, _)| *name)
    }

    /// Logs the QuakeC call stack, if `force` is set or backtraces are enabled with
    /// `RUST_LIB_BACKTRACE`/`RUST_BACKTRACE`.
    pub fn print_backtrace(&self, force: bool) {
        let backtrace_var =
            std::env::var("RUST_LIB_BACKTRACE").or_else(|_| std::env::var("RUST_BACKTRACE"));
        let backtrace_enabled = matches!(backtrace_var.as_deref(), Ok(v) if v != "0");
        if force || backtrace_enabled {
            let frames = self
                .backtrace()
                .enumerate()
                .map(|(depth, name)| format!("{depth}: {}", name.to_string_lossy()))
                .collect::<Vec<_>>()
                .join("\n");
            tracing::warn!("QuakeC backtrace:\n{frames}");
        }
    }
}

impl<Alloc, Caller> ExecutionCtx<'_, dyn ErasedContext, Alloc, Caller>
where
    Alloc: ScopedAlloc,
    Caller: fmt::Debug + QCMemory,
    <Caller as QCMemory>::Scalar: Into<Option<VmScalar>>,
{
    pub fn get_scalar<I, O>(&self, index: I) -> anyhow::Result<O>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        VmScalar: TryInto<O>,
        <VmScalar as TryInto<O>>::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        match self.context.dyn_global(index.try_into()?, Type::AnyScalar) {
            Ok(val) => Ok(val.try_scalar(VectorField::XOrScalar)?.try_into()?),
            Err(AddrError::Other { error: other }) => Err(other),
            Err(AddrError::OutOfRange) => Ok(self.memory.get(index)?.try_into()?),
        }
    }

    pub fn get_vec3<I>(&self, index: I) -> anyhow::Result<Vec3>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        match self.context.dyn_global(index.try_into()?, Type::Vector) {
            Ok(val) => Ok(val.try_into()?),
            Err(AddrError::Other { error: other }) => Err(other),
            Err(AddrError::OutOfRange) => {
                let [x, y, z] = self.memory.get_vector(index)?.map(f32::try_from);
                Ok(Vec3::new(x?, y?, z?))
            }
        }
    }
}

impl<Alloc> ExecutionCtx<'_, dyn ErasedContext, Alloc>
where
    Alloc: ScopedAlloc,
{
    /// Stores a scalar at `index`, which may be a global provided by the host.
    pub fn set<I, V>(&mut self, index: I, value: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: Into<VmScalar>,
    {
        let index = index.try_into()?;
        let value = value.into();

        // Field references and pointers only exist inside the VM, so they can't be host globals.
        if !matches!(
            value,
            VmScalar::Field(_) | VmScalar::Global(_) | VmScalar::EntityField(..)
        ) {
            let host_value = self.to_value(value.clone().into())?;
            match self.context.dyn_set_global(index.try_into()?, host_value) {
                Ok(()) => return Ok(()),
                Err(AddrError::Other { error: other }) => return Err(other),
                Err(AddrError::OutOfRange) => {}
            }
        }

        self.memory.set(index, value)
    }

    /// Sets the "last return" global. QuakeC only allows 1 function return to be accessible
    /// at a given time.
    ///
    /// This can't be done with the regular `set` as this shouldn't be accessible by regular
    /// QuakeC code, only from the engine.
    pub fn set_return(&mut self, values: [VmScalar; 3]) {
        self.memory.last_ret = Some(values);
    }

    /// Stores a vector at `index`. If the host provides a vector global there, it is set as a
    /// whole, otherwise each component is stored separately.
    pub fn set_vec3<I, V>(&mut self, index: I, values: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: Into<Vec3>,
    {
        let index = index.try_into()?;
        let values = values.into();

        match self
            .context
            .dyn_set_global(index.try_into()?, Value::Vector(values))
        {
            Ok(()) => Ok(()),
            Err(AddrError::Other { error: other }) => Err(other),
            Err(AddrError::OutOfRange) => {
                for (i, value) in values.to_array().into_iter().enumerate() {
                    self.set(index + i, value)?;
                }

                Ok(())
            }
        }
    }

    fn execute_internal(&mut self) -> anyhow::Result<[VmScalar; 3]> {
        let mut counter: usize = 0;

        loop {
            let ops_left = self.limits.ops_left.get() - 1;
            self.limits.ops_left.set(ops_left);
            if ops_left == 0 {
                self.print_backtrace(true);
                self.limits.printed_backtrace.set(true);
                anyhow::bail!(
                    "runaway loop error: more than {MAX_OPS_SINGLE_CALL} statements executed"
                );
            }

            let statement = self.instr(counter)?;
            let result = match self.execute_statement(statement, counter) {
                Ok(result) => result,
                Err(e) => {
                    if !self.limits.printed_backtrace.replace(true) {
                        self.print_backtrace(false);
                    }

                    return Err(e.context(format!(
                        "at statement {counter} ({} {} {} {})",
                        statement.opcode, statement.arg1, statement.arg2, statement.arg3
                    )));
                }
            };

            match result.into() {
                ControlFlow::Continue(idx) => {
                    counter = counter
                        .checked_add_signed(idx)
                        .ok_or_else(|| anyhow::format_err!("Out-of-bounds instruction access"))?;
                }
                ControlFlow::Break(vals) => return Ok(vals),
            }
        }
    }
}
