#![deny(missing_docs)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

use std::{
    any::Any,
    ffi::{CStr, CString},
    fmt,
    num::NonZeroIsize,
    ops::{ControlFlow, Range},
    str::FromStr,
    sync::Arc,
};

use arrayvec::ArrayVec;
use bump_scope::{Bump, BumpScope};
use glam::Vec3;
use itertools::Itertools;
use snafu::Snafu;

use crate::{
    entity::EntityTypeDef,
    load::Progs,
    progs::{
        StringRef, StringTable, VmFunctionRef, VmScalar, VmScalarType, VmValue,
        functions::{ArgSize, FunctionExecutionCtx, QuakeCFunctionDef, Statement},
        globals::GlobalRegistry,
    },
    userdata::{ErasedContext, ErasedEntityHandle, ErasedFunction, FnCall, QuakeCType},
};

mod entity;
mod load;
mod ops;
mod progs;
pub mod userdata;

pub use arrayvec;

pub use anyhow::{self, Error};

pub use crate::progs::{
    EntityRef, FieldDef, VectorField,
    functions::{Builtin, BuiltinDef, FunctionDef, FunctionRegistry, MAX_ARGS},
};

#[cfg(feature = "quake1")]
pub mod quake1;

/// The arguments passed to a QuakeC function.
#[derive(Debug)]
struct CallArgs<T>(T);

type HashMap<K, V> = hashbrown::HashMap<K, V, std::hash::BuildHasherDefault<hash32::FnvHasher>>;

// We don't implement this via `QuakeCArgs` as we want to support the scalars being
// flattened out rather than using `Value::Vec`.
impl QuakeCMemory for CallArgs<ArrayVec<VmScalar, MAX_ARGS>> {
    type Scalar = Option<VmScalar>;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
        if !ARG_ADDRS.contains(&index) {
            return Ok(None);
        }

        let arg_offset = index
            .checked_sub(ARG_ADDRS.start)
            .expect("Programmer error - index out of range for args");

        Ok(Some(
            self.0.get(arg_offset).unwrap_or(&VmScalar::Void).clone(),
        ))
    }
}

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

/// Errors that can occur when using [`Value::get`].
#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
pub struct GetValueError {
    /// The field that we attempted to get (see documentation for [`VectorField`]).
    pub field: VectorField,
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
    /// The [`Type`] of this value.
    pub fn type_(&self) -> Type {
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
    pub fn get(&self, field: impl Into<Option<VectorField>>) -> Result<Value, GetValueError> {
        Ok(match (self, field.into()) {
            (Value::Vector(v), Some(offset)) => match offset {
                VectorField::X => v.x.into(),
                VectorField::Y => v.y.into(),
                VectorField::Z => v.z.into(),
            },
            (other, None) => other.clone(),
            (_, Some(offset)) => return Err(GetValueError { field: offset }),
        })
    }

    /// Clone the source value to this value, with an optional field reference. Useful to implement
    /// [`userdata::EntityHandle::set`], as sometimes QuakeC only wants to set a single field of a
    /// vector.
    pub fn set(
        &mut self,
        field: impl Into<Option<VectorField>>,
        value: Value,
    ) -> Result<(), SetValueError> {
        match (self, field.into(), value) {
            (this, None, value) => *this = value,
            (Value::Vector(v), Some(offset), Value::Float(f)) => match offset {
                VectorField::X => v.x = f,
                VectorField::Y => v.y = f,
                VectorField::Z => v.z = f,
            },
            (Value::Vector(_), Some(_), val) => {
                return Err(SetValueError::TypeError {
                    expected: Type::Float,
                    found: val.type_(),
                });
            }
            (_, Some(offset), _) => return Err(SetValueError::NoSuchField { field: offset }),
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
        Self::Entity(EntityRef::Entity(value))
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

impl TryFrom<Value> for VmScalar {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(match value {
            Value::Void => VmScalar::Void,
            Value::Entity(entity_ref) => VmScalar::Entity(entity_ref),
            Value::Function(erased_function) => {
                VmScalar::Function(VmFunctionRef::Extern(erased_function))
            }
            Value::Float(float) => VmScalar::Float(float),
            // TODO: Just read x value?
            Value::Vector(_) => anyhow::bail!("Tried to read vector as a scalar"),
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
            Value::Function(erased_function) => {
                VmValue::Scalar(VmScalar::Function(VmFunctionRef::Extern(erased_function)))
            }
            Value::Float(f) => f.into(),
            Value::Vector(vec3) => vec3.into(),
            Value::String(cstr) => VmValue::Scalar(VmScalar::String(StringRef::Temp(cstr))),
        }
    }
}

/// The type of values that can be used as arguments to a QuakeC function. Implemented
/// for tuples up to 8 elements (the maximum number of arguments supported by the
/// vanilla Quake engine).
pub trait QuakeCArgs: fmt::Debug {
    /// The error returned by [`QuakeCArgs::nth`].
    type Error: std::error::Error;

    /// Get the nth argument, specified by `index`.
    fn nth(&self, index: usize) -> Result<Value, Self::Error>;
}

/// Errors that can happen when parsing an argument list.
#[derive(Debug)]
pub enum ArgError {
    /// An argument index was specified that was out of range for the number of supported arguments.
    ArgOutOfRange(usize),
    /// Some other kind of error occurred (usually in conversion to a [`Value`])
    Other(Box<dyn std::error::Error>),
}

impl From<Box<dyn std::error::Error>> for ArgError {
    fn from(other: Box<dyn std::error::Error>) -> Self {
        Self::Other(other)
    }
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArgOutOfRange(i) => write!(f, "Argument out of range: {i}"),
            Self::Other(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ArgError {}

macro_rules! impl_memory_tuple {
    ($first:ident $(, $rest:ident)*) => {
        impl<$first, $($rest),*> crate::QuakeCArgs for ($first, $($rest),*)
        where
        $first: Clone + fmt::Debug + TryInto<Value>,
        $first::Error: std::error::Error + 'static,
        $(
            $rest: Clone + fmt::Debug + TryInto<Value>,
            $rest::Error: std::error::Error + 'static,
        )*
        {
            type Error = ArgError;

            #[expect(non_snake_case)]
            fn nth(&self, index: usize) -> Result<Value, Self::Error> {
                let ($first, $($rest),*) = self;

                Ok(impl_memory_tuple!(@arg_match index $first $($rest)*))
            }
        }

        impl_memory_tuple!($($rest),*);
    };
    (@arg_match $match_name:ident $a0:ident $($a1:ident $($a2:ident $($a3:ident $($a4:ident $($a5:ident $($a6:ident $($a7:ident)?)?)?)?)?)?)?) => {
        match $match_name {
            0 => $a0.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(1 => $a1.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(2 => $a2.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(3 => $a3.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(4 => $a4.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(5 => $a5.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(6 => $a6.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
            $(7 => $a7.clone().try_into().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,)?)?)?)?)?)?)?
            _ => return Err(ArgError::ArgOutOfRange($match_name)),
        }
    };
    () => {
        impl crate::QuakeCArgs for () {
            type Error = ArgError;

            fn nth(&self, index: usize) -> Result<Value, Self::Error> {
                Err(ArgError::ArgOutOfRange(index))
            }
        }

    }
}

impl_memory_tuple!(A, B, C, D, E, F, G, H);

impl<T> QuakeCMemory for CallArgs<T>
where
    T: QuakeCArgs,
{
    type Scalar = Option<VmScalar>;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
        if !ARG_ADDRS.contains(&index) {
            return Ok(None);
        }

        let arg_offset = index
            .checked_sub(ARG_ADDRS.start)
            .expect("Programmer error - index out of range for args");
        let field_offset = arg_offset % 3;
        let index = arg_offset / 3;

        Ok(
            match self.0.nth(index).map_err(|e| anyhow::format_err!("{e}"))? {
                Value::Vector(vec) => Some(<[f32; 3]>::from(vec)[field_offset].into()),
                other => Some(other.try_into()?),
            },
        )
    }
}

impl QuakeCType for QuakeCFunctionDef {
    fn type_(&self) -> Type {
        Type::Function
    }

    fn is_null(&self) -> bool {
        self.offset == 0
    }
}

/// The possible types of values exposed to the host.
///
/// > NOTE: These do not represent all possible types of values internal to the QuakeC
/// > runtime. QuakeC has "entity field" and "pointer" types - which are relative pointers
/// > within entities and absolute pointers to globals respectively - but these are implementation
/// > details due to some quirks about how QuakeC's opcodes are defined. The values exposed to the
/// > host are deliberately limited to values that are exposed when writing QuakeC code.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Any scalar value. If a function wants to be generic in its arguments, it can specify
    /// [`Type::AnyScalar`] in its arguments. The only non-scalar type is vector, and for now
    /// it is not possible to be generic over both scalars and vectors in the same argument.
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

impl ErasedFunction for QuakeCFunctionDef {
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<Type, MAX_ARGS>> {
        Ok(self
            .args
            .iter()
            .map(|size| match size.arg_size() {
                ArgSize::Scalar => Type::AnyScalar,
                ArgSize::Vector => Type::Vector,
            })
            .collect())
    }

    fn dyn_call(&self, mut context: FnCall) -> anyhow::Result<Value> {
        let vm_value = context.execution.execute_def(self)?;

        context.execution.to_value(vm_value.try_into()?)
    }
}

/// The core QuakeC runtime.
pub struct QuakeCVm {
    progs: Progs,
}

/// A reference to a function stored in a `QuakeCVm`.
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionRef {
    /// A function specified by index. This can be useful for interacting with code
    /// that is ABI-compatible with your environment but not API-compatible.
    Offset(i32),
    /// A function specified by name.
    Name(Arc<CStr>),
}

impl From<i32> for FunctionRef {
    fn from(value: i32) -> Self {
        Self::Offset(value)
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

impl QuakeCVm {
    /// Load the runtime from a `progs.dat` file.
    pub fn load<R>(bytes: R) -> anyhow::Result<Self>
    where
        R: std::io::Read + std::io::Seek,
    {
        Ok(Self {
            progs: Progs::load(bytes)?,
        })
    }

    /// Run the specified function with the specified context and arguments.
    pub fn run<'a, T, C, F>(
        &'a self,
        context: &'a mut C,
        function: F,
        args: T,
    ) -> anyhow::Result<Value>
    where
        F: Into<FunctionRef>,
        T: QuakeCArgs,
        C: ErasedContext,
    {
        let mut ctx: ExecutionCtx<'_, dyn ErasedContext, _, CallArgs<T>> = ExecutionCtx {
            alloc: Bump::new(),
            memory: ExecutionMemory {
                local: CallArgs(args),
                global: &self.progs.globals,
                last_ret: None,
            },
            backtrace: Default::default(),
            context,
            entity_def: &self.progs.entity_def,
            string_table: &self.progs.string_table,
            functions: &self.progs.functions,
        };

        ctx.execute(function)
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

trait QuakeCMemory {
    type Scalar;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar>;

    fn get_vector(&self, index: usize) -> anyhow::Result<[Self::Scalar; 3]> {
        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }
}

impl<M> QuakeCMemory for ExecutionMemory<'_, M>
where
    M: QuakeCMemory<Scalar = Option<VmScalar>>,
{
    type Scalar = VmScalar;

    fn get(&self, index: usize) -> anyhow::Result<VmScalar> {
        if RETURN_ADDRS.contains(&index) {
            return self
                .last_ret
                .as_ref()
                .and_then(|val| val.get(index.checked_sub(RETURN_ADDRS.start)?).cloned())
                .ok_or_else(|| {
                    anyhow::format_err!("Tried to read return values before calling a function")
                });
        }

        match self.local.get(index)? {
            Some(val) => Ok(val),
            None => self.global.get_value(index),
        }
    }
}

impl ExecutionMemory<'_> {
    fn set(&mut self, index: usize, value: VmScalar) -> anyhow::Result<()> {
        self.local.set(index, value)
    }

    fn set_vector(&mut self, index: usize, values: [VmScalar; 3]) -> anyhow::Result<()> {
        self.local.set_vector(index, values)
    }
}

/// The name of the "magic function" implementing `OP_STATE`.
pub const MAGIC_OP_STATE_IMPL_FUNC: &CStr = c"__state__";
/// The number of arguments of the "magic function" implementing `OP_STATE`.
pub const MAGIC_OP_STATE_IMPL_NUM_ARGS: usize = 2;

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
    entity_def: &'a EntityTypeDef,

    string_table: &'a StringTable,

    /// Function definitions and bodies.
    functions: &'a FunctionRegistry,
}

impl<Ctx, Alloc, Caller> ExecutionCtx<'_, Ctx, Alloc, Caller> {}

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

const ARG_ADDRS: Range<usize> = 4..28;
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
    fn with_args<F, A, O>(&mut self, name: &CStr, args: A, func: F) -> O
    where
        F: FnOnce(ExecutionCtx<'_, dyn ErasedContext, BumpScope<'_>, CallArgs<A>>) -> O,
        CallArgs<A>: QuakeCMemory,
    {
        let ExecutionCtx {
            alloc,
            memory: ExecutionMemory { global, .. },
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        alloc.scoped(|alloc| {
            func(ExecutionCtx {
                alloc,
                memory: ExecutionMemory {
                    local: CallArgs(args),
                    global,
                    last_ret: None,
                },
                backtrace: BacktraceFrame(Some((name, backtrace))),
                context: context.as_erased_mut(),
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
            entity_def,
            string_table,
            functions,
        } = self;

        Some(ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context: (context as &mut dyn Any).downcast_mut()?,
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
                let arg_func = self.functions.get_by_index(ptr.0)?.clone();

                match arg_func.try_into_qc() {
                    Ok(quakec) => Ok(Value::Function(Arc::new(quakec))),
                    Err(builtin) => Ok(Value::Function(
                        self.context.as_erased().dyn_builtin(&builtin)?,
                    )),
                }
            }
            VmValue::Scalar(VmScalar::Function(VmFunctionRef::Extern(func))) => {
                Ok(Value::Function(func))
            }
            VmValue::Scalar(
                VmScalar::EntityField(..) | VmScalar::Field(_) | VmScalar::Global(_),
            ) => anyhow::bail!(
                "Values of type {} are unsupported as arguments to builtins",
                vm_value.type_()
            ),
        }
    }
}

impl<Alloc, Caller> ExecutionCtx<'_, dyn ErasedContext, Alloc, Caller>
where
    Alloc: ScopedAlloc,
    Caller: fmt::Debug + QuakeCMemory<Scalar = Option<VmScalar>>,
{
    pub fn execute<F>(&mut self, function: F) -> anyhow::Result<Value>
    where
        F: Into<FunctionRef>,
    {
        match function.into() {
            FunctionRef::Offset(idx) => self.execute_by_index(idx),
            FunctionRef::Name(name) => self.execute_by_name(name),
        }
    }

    pub fn execute_by_index(&mut self, function: i32) -> anyhow::Result<Value> {
        let quakec_func = self
            .functions
            .get_by_index(function)?
            .clone()
            .try_into_qc()
            .map_err(|_| anyhow::format_err!("Not a quakec function (TODO)"))?;

        let out = self.execute_def(&quakec_func)?;
        let value = VmValue::try_from(out)?;

        self.to_value(value)
    }

    pub fn execute_by_name<F>(&mut self, function: F) -> anyhow::Result<Value>
    where
        F: AsRef<CStr>,
    {
        let quakec_func = self
            .functions
            .get_by_name(function)?
            .clone()
            .try_into_qc()
            .map_err(|_| anyhow::format_err!("Not a quakec function (TODO)"))?;

        let out = self.execute_def(&quakec_func)?;
        let value = VmValue::try_from(out)?;

        self.to_value(value)
    }

    // TODO: We can use the unsafe checkpoint API if just recursing becomes too slow.
    pub fn execute_def(&mut self, function: &QuakeCFunctionDef) -> anyhow::Result<[VmScalar; 3]> {
        let Self {
            alloc,
            memory,
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        alloc.scoped(move |alloc| {
            let mut new_memory = ExecutionMemory {
                local: function.ctx(&alloc),
                global: memory.global,
                last_ret: None,
            };

            let mut dst_locals = function.body.locals.clone();

            for (mut src, dst_size) in ARG_ADDRS.chunks(3).into_iter().zip(&function.args) {
                match dst_size.arg_size() {
                    ArgSize::Scalar => {
                        let src = src.next().unwrap();
                        let dst = dst_locals
                            .next()
                            .ok_or_else(|| anyhow::format_err!("Too few locals for arguments"))?;
                        new_memory.set(dst, memory.get(dbg!(src))?)?;
                    }

                    ArgSize::Vector => {
                        for (src, dst) in src.zip(dst_locals.by_ref()) {
                            new_memory.set(dst, memory.get(src)?)?;
                        }
                    }
                }
            }

            let mut out = ExecutionCtx {
                memory: new_memory,
                alloc,
                backtrace: BacktraceFrame(Some((&function.name, &*backtrace))),
                context: &mut **context,
                entity_def,
                string_table,
                functions,
            };

            out.execute_internal()
        })
    }

    // TODO
    pub fn backtrace(&self) -> impl Iterator<Item = &'_ CStr> + '_ {
        std::iter::successors(self.backtrace.0.as_ref(), |(_, prev)| prev.0.as_ref())
            .map(|(name, _)| *name)
    }

    // TODO
    #[expect(dead_code)]
    pub fn print_backtrace(&self, force: bool) {
        let backtrace_var =
            std::env::var("RUST_LIB_BACKTRACE").or_else(|_| std::env::var("RUST_BACKTRACE"));
        let backtrace_enabled = matches!(backtrace_var.as_deref(), Err(_) | Ok("0"));
        if force || backtrace_enabled {
            for (depth, name) in self.backtrace().enumerate() {
                // TODO: More info about the function (e.g. builtin vs internal)
                println!("{}: {}", depth, name.to_string_lossy());
            }
        }
    }
}

impl ExecutionCtx<'_> {
    pub fn get<I, O>(&self, index: I) -> anyhow::Result<O>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        VmScalar: TryInto<O>,
        <VmScalar as TryInto<O>>::Error: std::error::Error + Send + Sync + 'static,
    {
        Ok(self.memory.get(index.try_into()?)?.try_into()?)
    }

    pub fn set<I, V>(&mut self, index: I, value: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<VmScalar>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        self.memory.set(index.try_into()?, value.try_into()?)
    }

    /// Sets the "last return" global. QuakeC only allows 1 function return to be accessible
    /// at a given time.
    ///
    /// This can't be done with the regular `set` as this shouldn't be accessible by regular
    /// QuakeC code, only from the engine.
    pub fn set_return(&mut self, values: [VmScalar; 3]) {
        self.memory.last_ret = Some(values);
    }

    pub fn get_vector<I, O>(&self, index: I) -> anyhow::Result<[O; 3]>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        VmScalar: TryInto<O>,
        <VmScalar as TryInto<O>>::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }

    pub fn get_vec3<I>(&self, index: I) -> anyhow::Result<Vec3>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok(self.get_vector::<_, f32>(index)?.into())
    }

    pub fn set_vector<I, V>(&mut self, index: I, values: [V; 3]) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<VmScalar>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        let [v0, v1, v2] = values.map(|val| val.try_into());
        let values = [v0?, v1?, v2?];
        self.memory.set_vector(index, values)
    }

    pub fn set_vec3<I, V>(&mut self, index: I, values: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<[f32; 3]>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        let values = values.try_into()?;
        self.memory.set_vector(index, values.map(Into::into))
    }

    fn execute_internal(&mut self) -> anyhow::Result<[VmScalar; 3]> {
        let mut counter: usize = 0;

        loop {
            match self.execute_statement(self.instr(counter)?)?.into() {
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
