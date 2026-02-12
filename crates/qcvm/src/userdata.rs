//! Types and traits related to host bindings for the QuakeC runtime.

use std::{
    any::Any,
    fmt,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use anyhow::bail;
use arrayvec::ArrayVec;
use bump_scope::BumpScope;

use crate::{
    ARG_ADDRS, AsErasedContext, BuiltinDef, CallArgs, ExecutionCtx, FunctionRef, MAX_ARGS,
    QuakeCArgs, QuakeCMemory, Type, Value,
    progs::{FieldDef, VectorField, VmScalar, VmValue},
};

/// User-provided global context. This is passed in to all functions and entity getters/setters.
pub trait Context {
    /// The type of entity handles.
    type Entity: ?Sized + EntityHandle<Context = Self>;
    /// The type of host-provided builtin functions.
    type Function: ?Sized + Function<Context = Self>;
    /// The error that is returned by [`Context::builtin`].
    type Error: std::error::Error;

    /// Given a function definition, get the builtin that it corresponds to (if one exists).
    fn builtin(&self, def: &BuiltinDef) -> Result<Arc<Self::Function>, Self::Error>;
}

impl Context for dyn ErasedContext {
    type Entity = dyn ErasedEntityHandle;
    type Function = dyn ErasedFunction;
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;

    fn builtin(&self, def: &BuiltinDef) -> Result<Arc<dyn ErasedFunction>, Self::Error> {
        self.dyn_builtin(def)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

/// A type-erased context that can be used for dynamic dispatch.
pub trait ErasedContext: Any {
    /// Dynamic version of [`Context::builtin`].
    fn dyn_builtin(&self, def: &BuiltinDef) -> anyhow::Result<Arc<dyn ErasedFunction>>;
}

impl fmt::Debug for &'_ mut dyn ErasedContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <&dyn ErasedContext>::fmt(&&**self, f)
    }
}

impl fmt::Debug for &'_ dyn ErasedContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "..")
    }
}

impl<T> ErasedContext for T
where
    T: Any + Context,
    T::Function: Sized + ErasedFunction,
{
    fn dyn_builtin(&self, def: &BuiltinDef) -> anyhow::Result<Arc<dyn ErasedFunction>> {
        Ok(self.builtin(def).map_err(|e| anyhow::format_err!("{e}"))? as Arc<dyn ErasedFunction>)
    }
}

/// The type of values that can be used in the QuakeC runtime.
///
/// > *TODO*: Implement proper userdata support.
pub trait QuakeCType: fmt::Debug {
    /// The QuakeC type of this value.
    fn type_(&self) -> Type;
    /// Whether this value should be considered null.
    fn is_null(&self) -> bool;
}

/// A dynamic form of [`PartialEq`] that can be used in type-erased contexts.
pub trait DynEq: Any {
    /// Dynamic version of [`PartialEq::eq`].
    fn dyn_eq(&self, other: &dyn Any) -> bool;
    /// Dynamic version of [`PartialEq::ne`].
    fn dyn_ne(&self, other: &dyn Any) -> bool {
        !self.dyn_eq(other)
    }
}

impl<T> DynEq for T
where
    T: PartialEq + Any,
{
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref().is_some_and(|other| self == other)
    }

    fn dyn_ne(&self, other: &dyn Any) -> bool {
        other.downcast_ref().is_some_and(|other| self != other)
    }
}

/// A handle to a host entity. This should _not_ be the type that stores the actual entity
/// data. All values in `qcvm` are immutable, the only mutable state is the context. This
/// should be implemented for a handle to an entity, with the entity data itself being stored
/// in the context.
pub trait EntityHandle: QuakeCType {
    /// The global context that holds the entity data.
    type Context: ?Sized + Context<Entity = Self>;
    /// The error returned from this
    type Error: std::error::Error;

    /// Get a field given this handle and reference to the context.
    fn get(&self, context: &Self::Context, field: &FieldDef) -> Result<Value, Self::Error>;

    /// Set a field given this handle and a mutable reference to the context.
    fn set(
        &self,
        context: &mut Self::Context,
        field: &FieldDef,
        offset: Option<VectorField>,
        value: Value,
    ) -> Result<(), Self::Error>;
}

/// Type-erased form of [`EntityHandle`], for dynamic dispatch.
pub trait ErasedEntityHandle: DynEq + Send + Sync + fmt::Debug {
    /// Dynamic form of [`EntityHandle::get`].
    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value>;

    /// Dynamic form of [`EntityHandle::set`].
    fn dyn_set(
        &self,
        context: &mut dyn Any,
        field: &FieldDef,
        offset: Option<VectorField>,
        value: Value,
    ) -> anyhow::Result<()>;
}

impl<T> ErasedEntityHandle for T
where
    T: EntityHandle + Send + Sync + PartialEq + Any,
    T::Context: Sized,
{
    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value> {
        match context.downcast_ref() {
            Some(context) => Ok(self
                .get(context, field)
                .map_err(|e| anyhow::format_err!("{e}"))?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<T::Context>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn dyn_set(
        &self,
        context: &mut dyn Any,
        field: &FieldDef,
        offset: Option<VectorField>,
        value: Value,
    ) -> anyhow::Result<()> {
        match context.downcast_mut() {
            Some(context) => Ok(self
                .set(context, field, offset, value)
                .map_err(|e| anyhow::format_err!("{e}"))?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<T::Context>(),
                std::any::type_name_of_val(context)
            ),
        }
    }
}

/// A function callable from QuakeC code. This may call further internal functions, and is
/// passed a configurable context type.
pub trait Function: QuakeCType {
    /// The user-provided context.
    type Context: ?Sized + Context<Function = Self>;
    /// The error returned by the methods in this trait.
    type Error: std::error::Error;

    /// Get the signature of the function. Note that only this number of arguments will
    /// be passed to the function.
    fn signature(&self) -> Result<ArrayVec<Type, MAX_ARGS>, Self::Error>;
    /// Call the function.
    fn call(&self, context: FnCall<'_, Self::Context>) -> Result<Value, Self::Error>;
}

/// The function call context, containing both the user context and the current VM context
/// (which can be used to call into QuakeC functions).
pub struct FnCall<'a, T: ?Sized = dyn ErasedContext> {
    pub(crate) execution:
        ExecutionCtx<'a, T, BumpScope<'a>, CallArgs<ArrayVec<VmScalar, MAX_ARGS>>>,
}

impl<T: ?Sized> Deref for FnCall<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.execution.context
    }
}

impl<T: ?Sized> DerefMut for FnCall<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.execution.context
    }
}

impl<T: ?Sized> FnCall<'_, T> {
    /// Explicitly get a mutable reference to the user-provided context. This
    /// is also available via the `Deref`/`DerefMut` implementation
    pub fn context_mut(&mut self) -> &mut T {
        self.execution.context
    }
}

impl<T> FnCall<'_, T>
where
    T: ?Sized + AsErasedContext,
{
    /// Get an iterator of the arguments to this function. For now, the signature
    /// must be explicitly provided.
    ///
    /// > TODO: The signature should not need to be passed here.
    pub fn arguments(&self, args: &[Type]) -> impl Iterator<Item = Value> {
        ARG_ADDRS.step_by(3).zip(args).map(|(i, ty)| match ty {
            Type::Vector => {
                let [x, y, z] = self.execution.memory.get_vector(i).unwrap();
                let vec = [
                    x.try_into().unwrap(),
                    y.try_into().unwrap(),
                    z.try_into().unwrap(),
                ];
                self.execution.to_value(VmValue::Vector(vec)).unwrap()
            }
            _ => {
                let value = self.execution.memory.get(i).unwrap().into();
                self.execution.to_value(value).unwrap()
            }
        })
    }
}

impl<'a> FnCall<'a, dyn ErasedContext> {
    fn downcast<T>(self) -> Option<FnCall<'a, T>>
    where
        T: Any,
    {
        Some(FnCall {
            execution: self.execution.downcast()?,
        })
    }
}

impl<'a, T> FnCall<'a, T>
where
    T: ErasedContext,
{
    /// Call a QuakeC function by index or name.
    pub fn call<A, F>(&mut self, function_ref: F, args: A) -> anyhow::Result<Value>
    where
        F: Into<FunctionRef>,
        A: QuakeCArgs,
    {
        let function_def = self
            .execution
            .functions
            .get(function_ref)?
            .clone()
            .try_into_qc()
            .map_err(|def| {
                anyhow::format_err!("Function {:?} is not a QuakeC function", def.name)
            })?;

        self.execution
            .with_args(&function_def.name, args, |mut exec| {
                Ok(exec.execute_def(&function_def)?.try_into()?)
            })
    }
}

/// Type-erased version of [`Function`], for dynamic dispatch.
pub trait ErasedFunction: QuakeCType + Send + Sync + DynEq {
    /// Dynamic version of [`Function::signature`].
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<Type, MAX_ARGS>>;

    /// Dynamic version of [`Function::call`].
    fn dyn_call<'a, 'b>(&'a self, context: FnCall<'b>) -> anyhow::Result<Value>;
}

impl PartialEq for dyn ErasedFunction {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}

impl Function for dyn ErasedFunction {
    type Context = dyn ErasedContext;
    // For some reason, `Box<E> where E: Error` only implements `Error` itself if `E` is sized,
    // but `Arc` allows unsized inner values.
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;

    fn signature(&self) -> Result<ArrayVec<Type, MAX_ARGS>, Self::Error> {
        self.dyn_signature()
            .map_err(|e| e.into_boxed_dyn_error().into())
    }

    fn call(&self, context: FnCall<Self::Context>) -> Result<Value, Self::Error> {
        self.dyn_call(context)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl QuakeCType for dyn ErasedEntityHandle {
    fn type_(&self) -> Type {
        Type::Entity
    }

    fn is_null(&self) -> bool {
        false
    }
}

impl PartialEq for dyn ErasedEntityHandle {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl EntityHandle for dyn ErasedEntityHandle {
    type Context = dyn ErasedContext;
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;

    fn get(&self, context: &Self::Context, field: &FieldDef) -> Result<Value, Self::Error> {
        self.dyn_get(context, field)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }

    fn set(
        &self,
        context: &mut Self::Context,
        field: &FieldDef,
        offset: Option<VectorField>,
        value: Value,
    ) -> Result<(), Self::Error> {
        self.dyn_set(context, field, offset, value)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl<T> ErasedFunction for T
where
    T: Function + PartialEq + Any + Send + Sync,
    T::Context: Sized,
{
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<Type, MAX_ARGS>> {
        self.signature().map_err(|e| anyhow::format_err!("{e}"))
    }

    fn dyn_call(&self, context: FnCall) -> anyhow::Result<Value> {
        let type_name = std::any::type_name_of_val(context.execution.context);
        match context.downcast() {
            Some(context) => Ok(self.call(context).map_err(|e| anyhow::format_err!("{e}"))?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<T::Context>(),
                type_name
            ),
        }
    }
}
