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
use snafu::Snafu;

use crate::{
    Address, AsErasedContext, BuiltinDef, CallArgs, ExecutionCtx, FunctionRef, MAX_ARGS, QCMemory,
    QCParams, Type, Value, function_args,
    progs::{VmScalar, VmValue},
};

/// A type-erased entity handle.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct ErasedEntityHandle(pub u64);

type ErasedAddr = u16;

/// An error when getting/setting an address
#[derive(Snafu, Debug, Copy, Clone, PartialEq, Eq)]
pub enum AddrError<E> {
    /// The address does not exist
    OutOfRange,
    /// Another error occurred
    Other {
        /// The underlying error.
        error: E,
    },
}

impl<E> From<E> for AddrError<E> {
    fn from(value: E) -> Self {
        Self::Other { error: value }
    }
}

impl<E> AddrError<E>
where
    E: fmt::Display,
{
    pub(crate) fn into_anyhow(self) -> AddrError<anyhow::Error> {
        {
            match self {
                Self::OutOfRange => AddrError::OutOfRange,
                Self::Other { error: e } => AddrError::Other {
                    error: anyhow::format_err!("{e}"),
                },
            }
        }
    }
}

/// User-provided global context. This is passed in to all functions and entity getters/setters.
pub trait Context {
    /// The type of entity handles.
    // TODO: It might be better to somehow handle this in a way that doesn't require wrapping it in an `Arc`,
    //       as usually entity handles will just be a simple number.
    type Entity: ?Sized + EntityHandle<Context = Self>;
    /// The type of host-provided builtin functions.
    type Function: ?Sized + Function<Context = Self>;
    /// The error that is returned by [`Context::builtin`].
    type Error: std::error::Error;
    /// The type representing valid globals
    type GlobalAddr: Address;

    /// Given a function definition, get the builtin that it corresponds to (if one exists).
    fn builtin(&self, def: &BuiltinDef) -> Result<Arc<Self::Function>, Self::Error>;

    /// Get a global with the given definition
    fn global(&self, def: Self::GlobalAddr) -> Result<Value, AddrError<Self::Error>>;

    /// Implement the `OP_STATE` opcode.
    ///
    /// This expected behavior is as follows:
    ///
    /// - Set `self.nextthink` to `self.time + delta_time`, where `delta_time` is the time between
    ///   frames.
    /// - Set `self.frame` to the first argument.
    /// - Set `self.think` to the second argument.
    fn state(&self, _frame: f32, _think_fn: Arc<dyn ErasedFunction>) -> Result<(), Self::Error> {
        unimplemented!("`OP_STATE` not available in this environment");
    }

    /// Set a global with the given definition
    fn set_global(
        &mut self,
        def: Self::GlobalAddr,
        value: Value,
    ) -> Result<(), AddrError<Self::Error>>;
}

/// A type-erased context that can be used for dynamic dispatch.
pub trait ErasedContext: Any {
    /// Dynamic version of [`Context::builtin`].
    fn dyn_builtin(&self, def: &BuiltinDef) -> anyhow::Result<Arc<dyn ErasedFunction>>;

    /// Dynamic version of [`Context::state`].
    fn dyn_state(&self, _frame: f32, _think_fn: Arc<dyn ErasedFunction>) -> anyhow::Result<()> {
        anyhow::bail!("`OP_STATE` not available in this environment")
    }

    /// Dynamic version of `<Context::Entity as EntityHandle>::get`.
    fn dyn_entity_get(
        &self,
        erased_ent: u64,
        field: ErasedAddr,
        ty: Type,
    ) -> anyhow::Result<Value, AddrError<anyhow::Error>>;

    /// Dynamic version of `<Context::Entity as EntityHandle>::set`.
    fn dyn_entity_set(
        &mut self,
        erased_ent: u64,
        field: ErasedAddr,
        value: Value,
    ) -> anyhow::Result<(), AddrError<anyhow::Error>>;

    /// Dynamic version of [`Context::global`]
    fn dyn_global(&self, def: ErasedAddr, ty: Type) -> Result<Value, AddrError<anyhow::Error>>;

    /// Dynamic version of [`Context::set_global`]
    fn dyn_set_global(
        &mut self,
        def: ErasedAddr,
        value: Value,
    ) -> Result<(), AddrError<anyhow::Error>>;
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

    fn dyn_entity_get(
        &self,
        erased_ent: u64,
        field: ErasedAddr,
        ty: Type,
    ) -> anyhow::Result<Value, AddrError<anyhow::Error>> {
        let field_addr =
            <<T::Entity as EntityHandle>::FieldAddr as Address>::from_u16_typed(field, ty)
                .ok_or(AddrError::OutOfRange)?;
        <T::Entity as EntityHandle>::from_erased(erased_ent, |ent| ent.get(self, field_addr))
            .map_err(|e| anyhow::format_err!("{e}"))?
            .map_err(|e| AddrError::Other {
                error: anyhow::format_err!("{e}"),
            })
    }

    fn dyn_entity_set(
        &mut self,
        erased_ent: u64,
        field: ErasedAddr,
        value: Value,
    ) -> Result<(), AddrError<anyhow::Error>> {
        let field_addr = <<T::Entity as EntityHandle>::FieldAddr as Address>::from_u16_typed(
            field,
            value.type_(),
        )
        .ok_or(AddrError::OutOfRange)?;
        <T::Entity as EntityHandle>::from_erased(erased_ent, |ent| ent.set(self, field_addr, value))
            .map_err(|e| anyhow::format_err!("{e}"))?
            .map_err(|e| AddrError::Other {
                error: anyhow::format_err!("{e}"),
            })
    }

    fn dyn_global(&self, def: ErasedAddr, ty: Type) -> Result<Value, AddrError<anyhow::Error>> {
        self.global(
            <T as Context>::GlobalAddr::from_u16_typed(def, ty).ok_or(AddrError::OutOfRange)?,
        )
        .map_err(AddrError::into_anyhow)
    }

    fn dyn_set_global(
        &mut self,
        def: ErasedAddr,
        value: Value,
    ) -> Result<(), AddrError<anyhow::Error>> {
        self.set_global(
            <T as Context>::GlobalAddr::from_u16_typed(def, value.type_())
                .ok_or(AddrError::OutOfRange)?,
            value,
        )
        .map_err(AddrError::into_anyhow)
    }
}

/// The type of values that can be used in the QuakeC runtime.
///
/// > *TODO*: Implement proper userdata support.
pub trait QCType: fmt::Debug {
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
pub trait EntityHandle: QCType {
    /// The global context that holds the entity data.
    type Context: ?Sized + Context<Entity = Self>;
    /// The error returned from this
    type Error: std::error::Error;
    /// The type representing fields
    type FieldAddr: Address;

    /// Convert from an opaque handle to a reference to this type (must be a reference in order to allow unsized handles)
    fn from_erased<F, O>(erased: u64, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&Self) -> O,
    {
        Self::from_erased_mut(erased, |this| callback(&*this))
    }

    /// Convert from an opaque handle to a mutable reference to this type (must be a reference in order to allow unsized handles)
    fn from_erased_mut<F, O>(erased: u64, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&mut Self) -> O;

    /// Convert this type to an opaque handle
    fn to_erased(&self) -> u64;

    /// Get a field given this handle and reference to the context.
    fn get(
        &self,
        context: &Self::Context,
        field: Self::FieldAddr,
    ) -> Result<Value, AddrError<Self::Error>>;

    /// Set a field given this handle and a mutable reference to the context.
    fn set(
        &self,
        context: &mut Self::Context,
        field: Self::FieldAddr,
        value: Value,
    ) -> Result<(), AddrError<Self::Error>>;
}

/// A function callable from QuakeC code. This may call further internal functions, and is
/// passed a configurable context type.
pub trait Function: QCType {
    /// The user-provided context.
    type Context: ?Sized + Context<Function = Self>;
    /// The error returned by the methods in this trait.
    type Error: std::error::Error;

    /// Get the signature of the function. Note that only this number of arguments will
    /// be passed to the function.
    ///
    /// > TODO: It may be useful to annotate the return type, too.
    fn signature(&self) -> Result<ArrayVec<Type, MAX_ARGS>, Self::Error>;

    /// Call the function.
    fn call(&self, context: FnCall<'_, Self::Context>) -> Result<Value, Self::Error>;
}

/// The function call context, containing both the user context and the current VM context
/// (which can be used to call into QuakeC functions).
pub struct FnCall<'a, T: ?Sized = dyn ErasedContext> {
    pub(crate) execution:
        ExecutionCtx<'a, T, BumpScope<'a>, CallArgs<ArrayVec<[VmScalar; 3], MAX_ARGS>>>,
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
        function_args()
            .into_iter()
            .zip(args)
            .map(|(i, ty)| match ty {
                Type::Vector => {
                    let [x, y, z] = self.execution.memory.get_vector(i.addr as _).unwrap();
                    let vec = [
                        x.try_into().unwrap(),
                        y.try_into().unwrap(),
                        z.try_into().unwrap(),
                    ];
                    self.execution.to_value(VmValue::Vector(vec)).unwrap()
                }
                _ => {
                    let value = self.execution.memory.get(i.addr as _).unwrap().into();
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
        A: QCParams,
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
pub trait ErasedFunction: QCType + Send + Sync + DynEq {
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

impl QCType for ErasedEntityHandle {
    fn type_(&self) -> Type {
        Type::Entity
    }

    fn is_null(&self) -> bool {
        false
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
