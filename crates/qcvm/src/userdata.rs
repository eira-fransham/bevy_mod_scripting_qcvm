use std::{
    any::Any,
    fmt::{self, Display},
    sync::Arc,
};

use anyhow::bail;
use arrayvec::ArrayVec;
use bump_scope::BumpScope;

use crate::{
    ARG_ADDRS, ArgType, CallArgs, ExecutionCtx, QuakeCMemory, Value,
    progs::{
        FieldDef, FieldOffset, ScalarType, VmScalar, VmValue,
        functions::{BuiltinDef, MAX_ARGS},
    },
};

pub trait Context {
    type Entity: ?Sized + Entity;
    type Function: ?Sized + Function;
    type Error: std::error::Error + Send + Sync + 'static;

    fn builtin(&self, def: &BuiltinDef) -> Result<Arc<Self::Function>, Self::Error>;
}

impl Context for dyn ErasedContext {
    type Entity = dyn ErasedEntity;
    type Function = dyn ErasedFunction;
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;

    fn builtin(&self, def: &BuiltinDef) -> Result<Arc<dyn ErasedFunction>, Self::Error> {
        self.dyn_builtin(def)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

pub trait ErasedContext: Any {
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
        Ok(self.builtin(def)? as Arc<dyn ErasedFunction>)
    }
}

pub trait QuakeCType: fmt::Debug {
    fn type_(&self) -> ScalarType;
    fn is_null(&self) -> bool;
}

pub trait DynEq: Any {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
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

pub trait Entity: QuakeCType {
    type Context: ?Sized + Context;
    type Error: Display + std::error::Error + Send + Sync + 'static;

    fn get(&self, context: &Self::Context, field: &FieldDef) -> Result<Value, Self::Error>;

    fn set(
        &self,
        context: &mut Self::Context,
        field: &FieldDef,
        offset: Option<FieldOffset>,
        value: Value,
    ) -> Result<(), Self::Error>;
}

pub trait ErasedEntity: DynEq + fmt::Debug {
    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value>;

    fn dyn_set(
        &self,
        context: &mut dyn Any,
        field: &FieldDef,
        offset: Option<FieldOffset>,
        value: Value,
    ) -> anyhow::Result<()>;
}

impl<T> ErasedEntity for T
where
    T: Entity + PartialEq + Any,
    T::Context: Sized,
{
    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value> {
        match context.downcast_ref() {
            Some(context) => Ok(self.get(context, field)?),
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
        offset: Option<FieldOffset>,
        value: Value,
    ) -> anyhow::Result<()> {
        match context.downcast_mut() {
            Some(context) => Ok(self.set(context, field, offset, value)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<T::Context>(),
                std::any::type_name_of_val(context)
            ),
        }
    }
}

pub trait Function: QuakeCType {
    type Context: ?Sized + Context;
    type Error: std::error::Error + Send + Sync + 'static;
    type Output: Into<Value>;

    fn signature(&self) -> Result<ArrayVec<ArgType, MAX_ARGS>, Self::Error>;
    fn call(&self, context: FnCall<'_, Self::Context>) -> Result<Self::Output, Self::Error>;
}

pub struct FnCall<'a, T: ?Sized = dyn ErasedContext> {
    pub(crate) execution:
        ExecutionCtx<'a, T, BumpScope<'a>, CallArgs<ArrayVec<VmScalar, MAX_ARGS>>>,
}

impl<T> FnCall<'_, T> {
    pub fn context(&mut self) -> &mut T {
        self.execution.context
    }
}

impl<T> FnCall<'_, T>
where
    T: ?Sized + ErasedContext,
{
    pub fn arguments(&self, args: &[ArgType]) -> impl Iterator<Item = Value> {
        ARG_ADDRS.step_by(3).zip(args).map(|(i, ty)| match ty {
            ArgType::Vector => {
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
    pub fn into_erased(self) -> FnCall<'a> {
        FnCall {
            execution: self.execution.into_erased(),
        }
    }
}

pub trait ErasedFunction: QuakeCType + Any + DynEq {
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<ArgType, MAX_ARGS>>;

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
    type Output = Value;

    fn signature(&self) -> Result<ArrayVec<ArgType, MAX_ARGS>, Self::Error> {
        self.dyn_signature()
            .map_err(|e| e.into_boxed_dyn_error().into())
    }

    fn call(&self, context: FnCall<Self::Context>) -> Result<Self::Output, Self::Error> {
        self.dyn_call(context)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl QuakeCType for dyn ErasedEntity {
    fn type_(&self) -> ScalarType {
        ScalarType::Entity
    }

    fn is_null(&self) -> bool {
        false
    }
}

impl PartialEq for dyn ErasedEntity {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Entity for dyn ErasedEntity {
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
        offset: Option<FieldOffset>,
        value: Value,
    ) -> Result<(), Self::Error> {
        self.dyn_set(context, field, offset, value)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl<T> ErasedFunction for T
where
    T: Function + PartialEq + Any,
    T::Context: Sized,
{
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<ArgType, MAX_ARGS>> {
        Ok(self.signature()?)
    }

    fn dyn_call(&self, context: FnCall) -> anyhow::Result<Value> {
        let type_name = std::any::type_name_of_val(context.execution.context);
        match context.downcast() {
            Some(context) => Ok(self.call(context)?.into()),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<T::Context>(),
                type_name
            ),
        }
    }
}
