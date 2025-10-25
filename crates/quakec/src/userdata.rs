use std::{
    any::Any,
    fmt::{self, Display},
    sync::Arc,
};

use anyhow::bail;
use arrayvec::ArrayVec;
use bump_scope::BumpScope;

use crate::{
    ArgType, CallArgs, ExecutionCtx, Value,
    entity::ScalarFieldDef,
    progs::{
        FieldDef, ScalarType, VmScalar, VmValue,
        functions::{BuiltinDef, MAX_ARGS},
    },
};

pub struct ErasedEnvironment;

impl Environment for ErasedEnvironment {
    type Function = dyn ErasedFunction;
    type Context<'a> = dyn ErasedContext;
    type Entity = dyn ErasedEntity;
}

pub trait Environment: 'static {
    type Function: ?Sized + Function;
    type Context<'a>: ?Sized + Context;
    type Entity: ?Sized + Entity;
}

pub trait Context {
    type Env: Environment<Function = Self::Function>;
    type Function: ?Sized + Function<Env = Self::Env>;
    type Error: std::error::Error + Send + Sync + 'static;

    fn builtin(
        &self,
        def: &BuiltinDef,
    ) -> Result<Arc<<Self::Env as Environment>::Function>, Self::Error>;
}

impl Context for dyn ErasedContext {
    type Env = ErasedEnvironment;
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
    <T::Env as Environment>::Function: Sized + ErasedFunction,
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
    type Env: Environment;
    type Error: Display + std::error::Error + Send + Sync + 'static;

    fn get_scalar(
        &self,
        context: &<Self::Env as Environment>::Context<'_>,
        field: &ScalarFieldDef,
    ) -> Result<VmScalar, Self::Error>;

    fn get(
        &self,
        context: &<Self::Env as Environment>::Context<'_>,
        field: &FieldDef,
    ) -> anyhow::Result<VmValue> {
        match field.to_scalar() {
            Ok(scalar) => Ok(self.get_scalar(context, &scalar)?.into()),
            Err(fields) => Ok(fields
                .each_ref()
                .try_map(|field| -> anyhow::Result<f32> {
                    Ok(self.get_scalar(context, field)?.try_into()?)
                })?
                .into()),
        }
    }

    fn set_scalar(
        &self,
        context: &mut <Self::Env as Environment>::Context<'_>,
        field: &ScalarFieldDef,
        value: VmScalar,
    ) -> Result<(), Self::Error>;

    fn set(
        &self,
        context: &mut <Self::Env as Environment>::Context<'_>,
        field: &FieldDef,
        value: VmValue,
    ) -> anyhow::Result<()> {
        match (field.to_scalar(), value) {
            (Ok(scalar), VmValue::Scalar(scalar_val)) => {
                self.set_scalar(context, &scalar, scalar_val)?;
            }
            (Err(fields), VmValue::Vector(vec)) => {
                for (field, float) in fields.iter().zip(vec) {
                    self.set_scalar(&mut *context, field, float.into())?;
                }
            }
            (_, value) => {
                bail!(
                    "Type mismatch when setting entity.{}: expected {}, found {}",
                    field.name.to_string_lossy(),
                    value.type_(),
                    field.type_,
                );
            }
        }

        Ok(())
    }
}

pub trait ErasedEntity: QuakeCType + DynEq {
    fn dyn_get_scalar(&self, context: &dyn Any, field: &ScalarFieldDef)
    -> anyhow::Result<VmScalar>;
    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<VmValue> {
        match field.to_scalar() {
            Ok(scalar) => Ok(self.dyn_get_scalar(context, &scalar)?.into()),
            Err(fields) => Ok(fields
                .each_ref()
                .try_map(|field| -> anyhow::Result<f32> {
                    Ok(self.dyn_get_scalar(context, field)?.try_into()?)
                })?
                .into()),
        }
    }

    fn dyn_set_scalar(
        &self,
        context: &mut dyn Any,
        field: &ScalarFieldDef,
        value: VmScalar,
    ) -> anyhow::Result<()>;

    fn dyn_set(
        &self,
        context: &mut dyn Any,
        field: &FieldDef,
        value: VmValue,
    ) -> anyhow::Result<()> {
        match (field.to_scalar(), value) {
            (Ok(scalar), VmValue::Scalar(scalar_val)) => {
                self.dyn_set_scalar(context, &scalar, scalar_val)?;
            }
            (Err(fields), VmValue::Vector(vec)) => {
                for (field, float) in fields.iter().zip(vec) {
                    self.dyn_set_scalar(&mut *context, field, float.into())?;
                }
            }
            (_, value) => {
                bail!(
                    "Type mismatch when setting entity.{}: expected {}, found {}",
                    field.name.to_string_lossy(),
                    value.type_(),
                    field.type_,
                );
            }
        }

        Ok(())
    }
}

impl<T> ErasedEntity for T
where
    T: Entity + PartialEq + Any,
    for<'a> <T::Env as Environment>::Context<'a>: Sized,
{
    fn dyn_get_scalar(
        &self,
        context: &dyn Any,
        field: &ScalarFieldDef,
    ) -> anyhow::Result<VmScalar> {
        match context.downcast_ref() {
            Some(context) => Ok(self.get_scalar(context, field)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn dyn_get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<VmValue> {
        match context.downcast_ref() {
            Some(context) => Ok(self.get(context, field)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn dyn_set_scalar(
        &self,
        context: &mut dyn Any,
        field: &ScalarFieldDef,
        value: VmScalar,
    ) -> anyhow::Result<()> {
        match context.downcast_mut() {
            Some(context) => Ok(self.set_scalar(context, field, value)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn dyn_set(
        &self,
        context: &mut dyn Any,
        field: &FieldDef,
        value: VmValue,
    ) -> anyhow::Result<()> {
        match context.downcast_mut() {
            Some(context) => Ok(self.set(context, field, value)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }
}

pub trait Function: QuakeCType {
    type Env: Environment;
    type Error: std::error::Error + Send + Sync + 'static;
    type Output: Into<Value>;

    fn signature(&self) -> Result<ArrayVec<ArgType, MAX_ARGS>, Self::Error>;
    fn call(
        &self,
        context: FnCall<<Self::Env as Environment>::Context<'_>>,
    ) -> Result<Self::Output, Self::Error>;
}

pub struct FnCall<'a, T: ?Sized = dyn ErasedContext> {
    pub(crate) execution:
        ExecutionCtx<'a, T, BumpScope<'a>, CallArgs<ArrayVec<VmScalar, MAX_ARGS>>>,
}

impl<T> FnCall<'_, T>
where
    T: Any,
{
    pub fn context(&mut self) -> &mut T {
        self.execution.context
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
        std::ptr::eq(self, other)
    }
}

impl Function for dyn ErasedFunction {
    type Env = ErasedEnvironment;
    // For some reason, `Box<E> where E: Error` only implements `Error` itself if `E` is sized,
    // but `Arc` allows unsized inner values.
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;
    type Output = Value;

    fn signature(&self) -> Result<ArrayVec<ArgType, MAX_ARGS>, Self::Error> {
        todo!()
    }

    fn call(
        &self,
        context: FnCall<<Self::Env as Environment>::Context<'_>>,
    ) -> Result<Self::Output, Self::Error> {
        self.dyn_call(context)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl PartialEq for dyn ErasedEntity {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Entity for dyn ErasedEntity {
    type Env = ErasedEnvironment;
    type Error = Arc<dyn std::error::Error + Send + Sync + 'static>;

    fn get_scalar(
        &self,
        context: &<Self::Env as Environment>::Context<'_>,
        field: &ScalarFieldDef,
    ) -> Result<VmScalar, Self::Error> {
        self.dyn_get_scalar(context, field)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }

    fn set_scalar(
        &self,
        context: &mut <Self::Env as Environment>::Context<'_>,
        field: &ScalarFieldDef,
        value: VmScalar,
    ) -> Result<(), Self::Error> {
        self.dyn_set_scalar(context, field, value)
            .map_err(|e| e.into_boxed_dyn_error().into())
    }
}

impl<T> ErasedFunction for T
where
    T: Function + PartialEq + Any,
    for<'a> <T::Env as Environment>::Context<'a>: Sized,
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
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                type_name
            ),
        }
    }
}
