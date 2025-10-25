use std::{
    any::Any,
    fmt::{self, Display},
};

use anyhow::bail;

use crate::{
    entity::ScalarFieldInfo,
    progs::{FieldDef, Scalar, ScalarType, Value},
};

pub trait Environment: 'static {
    type Function: Builtin<Env = Self>;
    type Context<'a>;
    type Entity;
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
        field: &ScalarFieldInfo,
    ) -> Result<Scalar, Self::Error>;

    fn get(
        &self,
        context: &<Self::Env as Environment>::Context<'_>,
        field: &FieldDef,
    ) -> anyhow::Result<Value> {
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
        field: &ScalarFieldInfo,
        value: Scalar,
    ) -> Result<(), Self::Error>;

    fn set(
        &self,
        context: &mut <Self::Env as Environment>::Context<'_>,
        field: &FieldDef,
        value: Value,
    ) -> anyhow::Result<()> {
        match (field.to_scalar(), value) {
            (Ok(scalar), Value::Scalar(scalar_val)) => {
                self.set_scalar(context, &scalar, scalar_val)?;
            }
            (Err(fields), Value::Vector(vec)) => {
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
    fn get_scalar(&self, context: &dyn Any, field: &ScalarFieldInfo) -> anyhow::Result<Scalar>;
    fn get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value> {
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
        context: &mut dyn Any,
        field: &ScalarFieldInfo,
        value: Scalar,
    ) -> anyhow::Result<()>;

    fn set(&self, context: &mut dyn Any, field: &FieldDef, value: Value) -> anyhow::Result<()> {
        match (field.to_scalar(), value) {
            (Ok(scalar), Value::Scalar(scalar_val)) => {
                self.set_scalar(context, &scalar, scalar_val)?;
            }
            (Err(fields), Value::Vector(vec)) => {
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

impl<T> ErasedEntity for T
where
    T: Entity + Any + DynEq,
{
    fn get_scalar(&self, context: &dyn Any, field: &ScalarFieldInfo) -> anyhow::Result<Scalar> {
        match context.downcast_ref() {
            Some(context) => Ok(self.get_scalar(context, field)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn get(&self, context: &dyn Any, field: &FieldDef) -> anyhow::Result<Value> {
        match context.downcast_ref() {
            Some(context) => Ok(self.get(context, field)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }

    fn set_scalar(
        &self,
        context: &mut dyn Any,
        field: &ScalarFieldInfo,
        value: Scalar,
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

    fn set(&self, context: &mut dyn Any, field: &FieldDef, value: Value) -> anyhow::Result<()> {
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

pub trait Builtin: QuakeCType {
    type Env: Environment;
    type Error: Display + std::error::Error + Send + Sync + 'static;

    fn call<I>(
        &self,
        context: &mut <Self::Env as Environment>::Context<'_>,
        args: I,
    ) -> Result<Scalar, Self::Error>
    where
        I: IntoIterator<Item = Value>,
        I::IntoIter: ExactSizeIterator;
}

pub trait ErasedBuiltin: QuakeCType + Any + DynEq {
    fn dyn_call(
        &self,
        context: &mut dyn Any,
        args: &mut dyn ExactSizeIterator<Item = Value>,
    ) -> anyhow::Result<Scalar>;
}

impl<T> ErasedBuiltin for T
where
    T: Builtin + DynEq,
{
    fn dyn_call(
        &self,
        context: &mut dyn Any,
        args: &mut dyn ExactSizeIterator<Item = Value>,
    ) -> anyhow::Result<Scalar> {
        match context.downcast_mut() {
            Some(context) => Ok(self.call(context, args)?),
            None => bail!(
                "Type mismatch for builtin context: expected {}, found {}",
                std::any::type_name::<<T::Env as Environment>::Context<'static>>(),
                std::any::type_name_of_val(context)
            ),
        }
    }
}
