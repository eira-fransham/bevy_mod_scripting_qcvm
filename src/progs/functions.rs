use std::{borrow::Cow, fmt, ops::Range, sync::Arc};

use arrayvec::ArrayVec;
use bevy_mod_scripting_bindings::IntoNamespace;
use bevy_mod_scripting_bindings::{DynamicScriptFunction, DynamicScriptFunctionMut, FunctionKey};
use indexmap::IndexMap;
use num::FromPrimitive;
use num_derive::FromPrimitive;

use crate::QuakeCVm;
use crate::progs::{Opcode, ProgsError};

pub const MAX_ARGS: usize = 8;

#[derive(Copy, Clone, Debug)]
pub struct Statement {
    pub opcode: Opcode,
    pub arg1: i16,
    pub arg2: i16,
    pub arg3: i16,
}

impl Statement {
    pub fn new(op: i16, arg1: i16, arg2: i16, arg3: i16) -> anyhow::Result<Statement> {
        let opcode = match Opcode::from_i16(op) {
            Some(o) => o,
            None => return Err(ProgsError::with_msg(format!("Bad opcode 0x{op:x}"))),
        };

        Ok(Statement {
            opcode,
            arg1,
            arg2,
            arg3,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Float,
    EntityId,
    StringId,
    FunctionId,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::Float => write!(f, "float"),
            ScalarType::EntityId => write!(f, "entity"),
            ScalarType::StringId => write!(f, "string_t"),
            ScalarType::FunctionId => write!(f, "func_t"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Scalar(ScalarType),
    Vec3,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vec3 => write!(f, "vec3_t"),
            Self::Scalar(t) => t.fmt(f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RetType {
    Type(Type),
    Void,
}

impl fmt::Display for RetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::Type(t) => t.fmt(f),
        }
    }
}

/// Definition for a QuakeC function.
#[derive(Clone, Debug)]
pub struct QuakeCFunction {
    pub name: Cow<'static, str>,
    /// Range of globals to copy to the local stack.
    pub locals: Range<usize>,
    pub args: ArrayVec<Type, MAX_ARGS>,
    pub ret: Type,
    pub body: Arc<[Statement]>,
}

impl fmt::Display for QuakeCFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}(", self.ret, self.name)?;

        let mut arg_iter = self.args.iter();

        if let Some(arg) = arg_iter.next() {
            write!(f, "{arg}")?;
        }

        for arg in arg_iter {
            write!(f, ", {arg}")?;
        }

        write!(f, ");")?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum FunctionDef {
    Builtin(DynamicScriptFunction),
    // TODO: What's the difference between `DynamicScriptFunctionMut` and `DynamicScriptFunctionMut`?
    // They both require `&` to call.
    BuiltinMut(DynamicScriptFunctionMut),
    Progs(QuakeCFunction),
}

impl FunctionDef {
    pub fn function_key(&self) -> FunctionKey {
        match self {
            Self::Builtin(func) => FunctionKey {
                name: func.info.name.clone(),
                namespace: func.info.namespace,
            },
            Self::BuiltinMut(func) => FunctionKey {
                name: func.info.name.clone(),
                namespace: func.info.namespace,
            },
            Self::Progs(func) => FunctionKey {
                name: func.name.clone(),
                namespace: QuakeCVm::into_namespace(),
            },
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(func) => &func.info.name,
            Self::BuiltinMut(func) => &func.info.name,
            Self::Progs(func) => &func.name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Functions {
    defintions: IndexMap<Cow<'static, str>, Option<FunctionDef>>,
    /// Offset `i32` indices by this amount - builtins have negative indices.
    num_builtins: usize,
}

pub enum FunctionRef<'a> {
    Id(i32),
    Name(&'a str),
}

impl From<i32> for FunctionRef<'_> {
    fn from(value: i32) -> Self {
        Self::Id(value)
    }
}

impl<'a, T> From<&'a T> for FunctionRef<'a>
where
    T: AsRef<str>,
{
    fn from(value: &'a T) -> Self {
        Self::Name(value.as_ref())
    }
}

impl Functions {
    pub fn get<'a, F: Into<FunctionRef<'a>>>(&self, func: F) -> anyhow::Result<FunctionDef> {
        match func.into() {
            FunctionRef::Id(id) => self.by_index(id),
            FunctionRef::Name(name) => self.by_name(name),
        }
    }

    fn by_index(&self, value: i32) -> anyhow::Result<FunctionDef> {
        let index = if value < 0 {
            (1 - value) as usize
        } else {
            value as usize
        };
        self.defintions
            .get_index(index)
            .and_then(|(_, v)| v.clone())
            .ok_or_else(|| {
                ProgsError::with_msg(format!("Function at index {value} does not exist")).into()
            })
    }

    fn by_name(&self, name: &str) -> anyhow::Result<FunctionDef> {
        self.defintions
            .get(name)
            .and_then(Clone::clone)
            .ok_or_else(|| ProgsError::with_msg(format!("No function named {name}")).into())
    }
}
