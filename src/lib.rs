// TODO: Ideally we would not use nightly
#![feature(iter_map_windows, array_try_map)]

use std::{borrow::Cow, ffi::CStr, fmt, num::NonZeroIsize, ops::Range, sync::Arc};

use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{In, SystemId},
};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{FunctionCallContext, ScriptFunctionRegistry};
use bump_scope::BumpAllocatorScopeExt;
use hashbrown::HashMap;

use crate::{
    entity::EntityTypeDef,
    progs::{
        FieldOffset, ScalarKind, StringTable, ValueKind,
        functions::{
            FunctionExecutionCtx, FunctionRegistry, QuakeCFunctionDef,
            Statement,
        },
        globals::GlobalRegistry,
    },
};

pub mod entity;
pub mod ops;
pub mod progs;

#[cfg(feature = "quake1")]
pub mod quake1;

const FUNCTION_CALL_CONTEXT: FunctionCallContext =
    FunctionCallContext::new(Language::External(Cow::Borrowed("quakec")));

/// Server-side level state.
#[derive(Component, Debug)]
pub struct QuakeCVm {
    /// Global values for QuakeC bytecode.
    globals: GlobalRegistry,

    entity_def: EntityTypeDef,

    string_table: StringTable,

    /// Function definitions and bodies.
    functions: FunctionRegistry,

    entity_fields: HashMap<Arc<CStr>, SystemId<In<(Entity, FieldOffset)>, ValueKind>>,
}

#[derive(Debug)]
struct BacktraceFrame<'a> {
    function: &'a CStr,
    prev: Option<&'a BacktraceFrame<'a>>,
}

impl fmt::Display for BacktraceFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.function.to_string_lossy());

        if let Some(prev) = self.prev {
            prev.fmt(f)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct CallContext<'a> {
    builtins: &'a ScriptFunctionRegistry,
}

#[derive(Debug)]
struct ExecutionCtx<'a> {
    memory: FunctionExecutionCtx<'a>,
    backtrace: BacktraceFrame<'a>,
    // call_context: &'a CallContext<'a>,
    globals: &'a GlobalRegistry,
    entity_def: &'a EntityTypeDef,
    string_table: &'a StringTable,
    functions: &'a FunctionRegistry,
    /// Technically, every QuakeC function returns 3 scalars of arbitrary types.
    /// Only one return value is available at once. We store it here in order
    /// to allow us to access it in `get` etc.
    last_ret: Option<[ScalarKind; 3]>,
}

enum OpResult {
    Jump(NonZeroIsize),
    Ret([ScalarKind; 3]),
    Continue,
}

impl From<()> for OpResult {
    fn from(_: ()) -> Self {
        Self::Continue
    }
}

impl From<[ScalarKind; 3]> for OpResult {
    fn from(value: [ScalarKind; 3]) -> Self {
        Self::Ret(value)
    }
}

const ARG_ADDRS: Range<usize> = 4..28;
const RETURN_ADDRS: Range<usize> = 1..4;

impl ExecutionCtx<'_> {
    pub fn enter<'scope>(
        &'scope self,
        alloc: impl BumpAllocatorScopeExt<'scope>,
        function: &'scope QuakeCFunctionDef,
    ) -> anyhow::Result<ExecutionCtx<'scope>> {
        let mut out = ExecutionCtx {
            memory: function.ctx(alloc),
            backtrace: BacktraceFrame {
                function: &function.name,
                prev: Some(&self.backtrace),
            },
            globals: self.globals,
            entity_def: self.entity_def,
            string_table: self.string_table,
            functions: self.functions,
            last_ret: None,
        };

        for (src, dst) in ARG_ADDRS.zip(function.body.locals.clone()) {
            out.set(dst, self.get::<_, ScalarKind>(src)?)?;
        }

        Ok(out)
    }

    pub fn backtrace(&self) -> impl Iterator<Item = &'_ CStr> + '_ {
        std::iter::successors(Some(&self.backtrace), |cur| cur.prev).map(|frame| frame.function)
    }

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

    pub fn instr(&self, pc: usize) -> anyhow::Result<Statement> {
        self.memory.instr(pc)
    }

    pub fn get<I, O>(&self, index: I) -> anyhow::Result<O>
    where
        I: TryInto<usize>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        ScalarKind: TryInto<O>,
        <ScalarKind as TryInto<O>>::Error:
            snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        if RETURN_ADDRS.contains(&index) {
            return Ok(self
                .last_ret
                .as_ref()
                .and_then(|val| val.get(index.checked_sub(RETURN_ADDRS.start)?).cloned())
                .ok_or_else(|| {
                    anyhow::Error::msg("Tried to read return values before calling a function")
                })?
                .try_into()?);
        }

        match self.memory.get(index)? {
            Some(val) => Ok(val.try_into()?),
            None => Ok(self.globals.get_value(index)?.try_into()?),
        }
    }

    pub fn get_vector<I, O>(&self, index: I) -> anyhow::Result<[O; 3]>
    where
        I: TryInto<usize>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        ScalarKind: TryInto<O>,
        <ScalarKind as TryInto<O>>::Error:
            snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok([
            self.get(index)?.try_into()?,
            self.get(index + 1)?.try_into()?,
            self.get(index + 2)?.try_into()?,
        ])
    }

    pub fn get_vec3<I>(&self, index: I) -> anyhow::Result<Vec3>
    where
        I: TryInto<usize>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok(self.get_vector::<_, f32>(index)?.into())
    }

    pub fn set<I, V>(&mut self, index: I, value: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        V: Into<ScalarKind>,
    {
        let index = index.try_into()?;
        let value = value.into();
        self.memory.set(index, value)
    }

    pub fn set_vector<I, V>(&mut self, index: I, values: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        V: TryInto<[f32; 3]>,
        V::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        let values = values.try_into()?;
        self.memory.set_vector(index, values.map(Into::into))
    }

    pub fn execute(&mut self) -> anyhow::Result<[ScalarKind; 3]> {
        let mut counter: usize = 0;

        loop {
            // *self.runaway.as_mut() -= 1;

            // if *self.runaway.as_mut() == 0 {
            //     self.print_backtrace(false);
            //     return Err(ProgsError::LocalStackOverflow {
            //         backtrace: Backtrace::capture(),
            //     }
            //     .into());
            // }

            match self.execute_statement(self.instr(counter)?)? {
                OpResult::Jump(idx) => {
                    counter = counter
                        .checked_add_signed(idx.get())
                        .ok_or_else(|| anyhow::Error::msg("Out-of-bounds instruction access"))?;
                }
                OpResult::Continue => {
                    counter += 1;
                }
                OpResult::Ret(vals) => return Ok(vals),
            }
        }
    }
}
