use std::ffi::CStr;
use std::{fmt, ops::Range, sync::Arc};

use arc_slice::ArcSlice;
use arrayvec::ArrayVec;
use bevy_mod_scripting_bindings::{DynamicScriptFunction, DynamicScriptFunctionMut};
use bevy_reflect::{Reflect, reflect_remote};
use bump_scope::{BumpAllocatorScopeExt, FixedBumpVec};
use hashbrown::HashMap;
use num::FromPrimitive as _;
use num_derive::FromPrimitive;

use crate::ops::Opcode;
use crate::progs::{LoadFn, ProgsError, Ptr, ScalarKind};
use crate::{ARG_ADDRS, quake1};

pub const MAX_ARGS: usize = 8;

#[derive(Debug, Copy, Clone, Reflect)]
pub struct Statement {
    pub opcode: Opcode,
    pub arg1: i16,
    pub arg2: i16,
    pub arg3: i16,
}

impl Statement {
    pub fn new(op: i16, arg1: i16, arg2: i16, arg3: i16) -> anyhow::Result<Statement> {
        let opcode = Opcode::from_i16(op)
            .ok_or_else(|| anyhow::Error::msg(format!("Bad opcode 0x{op:x}")))?;

        Ok(Statement {
            opcode,
            arg1,
            arg2,
            arg3,
        })
    }
}

#[derive(Copy, Clone, Debug, FromPrimitive, PartialEq, Reflect)]
#[repr(u8)]
pub enum ArgSize {
    Scalar = 1,
    Vector = 3,
}

impl std::fmt::Display for ArgSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "1 (scalar)"),
            Self::Vector => write!(f, "3 (vector)"),
        }
    }
}

#[derive(Debug)]
pub struct FunctionExecutionCtx<'a> {
    arguments: FixedBumpVec<'a, ScalarKind>,
    local_storage: FixedBumpVec<'a, ScalarKind>,
    /// For now, all functions are stateless. This allows for easier debugging, as we can see what the function
    /// accessed while running. It also ensures that a function that ran with errors can have modifications to
    /// the global state rolled back. QuakeC has few enough globals that this is cheap (although this will
    /// likely need to be limited to only the globals defined in `progdefs.h` for maximum efficiency)
    ///
    /// > TODO: Should QuakeC be able to modify state at all?
    delta: FixedBumpVec<'a, Option<ScalarKind>>,
    /// If the progs try to access a value within this range, it will access `local_storage` instead of globals.
    ///
    /// > TODO: We should only define globals at all if they are specified in the `progdefs.h` equivalent.
    local_range: Range<usize>,
    statements: ArcSlice<[Statement]>,
}

impl FunctionExecutionCtx<'_> {
    pub fn instr(&self, index: usize) -> anyhow::Result<Statement> {
        Ok(*self
            .statements
            .get(index)
            .ok_or_else(|| anyhow::Error::msg("Out-of-bounds instruction access"))?)
    }

    pub fn get(&self, index: usize) -> anyhow::Result<Option<ScalarKind>> {
        const LOCAL_STORAGE_ERR: &str =
            "Programmer error: `local_storage` was too small for `local_range`. This is a bug!";

        if self.local_range.contains(&index) {
            let index = index - self.local_range.start;

            Ok(Some(
                self.local_storage
                    .get(index)
                    .expect(LOCAL_STORAGE_ERR)
                    .clone(),
            ))
        } else {
            self.delta
                .get(index)
                .ok_or_else(|| anyhow::Error::msg(format!("Global {index} is out of range")))
                .cloned()
        }
    }

    pub fn get_vector(&self, index: usize) -> anyhow::Result<[Option<ScalarKind>; 3]> {
        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }

    pub fn set(&mut self, index: usize, value: ScalarKind) -> anyhow::Result<()> {
        const LOCAL_STORAGE_ERR: &str =
            "Programmer error: `local_storage` was too small for `local_range`. This is a bug!";
        const ARG_STORAGE_ERR: &str =
            "Programmer error: `arguments` was too small for `ARG_ADDRS`. This is a bug!";

        if ARG_ADDRS.contains(&index) {
            let index = index - ARG_ADDRS.start;

            let local = self.arguments.get_mut(index).expect(ARG_STORAGE_ERR);

            *local = value;
        } else if self.local_range.contains(&index) {
            let index = index - self.local_range.start;

            let local = self.local_storage.get_mut(index).expect(LOCAL_STORAGE_ERR);

            *local = value;
        } else {
            let global = self
                .delta
                .get_mut(index)
                .ok_or_else(|| anyhow::Error::msg(format!("Global {index} is out of range")))?;

            *global = Some(value);
        }

        Ok(())
    }

    pub fn set_vector(&mut self, index: usize, values: [ScalarKind; 3]) -> anyhow::Result<()> {
        for (offset, value) in values.into_iter().enumerate() {
            self.set(index + offset, value)?;
        }

        Ok(())
    }
}

#[reflect_remote(Vec<T>)]
#[reflect(opaque)]
#[derive(Clone)]
struct ReflectVec<T: Clone> {
    inner: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct QuakeCFunctionBody {
    /// `arg_start` + `locals` fields - we do not conflate locals and globals,
    /// so every access needs to check the locals first.
    pub locals: Range<usize>,
    pub statements: ArcSlice<[Statement]>,
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    Progs(QuakeCFunctionBody),
    Builtin,
}

impl FunctionBody {
    pub fn as_quakec(self) -> Option<QuakeCFunctionBody> {
        match self {
            Self::Progs(quakec) => Some(quakec),
            Self::Builtin => None,
        }
    }
}

pub type QuakeCFunctionDef = FunctionDef<QuakeCFunctionBody>;

/// Definition for a QuakeC function.
#[derive(Debug, Clone)]
pub struct FunctionDef<T = FunctionBody> {
    pub idx: i32,
    pub name: Arc<CStr>,
    pub source: Arc<CStr>,
    /// First N args get copied to the local stack.
    pub args: ArrayVec<ArgSize, MAX_ARGS>,
    pub body: T,
}

impl FunctionDef {
    pub fn as_quakec(self) -> Option<QuakeCFunctionDef> {
        Some(QuakeCFunctionDef {
            idx: self.idx,
            name: self.name,
            source: self.source,
            args: self.args,
            body: self.body.as_quakec()?,
        })
    }
}

impl QuakeCFunctionDef {
    pub fn ctx<'scope>(
        &self,
        mut alloc: impl BumpAllocatorScopeExt<'scope>,
    ) -> FunctionExecutionCtx<'scope> {
        FunctionExecutionCtx {
            arguments: FixedBumpVec::from_iter_exact_in(
                std::iter::repeat_n(ScalarKind::Void, ARG_ADDRS.len()),
                &mut alloc,
            ),
            local_storage: FixedBumpVec::from_iter_exact_in(
                std::iter::repeat_n(ScalarKind::Void, self.body.locals.len()),
                &mut alloc,
            ),
            delta: FixedBumpVec::from_iter_exact_in(
                std::iter::repeat_n(None, quake1::GLOBALS_RANGE.len()),
                &mut alloc,
            ),
            local_range: self.body.locals.clone(),
            statements: self.body.statements.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Reflect)]
pub enum ExternFn {
    Ref(DynamicScriptFunction),
    // TODO: What's the difference between `DynamicScriptFunctionMut` and `DynamicScriptFunctionMut`?
    // They both require `&` to call.
    Mut(DynamicScriptFunctionMut),
}

#[derive(Debug, Clone)]
pub struct FunctionRegistry {
    by_index: HashMap<i32, FunctionDef>,
    by_name: HashMap<Arc<CStr>, FunctionDef>,
}

impl FunctionRegistry {
    /// `iter` should be sorted by `offset`
    pub fn new<I>(statements: ArcSlice<[Statement]>, iter: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = LoadFn>,
    {
        let (by_index, by_name) = iter
            .into_iter()
            .map(Some)
            .chain(std::iter::once(None))
            .map_windows::<_, _, 2>(|[cur, next]| -> anyhow::Result<_> {
                match (cur, next) {
                    (Some(cur), next) => {
                        let func_def = FunctionDef {
                            name: cur.name.clone(),
                            source: cur.source.clone(),
                            args: cur.args.clone(),
                            idx: cur.offset,
                            body: if cur.offset < 0 {
                                debug_assert_eq!(cur.locals.len(), 0);

                                FunctionBody::Builtin
                            } else {
                                let cur_offset = usize::try_from(cur.offset)?;
                                let next_offset = next
                                    .as_ref()
                                    .map(|n| usize::try_from(n.offset))
                                    .unwrap_or(Ok(statements.len()))?;

                                FunctionBody::Progs(QuakeCFunctionBody {
                                    locals: cur.locals.clone(),
                                    statements: statements.subslice(cur_offset..next_offset),
                                })
                            },
                        };

                        Ok(((cur.offset, func_def.clone()), (cur.name.clone(), func_def)))
                    }
                    (None, _) => unreachable!(),
                }
            })
            .collect::<anyhow::Result<(_, _)>>()?;

        Ok(Self { by_index, by_name })
    }

    pub fn get<F: Into<Ptr>>(&self, func: F) -> anyhow::Result<&FunctionDef> {
        match func.into() {
            Ptr::Id(id) => self.by_index.get(&id).ok_or_else(|| {
                ProgsError::with_msg(format!("Function at index {id} does not exist")).into()
            }),
            Ptr::Name(name) => self.by_name.get(&name).ok_or_else(|| {
                ProgsError::with_msg(format!("Function with name {:?} does not exist", name)).into()
            }),
        }
    }
}
