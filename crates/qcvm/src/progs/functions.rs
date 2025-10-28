use std::ffi::CStr;
use std::{fmt, ops::Range, sync::Arc};

use arc_slice::ArcSlice;
use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use bump_scope::{BumpAllocatorScopeExt, FixedBumpVec};
use hashbrown::HashMap;
use num::FromPrimitive as _;
use num_derive::FromPrimitive;

use crate::load::LoadFn;
use crate::ops::Opcode;
use crate::progs::{ScalarType, VmScalar};
use crate::{ARG_ADDRS, QuakeCMemory};

pub const MAX_ARGS: usize = 8;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
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

#[derive(Copy, Clone, Debug, FromPrimitive, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
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
    arguments: FixedBumpVec<'a, VmScalar>,
    local_storage: FixedBumpVec<'a, VmScalar>,
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
}

const LOCAL_STORAGE_ERR: &str =
    "Programmer error: `local_storage` was too small for `local_range`. This is a bug!";
const ARG_STORAGE_ERR: &str =
    "Programmer error: `arguments` was too small for `ARG_ADDRS`. This is a bug!";

impl QuakeCMemory for FunctionExecutionCtx<'_> {
    type Scalar = Option<VmScalar>;

    fn get(&self, index: usize) -> anyhow::Result<Option<VmScalar>> {
        if ARG_ADDRS.contains(&index) {
            let index = index - ARG_ADDRS.start;

            Ok(Some(
                self.arguments.get(index).expect(ARG_STORAGE_ERR).clone(),
            ))
        } else if self.local_range.contains(&index) {
            let index = index - self.local_range.start;

            Ok(Some(
                self.local_storage
                    .get(index)
                    .expect(LOCAL_STORAGE_ERR)
                    .clone(),
            ))
        } else {
            Ok(None)
        }
    }
}

impl FunctionExecutionCtx<'_> {
    pub fn set(&mut self, index: usize, value: VmScalar) -> anyhow::Result<()> {
        if value.type_() == ScalarType::Void {
            return Ok(());
        }

        if ARG_ADDRS.contains(&index) {
            let index = index - ARG_ADDRS.start;

            let local = self.arguments.get_mut(index).expect(ARG_STORAGE_ERR);

            *local = value;
        } else if self.local_range.contains(&index) {
            let index = index - self.local_range.start;

            let local = self.local_storage.get_mut(index).expect(LOCAL_STORAGE_ERR);

            *local = value;
        } else {
            anyhow::bail!("Global {index} is out of range");
        }

        Ok(())
    }

    pub fn set_vector(&mut self, index: usize, values: [VmScalar; 3]) -> anyhow::Result<()> {
        for (offset, value) in values.into_iter().enumerate() {
            self.set(index + offset, value)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn try_into_quakec(self) -> Result<QuakeCFunctionBody, Builtin> {
        match self {
            Self::Progs(quakec) => Ok(quakec),
            Self::Builtin => Err(Builtin),
        }
    }
}

pub type QuakeCFunctionDef = FunctionDef<QuakeCFunctionBody>;

/// Definition for a QuakeC function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDef<T = FunctionBody> {
    pub offset: i32,
    pub name: Arc<CStr>,
    pub source: Arc<CStr>,
    /// First N args get copied to the local stack.
    pub args: ArrayVec<ArgSize, MAX_ARGS>,
    pub body: T,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Builtin;

pub type BuiltinDef = FunctionDef<Builtin>;

impl FunctionDef {
    pub fn try_into_quakec(self) -> Result<QuakeCFunctionDef, FunctionDef<Builtin>> {
        match self.body.try_into_quakec() {
            Ok(quakec) => Ok(QuakeCFunctionDef {
                offset: self.offset,
                name: self.name,
                source: self.source,
                args: self.args,
                body: quakec,
            }),
            Err(builtin) => Err(FunctionDef {
                offset: self.offset,
                name: self.name,
                source: self.source,
                args: self.args,
                body: builtin,
            }),
        }
    }
}

impl QuakeCFunctionDef {
    pub fn ctx<'scope>(
        &self,
        mut alloc: impl BumpAllocatorScopeExt<'scope>,
    ) -> FunctionExecutionCtx<'scope> {
        FunctionExecutionCtx {
            arguments: FixedBumpVec::from_iter_exact_in(
                std::iter::repeat_n(VmScalar::Void, ARG_ADDRS.len()),
                &mut alloc,
            ),
            local_storage: FixedBumpVec::from_iter_exact_in(
                std::iter::repeat_n(VmScalar::Void, self.body.locals.len()),
                &mut alloc,
            ),
            local_range: self.body.locals.clone(),
            statements: self.body.statements.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionRegistry {
    by_index: HashMap<i32, FunctionDef>,
    by_name: HashMap<Arc<CStr>, FunctionDef>,
}

impl FunctionRegistry {
    pub(crate) fn new<I>(statements: ArcSlice<[Statement]>, iter: I) -> anyhow::Result<Self>
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
                            offset: cur.offset,
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

    pub fn get_by_index<F>(&self, func: F) -> anyhow::Result<&FunctionDef>
    where
        F: TryInto<i32>,
        F::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let func = func.try_into()?;

        self.by_index
            .get(&func)
            .ok_or_else(|| anyhow::format_err!("Function {func} does not exist"))
    }

    pub fn get_by_name<F>(&self, func: F) -> anyhow::Result<&FunctionDef>
    where
        F: AsRef<CStr>,
    {
        let func = func.as_ref();

        self.by_name
            .get(func)
            .ok_or_else(|| anyhow::format_err!("Function {func:?} does not exist"))
    }
}
