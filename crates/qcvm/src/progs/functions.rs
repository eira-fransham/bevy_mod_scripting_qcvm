use std::ffi::CStr;
use std::{fmt, ops::Range, sync::Arc};

use crate::{FunctionRef, HashMap, Type};
use arc_slice::ArcSlice;
use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use bump_scope::{BumpAllocatorScopeExt, FixedBumpVec};
use num::FromPrimitive as _;
use num_derive::FromPrimitive;

use crate::QCMemory;
use crate::load::LoadFn;
use crate::ops::Opcode;
use crate::progs::VmScalar;

/// The maximum number of arguments supported by the runtime.
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
pub(crate) enum ArgSize {
    Scalar = 1,
    Vector = 3,
}

impl From<ArgSize> for Type {
    fn from(value: ArgSize) -> Self {
        match value {
            ArgSize::Scalar => Type::AnyScalar,
            ArgSize::Vector => Type::Vector,
        }
    }
}

impl std::fmt::Display for ArgSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "1 (scalar)"),
            Self::Vector => write!(f, "3 (vector)"),
        }
    }
}

/// A QuakeC function's call frame.
///
/// Only the function's locals (including its arguments, see [`QCFunctionDef::frame_range`]) are
/// stored per frame. Everything else, including the parameter slots that arguments are passed in,
/// is global memory.
#[derive(Debug)]
pub(crate) struct FunctionExecutionCtx<'a> {
    local_storage: FixedBumpVec<'a, VmScalar>,
    /// If the progs try to access a value within this range, it will access `local_storage` instead of globals.
    ///
    /// > TODO: We should only define globals at all if they are specified in the `progdefs.h` equivalent.
    local_range: Range<usize>,
    statements: ArcSlice<[Statement]>,
    /// The index in the `progs.dat` of the function's first statement.
    first_statement: usize,
}

impl FunctionExecutionCtx<'_> {
    /// The index in the `progs.dat` of the function's first statement, so that statements can be
    /// identified across functions.
    pub fn first_statement(&self) -> usize {
        self.first_statement
    }

    pub fn instr(&self, index: usize) -> anyhow::Result<Statement> {
        Ok(*self
            .statements
            .get(index)
            .ok_or_else(|| anyhow::Error::msg("Out-of-bounds instruction access"))?)
    }
}

const LOCAL_STORAGE_ERR: &str =
    "Programmer error: `local_storage` was too small for `local_range`. This is a bug!";

impl QCMemory for FunctionExecutionCtx<'_> {
    type Scalar = Option<VmScalar>;

    fn get(&self, index: usize) -> anyhow::Result<Option<VmScalar>> {
        if self.local_range.contains(&index) {
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
    /// Stores `value` if `index` is one of this frame's locals. Otherwise, gives the value back so
    /// that the caller can store it in global memory.
    ///
    /// Storing `Void` is a real store (of zero), as in Quake: e.g. assigning an uninitialized
    /// temporary to a local must clear the local.
    pub fn try_set(&mut self, index: usize, value: VmScalar) -> Result<(), VmScalar> {
        if self.local_range.contains(&index) {
            let index = index - self.local_range.start;

            let local = self.local_storage.get_mut(index).expect(LOCAL_STORAGE_ERR);

            *local = value;

            Ok(())
        } else {
            Err(value)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QCFunctionBody {
    /// `arg_start` + `locals` fields - we do not conflate locals and globals,
    /// so every access needs to check the locals first.
    pub locals: Range<usize>,
    pub statements: ArcSlice<[Statement]>,
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    Progs(QCFunctionBody),
    Builtin,
}

impl FunctionBody {
    pub fn try_into_qc(self) -> Result<QCFunctionBody, Builtin> {
        match self {
            Self::Progs(quakec) => Ok(quakec),
            Self::Builtin => Err(Builtin),
        }
    }
}

pub type QCFunctionDef = FunctionDef<QCFunctionBody>;

/// Definition for a QuakeC function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDef<T = FunctionBody> {
    /// The index of this function in the `progs.dat` function table. This is the value that
    /// QuakeC uses to refer to the function (e.g. in `self.think`), and index 0 is the null function.
    pub index: i32,
    /// The offset to the start of this function's statements for QuakeC
    /// functions, the negative index of a builtin for functions provided
    /// by the host.
    pub offset: i32,
    /// The name of this function.
    pub name: Arc<CStr>,
    /// The source file that this function is defined in.
    pub source: Arc<CStr>,
    /// First N args get copied to the local stack.
    pub args: ArrayVec<Type, MAX_ARGS>,
    /// If the function is defined in QuakeC, the function body.
    pub body: T,
}

impl<T> FunctionDef<T> {
    /// Whether this function is provided by the host.
    pub fn is_builtin(&self) -> bool {
        self.offset < 0
    }

    /// For builtins, the builtin number (`#n` in QuakeC), which is what Quake dispatches on.
    ///
    /// This is not the same as [`Self::index`]: builtin numbers can have gaps, and several
    /// functions can alias the same builtin (e.g. LibreQuake's `bprint_e = #23`).
    pub fn builtin_number(&self) -> Option<u32> {
        if self.is_builtin() {
            self.offset
                .checked_neg()
                .and_then(|n| u32::try_from(n).ok())
        } else {
            None
        }
    }
}

/// No body as the function is provided by the host.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Builtin;

/// A function definition for a function provided by the host.
pub type BuiltinDef = FunctionDef<Builtin>;

impl FunctionDef {
    pub(crate) fn try_into_qc(self) -> Result<QCFunctionDef, FunctionDef<Builtin>> {
        match self.body.try_into_qc() {
            Ok(quakec) => Ok(QCFunctionDef {
                index: self.index,
                offset: self.offset,
                name: self.name,
                source: self.source,
                args: self.args,
                body: quakec,
            }),
            Err(builtin) => Err(FunctionDef {
                index: self.index,
                offset: self.offset,
                name: self.name,
                source: self.source,
                args: self.args,
                body: builtin,
            }),
        }
    }
}

impl QCFunctionDef {
    /// The addresses stored in this function's call frame.
    pub(crate) fn frame_range(&self) -> Range<usize> {
        let locals = &self.body.locals;
        let args_size = self
            .args
            .iter()
            .map(|arg| arg.arg_size() as usize)
            .sum::<usize>();

        // `qcc` only counts locals up to the last `local` declaration, so functions that take
        // arguments but don't declare any locals say that they have no locals. To handle this,
        // we reserve at least enough space for the number of arguments.
        locals.start..locals.end.max(locals.start + args_size)
    }

    /// Creates the call frame for this function.
    pub(crate) fn ctx<'scope>(
        &self,
        mut alloc: impl BumpAllocatorScopeExt<'scope>,
        initial_locals: &[VmScalar],
    ) -> FunctionExecutionCtx<'scope> {
        FunctionExecutionCtx {
            local_storage: FixedBumpVec::from_iter_exact_in(
                initial_locals.iter().cloned(),
                &mut alloc,
            ),
            local_range: self.frame_range(),
            statements: self.body.statements.clone(),
            first_statement: usize::try_from(self.offset).unwrap_or_default(),
        }
    }
}

/// The registry of functions known to the `progs.dat`.
#[derive(Debug, Clone)]
pub struct FunctionRegistry {
    /// Functions in `progs.dat` order, so that function pointers index into this.
    by_index: Box<[FunctionDef]>,
    by_name: HashMap<Arc<CStr>, i32>,
}

impl FunctionRegistry {
    /// `definitions` must be in `progs.dat` order, as function pointers are indices into it.
    pub(crate) fn new(
        statements: ArcSlice<[Statement]>,
        definitions: &[LoadFn],
    ) -> anyhow::Result<Self> {
        // A function's body runs until the next function's first statement.
        let mut entry_points = definitions
            .iter()
            .map(|def| def.offset)
            .filter(|offset| *offset >= 0)
            .collect::<Vec<_>>();
        entry_points.sort_unstable();
        entry_points.dedup();

        let by_index = definitions
            .iter()
            .enumerate()
            .map(|(index, def)| -> anyhow::Result<FunctionDef> {
                let body = if def.offset < 0 {
                    FunctionBody::Builtin
                } else {
                    let start = usize::try_from(def.offset)?;
                    let end = entry_points
                        .iter()
                        .find(|offset| **offset > def.offset)
                        .map(|offset| usize::try_from(*offset))
                        .transpose()?
                        .unwrap_or(statements.len());

                    if start > end || end > statements.len() {
                        anyhow::bail!(
                            "Function {:?} has an invalid body ({start}..{end})",
                            def.name
                        );
                    }

                    FunctionBody::Progs(QCFunctionBody {
                        locals: def.locals.clone(),
                        statements: statements.subslice(start..end),
                    })
                };

                Ok(FunctionDef {
                    index: index.try_into()?,
                    offset: def.offset,
                    name: def.name.clone(),
                    source: def.source.clone(),
                    args: def.args.iter().copied().map(Into::into).collect(),
                    body,
                })
            })
            .collect::<anyhow::Result<Box<[_]>>>()?;

        // If names are duplicated, the first definition wins.
        let mut by_name = HashMap::default();
        for def in by_index.iter() {
            by_name.entry(def.name.clone()).or_insert(def.index);
        }

        Ok(Self { by_index, by_name })
    }

    /// Get a function by either an index or a name.
    pub fn get<F>(&self, func: F) -> anyhow::Result<&FunctionDef>
    where
        F: Into<FunctionRef>,
    {
        match func.into() {
            FunctionRef::Index(i) => self.get_by_index(i),
            FunctionRef::Name(name) => self.get_by_name(name),
        }
    }

    /// Get a function by its index.
    pub fn get_by_index<F>(&self, func: F) -> anyhow::Result<&FunctionDef>
    where
        F: TryInto<i32>,
        F::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let func = func.try_into()?;

        usize::try_from(func)
            .ok()
            .and_then(|index| self.by_index.get(index))
            .ok_or_else(|| anyhow::format_err!("Function {func} does not exist"))
    }

    /// Get a function by its name.
    pub fn get_by_name<F>(&self, func: F) -> anyhow::Result<&FunctionDef>
    where
        F: AsRef<CStr>,
    {
        let func = func.as_ref();

        self.by_name
            .get(func)
            .and_then(|index| self.by_index.get(*index as usize))
            .ok_or_else(|| anyhow::format_err!("Function {func:?} does not exist"))
    }

    /// All functions, in `progs.dat` order.
    pub fn iter(&self) -> impl Iterator<Item = &FunctionDef> {
        self.by_index.iter()
    }
}
