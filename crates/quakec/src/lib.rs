// TODO: Ideally we would not use nightly
#![feature(iter_map_windows, array_try_map)]

use std::{
    any::Any,
    ffi::CStr,
    fmt,
    num::NonZeroIsize,
    ops::{ControlFlow, Range},
    sync::Arc,
};

use arrayvec::ArrayVec;
use bump_scope::{Bump, BumpScope};
use glam::Vec3;

use crate::{
    entity::EntityTypeDef,
    load::Progs,
    progs::{
        FunctionRef, ScalarType, StringRef, StringTable, VmScalar, VmValue,
        functions::{
            ArgSize, FunctionExecutionCtx, FunctionRegistry, MAX_ARGS, QuakeCFunctionDef, Statement,
        },
        globals::GlobalRegistry,
    },
    userdata::{Context, ErasedContext, ErasedFunction, FnCall, QuakeCType},
};

pub use crate::progs::{EntityRef, ScalarCastError};

mod entity;
pub mod load;
mod ops;
mod progs;
pub mod userdata;

#[cfg(feature = "quake1")]
pub mod quake1;

pub struct CallArgs<T>(T);

impl QuakeCMemory for CallArgs<ArrayVec<VmScalar, MAX_ARGS>> {
    type Scalar = Option<VmScalar>;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
        if !ARG_ADDRS.contains(&index) {
            return Ok(None);
        }

        let arg_offset = index
            .checked_sub(ARG_ADDRS.start)
            .expect("Programmer error - index out of range for args");

        Ok(Some(
            self.0.get(arg_offset).unwrap_or(&VmScalar::Void).clone(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Void,
    Entity(EntityRef),
    Function(Arc<dyn ErasedFunction>),
    Float(f32),
    Vector(Vec3),
    String(Arc<CStr>),
}

impl Value {
    pub fn type_(&self) -> ArgType {
        match self {
            Value::Void => ArgType::Void,
            Value::Entity(_) => ArgType::Entity,
            Value::Function(_) => ArgType::Function,
            Value::Float(_) => ArgType::Float,
            Value::Vector(_) => ArgType::Vector,
            Value::String(_) => ArgType::String,
        }
    }
}

impl TryFrom<Value> for EntityRef {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Entity(entity_ref) => Ok(entity_ref),
            _ => anyhow::bail!("Expected {}, found {}", ArgType::Entity, value.type_()),
        }
    }
}

impl TryFrom<Value> for Arc<dyn ErasedFunction> {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Function(func) => Ok(func),
            _ => anyhow::bail!("Expected {}, found {}", ArgType::Function, value.type_()),
        }
    }
}

impl TryFrom<Value> for f32 {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(f) => Ok(f),
            _ => anyhow::bail!("Expected {}, found {}", ArgType::Float, value.type_()),
        }
    }
}

impl TryFrom<Value> for Vec3 {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Vector(vec) => Ok(vec),
            _ => anyhow::bail!("Expected {}, found {}", ArgType::Vector, value.type_()),
        }
    }
}

impl TryFrom<Value> for Arc<CStr> {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(str) => Ok(str),
            _ => anyhow::bail!("Expected {}, found {}", ArgType::String, value.type_()),
        }
    }
}

impl From<Value> for VmValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Void => VmValue::Scalar(VmScalar::Void),
            Value::Entity(entity_ref) => entity_ref.into(),
            Value::Function(erased_function) => {
                VmValue::Scalar(VmScalar::Function(FunctionRef::Extern(erased_function)))
            }
            Value::Float(f) => f.into(),
            Value::Vector(vec3) => vec3.into(),
            Value::String(cstr) => VmValue::Scalar(VmScalar::String(StringRef::Temp(cstr))),
        }
    }
}

macro_rules! impl_memory_tuple {
    ($first:ident $(, $rest:ident)*) => {
        impl<$first, $($rest),*> QuakeCMemory for CallArgs<($first, $($rest),*)>
        where
        $first: Clone + TryInto<progs::VmValue>,
        $first::Error: std::error::Error + Send + Sync + 'static,
        $(
            $rest: Clone + TryInto<progs::VmValue>,
            $rest::Error: std::error::Error + Send + Sync + 'static,
        )*
        {
            type Scalar = Option<VmScalar>;

            #[expect(non_snake_case)]
            fn get(&self, index: usize) -> anyhow::Result<Self::Scalar> {
                if !ARG_ADDRS.contains(&index) {
                    return Ok(None);
                }

                let CallArgs(($first, $($rest),*)) = self;

                let arg_offset = index.checked_sub(ARG_ADDRS.start).expect("Programmer error - index out of range for args");
                let field_offset = arg_offset % 3;

                let value = impl_memory_tuple!(@arg_match arg_offset $first $($rest)*);

                match (value, field_offset) {
                    (progs::VmValue::Scalar(scalar), 0) => Ok(Some(scalar)),
                    (progs::VmValue::Vector([x, _, _]), 0) => Ok(Some(x.into())),
                    (progs::VmValue::Vector([_, y, _]), 1) => Ok(Some(y.into())),
                    (progs::VmValue::Vector([_, _, z]), 2) => Ok(Some(z.into())),
                    (value, field_offset) => anyhow::bail!("Invalid field access {field_offset} for {value:?}"),
                }
            }

            fn get_vector(&self, index: usize) -> anyhow::Result<[Self::Scalar; 3]> {
                Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
            }
        }

        impl_memory_tuple!($($rest),*);
    };
    (@arg_match $match_name:ident $a0:ident $($a1:ident $($a2:ident $($a3:ident $($a4:ident $($a5:ident $($a6:ident $($a7:ident)?)?)?)?)?)?)?) => {
        match $match_name {
            0 => $a0.clone().try_into()?,
            $(3 => $a1.clone().try_into()?,
            $(6 => $a2.clone().try_into()?,
            $(9 => $a3.clone().try_into()?,
            $(12 => $a4.clone().try_into()?,
            $(15 => $a5.clone().try_into()?,
            $(18 => $a6.clone().try_into()?,
            $(21 => $a7.clone().try_into()?,)?)?)?)?)?)?)?
            _ => anyhow::bail!("{} is out of range for arguments (max: {})", $match_name, progs::functions::MAX_ARGS),
        }
    };
    () => {

    }
}

impl_memory_tuple!(A, B, C, D, E, F, G, H);

impl QuakeCType for QuakeCFunctionDef {
    fn type_(&self) -> ScalarType {
        ScalarType::Function
    }

    fn is_null(&self) -> bool {
        self.offset == 0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ArgType {
    Void,
    Any,
    Entity,
    Function,
    Float,
    Vector,
    String,
}

impl fmt::Display for ArgType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgType::Void => write!(f, "void"),
            ArgType::Any => write!(f, "<any>"),
            ArgType::Entity => write!(f, "entity"),
            ArgType::Function => write!(f, "function"),
            ArgType::Float => write!(f, "float"),
            ArgType::Vector => write!(f, "vector"),
            ArgType::String => write!(f, "string"),
        }
    }
}

impl ErasedFunction for QuakeCFunctionDef {
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<ArgType, MAX_ARGS>> {
        Ok(self
            .args
            .iter()
            .map(|size| match size {
                ArgSize::Scalar => ArgType::Any,
                ArgSize::Vector => ArgType::Vector,
            })
            .collect())
    }

    fn dyn_call(&self, mut context: FnCall) -> anyhow::Result<Value> {
        let vm_value = context.execution.execute(self)?;

        context.execution.to_value(vm_value.try_into()?)
    }
}

// TODO: Add ways to persist context, so it doesn't always need to be fetched from the environment
// between frames.
#[non_exhaustive]
pub struct QuakeCVm {
    pub progs: Progs,
}

impl QuakeCVm {
    pub fn execution_ctx<'a, T, C>(
        &'a self,
        context: &'a mut C,
        args: T,
    ) -> ExecutionCtx<'a, dyn ErasedContext, Bump, CallArgs<T>>
    where
        CallArgs<T>: QuakeCMemory,
        C: ErasedContext,
    {
        ExecutionCtx {
            alloc: Bump::new(),
            memory: ExecutionMemory {
                local: CallArgs(args),
                global: &self.progs.globals,
                last_ret: None,
            },
            backtrace: Default::default(),
            context,
            entity_def: &self.progs.entity_def,
            string_table: &self.progs.string_table,
            functions: &self.progs.functions,
        }
    }
}

#[derive(Default, Debug)]
struct BacktraceFrame<'a>(Option<(&'a CStr, &'a BacktraceFrame<'a>)>);

impl fmt::Display for BacktraceFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((function, prev)) = self.0 {
            write!(f, "{}", function.to_string_lossy())?;

            prev.fmt(f)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
struct ExecutionMemory<'a, Caller = FunctionExecutionCtx<'a>> {
    local: Caller,
    global: &'a GlobalRegistry,
    /// Technically, every QuakeC function returns 3 scalars of arbitrary types.
    /// Only one return value is available at once, if another function is called then the previous return
    /// value is lost unless it was saved to a local. We store it here in order to allow us to access it in `get` etc.
    last_ret: Option<[VmScalar; 3]>,
}

impl ExecutionMemory<'_> {
    fn instr(&self, pc: usize) -> anyhow::Result<Statement> {
        self.local.instr(pc)
    }
}

pub trait QuakeCMemory {
    type Scalar;

    fn get(&self, index: usize) -> anyhow::Result<Self::Scalar>;

    fn get_vector(&self, index: usize) -> anyhow::Result<[Self::Scalar; 3]> {
        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }
}

impl<M> QuakeCMemory for ExecutionMemory<'_, M>
where
    M: QuakeCMemory<Scalar = Option<VmScalar>>,
{
    type Scalar = VmScalar;

    fn get(&self, index: usize) -> anyhow::Result<VmScalar> {
        if RETURN_ADDRS.contains(&index) {
            return self
                .last_ret
                .as_ref()
                .and_then(|val| val.get(index.checked_sub(RETURN_ADDRS.start)?).cloned())
                .ok_or_else(|| {
                    anyhow::format_err!("Tried to read return values before calling a function")
                });
        }

        match self.local.get(index)? {
            Some(val) => Ok(val),
            None => self.global.get_value(index),
        }
    }
}

impl ExecutionMemory<'_> {
    fn set(&mut self, index: usize, value: VmScalar) -> anyhow::Result<()> {
        self.local.set(index, value)
    }

    fn set_vector(&mut self, index: usize, values: [VmScalar; 3]) -> anyhow::Result<()> {
        self.local.set_vector(index, values)
    }
}

#[derive(Debug)]
pub struct ExecutionCtx<
    'a,
    Ctx = dyn ErasedContext,
    Alloc = BumpScope<'a>,
    Caller = FunctionExecutionCtx<'a>,
> where
    Ctx: ?Sized,
{
    alloc: Alloc,
    memory: ExecutionMemory<'a, Caller>,
    backtrace: BacktraceFrame<'a>,
    context: &'a mut Ctx,
    entity_def: &'a EntityTypeDef,

    string_table: &'a StringTable,

    /// Function definitions and bodies.
    functions: &'a FunctionRegistry,
}

enum OpResult {
    Jump(NonZeroIsize),
    Ret([VmScalar; 3]),
    Continue,
}

impl From<OpResult> for ControlFlow<[VmScalar; 3], isize> {
    fn from(value: OpResult) -> Self {
        match value {
            OpResult::Jump(ofs) => ControlFlow::Continue(ofs.get()),
            OpResult::Continue => ControlFlow::Continue(1),
            OpResult::Ret(ret) => ControlFlow::Break(ret),
        }
    }
}

impl From<()> for OpResult {
    fn from(_: ()) -> Self {
        Self::Continue
    }
}

impl From<[VmScalar; 3]> for OpResult {
    fn from(value: [VmScalar; 3]) -> Self {
        Self::Ret(value)
    }
}

const ARG_ADDRS: Range<usize> = 4..28;
const RETURN_ADDRS: Range<usize> = 1..4;

/// Frustratingly, the `bump-scope` crate doesn't have a way to be generic over `Bump` or `BumpScope`.
pub trait ScopedAlloc {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O;
}

impl ScopedAlloc for Bump {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        self.scoped(func)
    }
}

impl ScopedAlloc for BumpScope<'_> {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        self.scoped(func)
    }
}

impl ScopedAlloc for &'_ mut BumpScope<'_> {
    fn scoped<F, O>(&mut self, func: F) -> O
    where
        F: FnOnce(BumpScope<'_>) -> O,
    {
        (**self).scoped(func)
    }
}

impl<Ctx, Alloc> ExecutionCtx<'_, Ctx, Alloc>
where
    Ctx: ?Sized,
    Alloc: ScopedAlloc,
{
    pub fn instr(&self, pc: usize) -> anyhow::Result<Statement> {
        self.memory.instr(pc)
    }
}

impl ExecutionCtx<'_> {
    fn with_builtin_call_context<F, O>(
        &mut self,
        name: &CStr,
        args: ArrayVec<VmScalar, MAX_ARGS>,
        func: F,
    ) -> O
    where
        F: FnOnce(
            ExecutionCtx<
                '_,
                dyn ErasedContext,
                BumpScope<'_>,
                CallArgs<ArrayVec<VmScalar, MAX_ARGS>>,
            >,
        ) -> O,
    {
        let ExecutionCtx {
            alloc,
            memory: ExecutionMemory { global, .. },
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        alloc.scoped(|alloc| {
            func(ExecutionCtx {
                alloc,
                memory: ExecutionMemory {
                    local: CallArgs(args),
                    global,
                    last_ret: None,
                },
                backtrace: BacktraceFrame(Some((name, backtrace))),
                context: &mut **context,
                entity_def,
                string_table,
                functions,
            })
        })
    }
}

impl<'a, Alloc, Caller> ExecutionCtx<'a, dyn ErasedContext, Alloc, Caller> {
    fn downcast<T>(self) -> Option<ExecutionCtx<'a, T, Alloc, Caller>>
    where
        T: Any,
    {
        let ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        Some(ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context: (context as &mut dyn Any).downcast_mut()?,
            entity_def,
            string_table,
            functions,
        })
    }
}

impl<'a, Ctx, Alloc, Caller> ExecutionCtx<'a, Ctx, Alloc, Caller>
where
    Ctx: ErasedContext,
{
    fn into_erased(self) -> ExecutionCtx<'a, dyn ErasedContext, Alloc, Caller> {
        let ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        ExecutionCtx {
            alloc,
            memory,
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        }
    }
}

impl<'a, Ctx, Alloc, Caller> ExecutionCtx<'a, Ctx, Alloc, Caller>
where
    Ctx: ?Sized + ErasedContext,
{
    fn to_value(&self, vm_value: VmValue) -> anyhow::Result<Value> {
        match vm_value {
            VmValue::Vector(vec) => Ok(Value::Vector(vec.into())),
            VmValue::Scalar(VmScalar::Float(f)) => Ok(Value::Float(f)),
            VmValue::Scalar(VmScalar::Void) => Ok(Value::Void),
            VmValue::Scalar(VmScalar::Entity(ent)) => Ok(Value::Entity(ent)),
            VmValue::Scalar(VmScalar::String(str)) => {
                Ok(Value::String(self.string_table.get(str)?))
            }
            VmValue::Scalar(VmScalar::Function(FunctionRef::Ptr(ptr))) => {
                let arg_func = self.functions.get(ptr.0)?.clone();

                match arg_func.try_into_quakec() {
                    Ok(quakec) => Ok(Value::Function(Arc::new(quakec))),
                    Err(builtin) => Ok(Value::Function(self.context.dyn_builtin(&builtin)?)),
                }
            }
            VmValue::Scalar(VmScalar::Function(FunctionRef::Extern(func))) => {
                Ok(Value::Function(func))
            }
            VmValue::Scalar(VmScalar::Field(_) | VmScalar::Global(_)) => anyhow::bail!(
                "Values of type {} are unsupported as arguments to builtins",
                vm_value.type_()
            ),
        }
    }
}

impl<Alloc, Caller> ExecutionCtx<'_, dyn ErasedContext, Alloc, Caller>
where
    Alloc: ScopedAlloc,
    Caller: QuakeCMemory,
{
    // TODO: We can use the unsafe checkpoint API if just recursing becomes too slow.
    pub fn execute(&mut self, function: &QuakeCFunctionDef) -> anyhow::Result<[VmScalar; 3]> {
        let Self {
            alloc,
            memory,
            backtrace,
            context,
            entity_def,
            string_table,
            functions,
        } = self;

        alloc.scoped(move |alloc| {
            let mut memory = ExecutionMemory {
                local: function.ctx(&alloc),
                global: memory.global,
                last_ret: None,
            };

            for (src, dst) in ARG_ADDRS.zip(function.body.locals.clone()) {
                memory.set(dst, memory.get(src)?)?;
            }

            let mut out = ExecutionCtx {
                memory,
                alloc,
                backtrace: BacktraceFrame(Some((&function.name, &*backtrace))),
                context: &mut **context,
                entity_def,
                string_table,
                functions,
            };

            out.execute_internal()
        })
    }

    pub fn backtrace(&self) -> impl Iterator<Item = &'_ CStr> + '_ {
        std::iter::successors(self.backtrace.0.as_ref(), |(_, prev)| prev.0.as_ref())
            .map(|(name, _)| *name)
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
}

impl ExecutionCtx<'_> {
    pub fn get<I, O>(&self, index: I) -> anyhow::Result<O>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        VmScalar: TryInto<O>,
        <VmScalar as TryInto<O>>::Error: std::error::Error + Send + Sync + 'static,
    {
        Ok(self.memory.get(index.try_into()?)?.try_into()?)
    }

    pub fn set<I, V>(&mut self, index: I, value: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<VmScalar>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        self.memory.set(index.try_into()?, value.try_into()?)
    }

    /// Sets the "last return" global. QuakeC only allows 1 function return to be accessible
    /// at a given time.
    ///
    /// This can't be done with the regular `set` as this shouldn't be accessible by regular
    /// QuakeC code, only from the engine.
    pub fn set_return(&mut self, values: [VmScalar; 3]) {
        self.memory.last_ret = Some(values);
    }

    pub fn get_vector<I, O>(&self, index: I) -> anyhow::Result<[O; 3]>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        VmScalar: TryInto<O>,
        <VmScalar as TryInto<O>>::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok([self.get(index)?, self.get(index + 1)?, self.get(index + 2)?])
    }

    pub fn get_vec3<I>(&self, index: I) -> anyhow::Result<Vec3>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok(self.get_vector::<_, f32>(index)?.into())
    }

    pub fn set_vector<I, V>(&mut self, index: I, values: [V; 3]) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<VmScalar>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        let values = values.try_map(|val| val.try_into())?;
        self.memory.set_vector(index, values)
    }

    pub fn set_vec3<I, V>(&mut self, index: I, values: V) -> anyhow::Result<()>
    where
        I: TryInto<usize>,
        I::Error: std::error::Error + Send + Sync + 'static,
        V: TryInto<[f32; 3]>,
        V::Error: std::error::Error + Send + Sync + 'static,
    {
        let index = index.try_into()?;
        let values = values.try_into()?;
        self.memory.set_vector(index, values.map(Into::into))
    }

    fn execute_internal(&mut self) -> anyhow::Result<[VmScalar; 3]> {
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

            match self.execute_statement(self.instr(counter)?)?.into() {
                ControlFlow::Continue(idx) => {
                    counter = counter
                        .checked_add_signed(idx)
                        .ok_or_else(|| anyhow::format_err!("Out-of-bounds instruction access"))?;
                }
                ControlFlow::Break(vals) => return Ok(vals),
            }
        }
    }
}
