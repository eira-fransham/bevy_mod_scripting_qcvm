use anyhow::Context as _;
use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use glam::Vec3;
use num_derive::FromPrimitive;
use std::{ffi::CStr, fmt, num::NonZeroIsize};
use tracing::debug;

use crate::{
    ArgAddr, CallSite, ExecutionCtx, OpResult, QCMemory, RawArgs, ScopedAlloc, Type, function_args,
    progs::{
        EntityField, EntityRef, FieldPtr, StringRef, VmFunctionRef, VmScalar, VmValue,
        functions::{MAX_ARGS, Statement},
    },
    userdata::{AddrError, ErasedContext, ErasedFunction, FnCall},
};

/// The return value and parameters, i.e. the addresses before the first global defined by the
/// `progs.dat`.
const RESERVED_GLOBALS: i16 = 28;

/// Converts an error accessing an entity field to an error, as there's no fallback for fields.
fn entity_addr_error(error: AddrError<anyhow::Error>, field: i32) -> anyhow::Error {
    match error {
        AddrError::OutOfRange => anyhow::format_err!("No entity field at offset {field}"),
        AddrError::Other { error } => error,
    }
}

#[derive(Copy, Clone, Debug, FromPrimitive, PartialEq, Eq)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[repr(i16)]
pub enum Opcode {
    Done = 0,
    MulF = 1,
    MulV = 2,
    MulFV = 3,
    MulVF = 4,
    Div = 5,
    AddF = 6,
    AddV = 7,
    SubF = 8,
    SubV = 9,
    EqF = 10,
    EqV = 11,
    EqS = 12,
    EqEnt = 13,
    EqFnc = 14,
    NeF = 15,
    NeV = 16,
    NeS = 17,
    NeEnt = 18,
    NeFnc = 19,
    Le = 20,
    Ge = 21,
    Lt = 22,
    Gt = 23,
    LoadF = 24,
    LoadV = 25,
    LoadS = 26,
    LoadEnt = 27,
    LoadFld = 28,
    LoadFnc = 29,
    Address = 30,
    StoreF = 31,
    StoreV = 32,
    StoreS = 33,
    StoreEnt = 34,
    StoreFld = 35,
    StoreFnc = 36,
    StorePF = 37,
    StorePV = 38,
    StorePS = 39,
    StorePEnt = 40,
    StorePFld = 41,
    StorePFnc = 42,
    Return = 43,
    NotF = 44,
    NotV = 45,
    NotS = 46,
    NotEnt = 47,
    NotFnc = 48,
    If = 49,
    IfNot = 50,
    Call0 = 51,
    Call1 = 52,
    Call2 = 53,
    Call3 = 54,
    Call4 = 55,
    Call5 = 56,
    Call6 = 57,
    Call7 = 58,
    Call8 = 59,
    State = 60,
    Goto = 61,
    And = 62,
    Or = 63,
    BitAnd = 64,
    BitOr = 65,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done => write!(f, "done"),
            Self::MulF => write!(f, "mul_f"),
            Self::MulV => write!(f, "mul_v"),
            Self::MulFV => write!(f, "mul_f_v"),
            Self::MulVF => write!(f, "mul_v_f"),
            Self::Div => write!(f, "div"),
            Self::AddF => write!(f, "add_f"),
            Self::AddV => write!(f, "add_v"),
            Self::SubF => write!(f, "sub_f"),
            Self::SubV => write!(f, "sub_v"),
            Self::EqF => write!(f, "eq_f"),
            Self::EqV => write!(f, "eq_v"),
            Self::EqS => write!(f, "eq_s"),
            Self::EqEnt => write!(f, "eq_ent"),
            Self::EqFnc => write!(f, "eq_fnc"),
            Self::NeF => write!(f, "ne_f"),
            Self::NeV => write!(f, "ne_v"),
            Self::NeS => write!(f, "ne_s"),
            Self::NeEnt => write!(f, "ne_ent"),
            Self::NeFnc => write!(f, "ne_fnc"),
            Self::Le => write!(f, "le"),
            Self::Ge => write!(f, "ge"),
            Self::Lt => write!(f, "lt"),
            Self::Gt => write!(f, "gt"),
            Self::LoadF => write!(f, "load_f"),
            Self::LoadV => write!(f, "load_v"),
            Self::LoadS => write!(f, "load_s"),
            Self::LoadEnt => write!(f, "load_ent"),
            Self::LoadFld => write!(f, "load_fld"),
            Self::LoadFnc => write!(f, "load_fnc"),
            Self::Address => write!(f, "address"),
            Self::StoreF => write!(f, "store_f"),
            Self::StoreV => write!(f, "store_v"),
            Self::StoreS => write!(f, "store_s"),
            Self::StoreEnt => write!(f, "store_ent"),
            Self::StoreFld => write!(f, "store_fld"),
            Self::StoreFnc => write!(f, "store_fnc"),
            Self::StorePF => write!(f, "store_p_f"),
            Self::StorePV => write!(f, "store_p_v"),
            Self::StorePS => write!(f, "store_p_s"),
            Self::StorePEnt => write!(f, "store_p_ent"),
            Self::StorePFld => write!(f, "store_p_fld"),
            Self::StorePFnc => write!(f, "store_p_fnc"),
            Self::Return => write!(f, "return"),
            Self::NotF => write!(f, "not_f"),
            Self::NotV => write!(f, "not_v"),
            Self::NotS => write!(f, "not_s"),
            Self::NotEnt => write!(f, "not_ent"),
            Self::NotFnc => write!(f, "not_fnc"),
            Self::If => write!(f, "if"),
            Self::IfNot => write!(f, "if_not"),
            Self::Call0 => write!(f, "call0"),
            Self::Call1 => write!(f, "call1"),
            Self::Call2 => write!(f, "call2"),
            Self::Call3 => write!(f, "call3"),
            Self::Call4 => write!(f, "call4"),
            Self::Call5 => write!(f, "call5"),
            Self::Call6 => write!(f, "call6"),
            Self::Call7 => write!(f, "call7"),
            Self::Call8 => write!(f, "call8"),
            Self::State => write!(f, "state"),
            Self::Goto => write!(f, "goto"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::BitAnd => write!(f, "bit_and"),
            Self::BitOr => write!(f, "bit_or"),
        }
    }
}

impl<Alloc, Caller> ExecutionCtx<'_, dyn ErasedContext, Alloc, Caller>
where
    Alloc: ScopedAlloc,
    Caller: fmt::Debug + QCMemory,
    <Caller as QCMemory>::Scalar: Into<Option<VmScalar>>,
{
    pub(crate) fn enter_builtin(
        &mut self,
        name: &CStr,
        builtin: &dyn ErasedFunction,
        num_args: usize,
    ) -> anyhow::Result<[VmScalar; 3]> {
        let sig = builtin.dyn_signature()?;
        if sig.len() != num_args {
            // This is expected for variadic builtins like `sprint`.
            debug!(
                "Builtin {} called with {num_args} args, but it takes {}",
                name.to_string_lossy(),
                sig.len(),
            );
        }

        let params = function_args()
            .into_iter()
            .take(num_args)
            .zip(sig)
            .map(|(ArgAddr { addr, .. }, ty)| match ty {
                Type::Vector => Ok(self.get_vec3(addr)?.into()),
                Type::AnyScalar => Ok(self.get_scalar::<_, VmValue>(addr)?),
                Type::Entity => Ok(self.get_scalar::<_, EntityRef>(addr)?.into()),
                Type::Function => Ok(self.get_scalar::<_, VmFunctionRef>(addr)?.into()),
                Type::Float => Ok(self.get_scalar::<_, f32>(addr)?.into()),
                Type::String => Ok(self.get_scalar::<_, StringRef>(addr)?.into()),
                Type::Void => Ok(Default::default()),
            })
            .map(|value| match value {
                Ok(value) => Ok(<[VmScalar; 3]>::from(value)),
                Err(e) => Err(e),
            })
            .collect::<anyhow::Result<ArrayVec<[VmScalar; 3], MAX_ARGS>>>()?;

        let vm_value: VmValue = self
            .with_memory(name, RawArgs(params), |ctx| {
                builtin.dyn_call(FnCall { execution: ctx })
            })
            .with_context(|| format!("in builtin {}", name.to_string_lossy()))?
            .into();

        Ok(vm_value.into())
    }
}

impl<Alloc> ExecutionCtx<'_, dyn ErasedContext, Alloc>
where
    Alloc: ScopedAlloc,
{
    /// Executes `statement`, which is at index `pc` in the current function.
    pub(crate) fn execute_statement(
        &mut self,
        statement: Statement,
        pc: usize,
    ) -> anyhow::Result<OpResult> {
        use Opcode as O;

        let op = statement.opcode;
        let a = statement.arg1;
        let b = statement.arg2;
        let c = statement.arg3;

        debug!("{:<12} {:>5} {:>5} {:>5}", op, a, b, c);

        Ok(match op {
            // Control flow ================================================
            O::If => self.op_if(a, b, c)?,
            O::IfNot => self.op_if_not(a, b, c)?,
            O::Goto => self.op_goto(a, b, c)?,
            O::Call0
            | O::Call1
            | O::Call2
            | O::Call3
            | O::Call4
            | O::Call5
            | O::Call6
            | O::Call7
            | O::Call8 => {
                let func_ref: VmFunctionRef = self.get_scalar(a)?;

                let num_args = op as usize - O::Call0 as usize;

                let result = match func_ref.normalize() {
                    VmFunctionRef::Ptr(p) => {
                        if p.is_null() {
                            anyhow::bail!("NULL function");
                        }

                        let Ok(def) = self.functions.get_by_index(p.0) else {
                            anyhow::bail!("NULL function ({})", p.0);
                        };

                        match def.clone().try_into_qc() {
                            Ok(quakec) => {
                                let site =
                                    CallSite::Statement(self.memory.local.first_statement() + pc);
                                self.check_num_args(&quakec, num_args, site)?;

                                self.execute_def(&quakec)?
                            }
                            Err(builtin) => self.enter_builtin(
                                &builtin.name,
                                &*self.context.dyn_builtin(&builtin)?,
                                num_args,
                            )?,
                        }
                    }
                    VmFunctionRef::Extern(func) => {
                        if func.is_null() {
                            anyhow::bail!("NULL function");
                        }

                        self.enter_builtin(c"{anonymous}", &*func, num_args)?
                    }
                };

                self.set_return(result);

                OpResult::Continue
            }

            O::Done | O::Return => self.op_return(a, b, c)?.into(),

            O::MulF => self.op_mul_f(a, b, c)?.into(),
            O::MulV => self.op_mul_v(a, b, c)?.into(),
            O::MulFV => self.op_mul_fv(a, b, c)?.into(),
            O::MulVF => self.op_mul_vf(a, b, c)?.into(),
            O::Div => self.op_div(a, b, c)?.into(),
            O::AddF => self.op_add_f(a, b, c)?.into(),
            O::AddV => self.op_add_v(a, b, c)?.into(),
            O::SubF => self.op_sub_f(a, b, c)?.into(),
            O::SubV => self.op_sub_v(a, b, c)?.into(),
            O::EqF => self.op_eq_f(a, b, c)?.into(),
            O::EqV => self.op_eq_v(a, b, c)?.into(),
            O::EqS => self.op_eq_s(a, b, c)?.into(),
            O::EqEnt => self.op_eq_ent(a, b, c)?.into(),
            O::EqFnc => self.op_eq_fnc(a, b, c)?.into(),
            O::NeF => self.op_ne_f(a, b, c)?.into(),
            O::NeV => self.op_ne_v(a, b, c)?.into(),
            O::NeS => self.op_ne_s(a, b, c)?.into(),
            O::NeEnt => self.op_ne_ent(a, b, c)?.into(),
            O::NeFnc => self.op_ne_fnc(a, b, c)?.into(),
            O::Le => self.op_le(a, b, c)?.into(),
            O::Ge => self.op_ge(a, b, c)?.into(),
            O::Lt => self.op_lt(a, b, c)?.into(),
            O::Gt => self.op_gt(a, b, c)?.into(),
            O::LoadV => self.op_load_v(a, b, c)?.into(),
            O::LoadF | O::LoadS | O::LoadEnt | O::LoadFld | O::LoadFnc => {
                self.load_scalar(a, b, c)?.into()
            }
            O::Address => self.op_address(a, b, c)?.into(),
            O::StoreV => self.op_store_v(a, b, c)?.into(),
            O::StoreF | O::StoreS | O::StoreEnt | O::StoreFld | O::StoreFnc => {
                self.copy(a, b, c)?.into()
            }
            O::StorePV => self.op_storep_v(a, b, c)?.into(),
            O::StorePF | O::StorePS | O::StorePEnt | O::StorePFld | O::StorePFnc => {
                self.store_scalar_to_field(a, b, c)?.into()
            }
            O::NotF => self.op_not_f(a, b, c)?.into(),
            O::NotV => self.op_not_v(a, b, c)?.into(),
            O::NotS => self.op_not_s(a, b, c)?.into(),
            O::NotEnt => self.op_not_ent(a, b, c)?.into(),
            O::NotFnc => self.op_not_fnc(a, b, c)?.into(),
            O::And => self.op_and(a, b, c)?.into(),
            O::Or => self.op_or(a, b, c)?.into(),
            O::BitAnd => self.op_bit_and(a, b, c)?.into(),
            O::BitOr => self.op_bit_or(a, b, c)?.into(),

            O::State => self.op_state(a, b, c)?.into(),
        })
    }

    fn op_if(&mut self, a: i16, b: i16, _c: i16) -> anyhow::Result<OpResult> {
        let cond = !self.get_scalar::<_, VmScalar>(a)?.is_null();
        // debug!("{op}: cond == {cond}");

        if cond {
            Ok(OpResult::Jump(NonZeroIsize::new(b as isize).ok_or_else(
                || anyhow::Error::msg("Tried to jump with an offset of 0"),
            )?))
        } else {
            Ok(OpResult::Continue)
        }
    }

    fn op_if_not(&mut self, a: i16, b: i16, _c: i16) -> anyhow::Result<OpResult> {
        let cond = !self.get_scalar::<_, VmScalar>(a)?.is_null();
        // debug!("{op}: cond == {cond}");

        if cond {
            Ok(OpResult::Continue)
        } else {
            Ok(OpResult::Jump(NonZeroIsize::new(b as isize).ok_or_else(
                || anyhow::Error::msg("Tried to jump with an offset of 0"),
            )?))
        }
    }

    fn op_goto(&mut self, a: i16, _b: i16, _c: i16) -> anyhow::Result<OpResult> {
        Ok(OpResult::Jump(NonZeroIsize::new(a as isize).ok_or_else(
            || anyhow::Error::msg("Tried to jump with an offset of 0"),
        )?))
    }

    fn op_return(&mut self, a: i16, _b: i16, _c: i16) -> anyhow::Result<[VmScalar; 3]> {
        Ok([
            self.get_scalar(a).unwrap_or_default(),
            self.get_scalar(a + 1).unwrap_or_default(),
            self.get_scalar(a + 2).unwrap_or_default(),
        ])
    }

    fn load_scalar(&mut self, entity: i16, field: i16, out_ptr: i16) -> anyhow::Result<()> {
        let ent: EntityRef = self.get_scalar(entity)?;
        let FieldPtr(ptr) = self.get_scalar(field)?;

        let value = self
            .context
            .dyn_entity_get(ent, ptr.0.try_into()?, Type::AnyScalar, self.entity_def)
            .map_err(|e| entity_addr_error(e, ptr.0))?;

        self.set(
            out_ptr,
            VmScalar::try_from(value).map_err(|_| {
                anyhow::format_err!("Field {} is a vector, but was loaded as a scalar", ptr.0)
            })?,
        )
    }

    // LOAD_V: load vector field from entity
    fn op_load_v(&mut self, entity: i16, field: i16, out_ptr: i16) -> anyhow::Result<()> {
        let ent: EntityRef = self.get_scalar(entity)?;
        let FieldPtr(ptr) = self.get_scalar(field)?;

        let value: Vec3 = self
            .context
            .dyn_entity_get(ent, ptr.0.try_into()?, Type::Vector, self.entity_def)
            .map_err(|e| entity_addr_error(e, ptr.0))?
            .try_into()?;

        self.set_vec3(out_ptr, value)
    }

    fn op_address(&mut self, entity: i16, field: i16, out_ptr: i16) -> anyhow::Result<()> {
        let entity: EntityRef = self.get_scalar(entity)?;
        let field_offset: FieldPtr = self.get_scalar(field)?;

        let field_ptr = VmScalar::EntityField(entity, field_offset.0);

        self.set(out_ptr, field_ptr)
    }

    fn store_scalar_to_field(
        &mut self,
        src_ptr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_f: nonzero arg3"));
        }

        let f: VmScalar = self.get_scalar(src_ptr)?;
        let value = self.to_value(f.into())?;
        let EntityField { entity, field } = self.get_scalar(out_ptr)?;

        self.context
            .dyn_entity_set(entity, field.0.try_into()?, value, self.entity_def)
            .map_err(|e| entity_addr_error(e, field.0))?;

        Ok(())
    }

    fn op_storep_v(&mut self, src_ptr: i16, out_ptr: i16, unused: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_f: nonzero arg3"));
        }

        let f = self.get_vec3(src_ptr)?;
        let EntityField { entity, field } = self.get_scalar(out_ptr)?;

        self.context
            .dyn_entity_set(entity, field.0.try_into()?, f.into(), self.entity_def)
            .map_err(|e| entity_addr_error(e, field.0))?;

        Ok(())
    }

    fn scalar_binop<T, F, O>(
        &mut self,
        ptr1: i16,
        ptr2: i16,
        out: i16,
        map: F,
    ) -> anyhow::Result<()>
    where
        VmScalar: TryInto<T>,
        <VmScalar as TryInto<T>>::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        F: FnOnce(T, T) -> O,
        O: Into<VmScalar>,
    {
        let val1: T = self.get_scalar(ptr1)?;
        let val2: T = self.get_scalar(ptr2)?;

        self.set(out, map(val1, val2))
    }

    // MUL_F: Float multiplication
    fn op_mul_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a * b)
    }

    // MUL_V: Vector dot-product
    fn op_mul_v(&mut self, v1_id: i16, v2_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_id)?;
        let v2 = self.get_vec3(v2_id)?;

        // log_op!(self; dot_id = MulV(v1, v2));

        self.set(out_ptr, v1.dot(v2))?;

        Ok(())
    }

    // MUL_FV: Component-wise multiplication of vector by scalar
    fn op_mul_fv(&mut self, f_id: i16, v_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let f: f32 = self.get_scalar(f_id)?;
        let v = self.get_vec3(v_id)?;

        // log_op!(self; prod_id = MulFV(f, v));

        self.set_vec3(out_ptr, f * v)?;

        Ok(())
    }

    // MUL_VF: Component-wise multiplication of vector by scalar
    fn op_mul_vf(&mut self, v_id: i16, f_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v = self.get_vec3(v_id)?;
        let f: f32 = self.get_scalar(f_id)?;

        // log_op!(self; prod_id = MulVF(v, f));

        self.set_vec3(out_ptr, f * v)?;

        Ok(())
    }

    // DIV: Float division
    fn op_div(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a / b)
    }

    // ADD_F: Float addition
    fn op_add_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a + b)
    }

    // ADD_V: Vector addition
    fn op_add_v(&mut self, v1_ptr: i16, v2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ptr)?;
        let v2 = self.get_vec3(v2_ptr)?;

        // log_op!(self; sum_id = AddV(v1, v2));

        self.set_vec3(out_ptr, v1 + v2)?;

        Ok(())
    }

    // SUB_F: Float subtraction
    fn op_sub_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a - b)
    }

    // SUB_V: Vector subtraction
    fn op_sub_v(&mut self, v1_id: i16, v2_id: i16, diff_id: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_id)?;
        let v2 = self.get_vec3(v2_id)?;

        // log_op!(self; diff_id = SubV(v1, v2));

        self.set_vec3(diff_id, v1 - v2)?;

        Ok(())
    }

    fn scalar_eq<T>(&mut self, ptr1: i16, ptr2: i16, out_ptr: i16) -> anyhow::Result<()>
    where
        VmScalar: TryInto<T>,
        <VmScalar as TryInto<T>>::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        T: PartialEq,
    {
        self.scalar_binop(ptr1, ptr2, out_ptr, |a: T, b: T| a == b)
    }

    // EQ_F: Test equality of two floats
    fn op_eq_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_eq::<f32>(f1_ptr, f2_ptr, out_ptr)
    }

    // EQ_V: Test equality of two vectors
    fn op_eq_v(&mut self, v1_ptr: i16, v2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ptr)?;
        let v2 = self.get_vec3(v2_ptr)?;

        // log_op!(self; eq_id = EqV(v1, v2));

        self.set(out_ptr, v1 == v2)?;

        Ok(())
    }

    // EQ_S: Test equality of two strings
    fn op_eq_s(&mut self, s1_ptr: i16, s2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let s1: StringRef = self.get_scalar(s1_ptr)?;
        let s2: StringRef = self.get_scalar(s2_ptr)?;

        let s1 = self.string_table.get(s1)?;
        let s2 = self.string_table.get(s2)?;

        // log_op!(self; ne_ofs = NeS(s1, s2));

        self.set(out_ptr, s1 == s2)?;

        Ok(())
    }

    // EQ_ENT: Test equality of two entities (by identity)
    fn op_eq_ent(&mut self, e1_ptr: i16, e2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_eq::<EntityRef>(e1_ptr, e2_ptr, out_ptr)
    }

    // EQ_FNC: Test equality of two functions (by identity)
    fn op_eq_fnc(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_eq::<VmFunctionRef>(f1_ptr, f2_ptr, out_ptr)
    }

    fn scalar_ne<T>(&mut self, ptr1: i16, ptr2: i16, out_ptr: i16) -> anyhow::Result<()>
    where
        VmScalar: TryInto<T>,
        <VmScalar as TryInto<T>>::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
        T: PartialEq,
    {
        self.scalar_binop(ptr1, ptr2, out_ptr, |a: T, b: T| a != b)
    }

    // NE_F: Test inequality of two floats
    fn op_ne_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_ne::<f32>(f1_ptr, f2_ptr, out_ptr)
    }

    // NE_V: Test inequality of two vectors
    fn op_ne_v(&mut self, v1_ofs: i16, v2_ofs: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ofs)?;
        let v2 = self.get_vec3(v2_ofs)?;

        // log_op!(self; ne_ofs = NeV(v1, v2));

        self.set(out_ptr, v1 != v2)?;

        Ok(())
    }

    // NE_S: Test inequality of two strings
    fn op_ne_s(&mut self, s1_ptr: i16, s2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let s1: StringRef = self.get_scalar(s1_ptr)?;
        let s2: StringRef = self.get_scalar(s2_ptr)?;

        let s1 = self.string_table.get(s1)?;
        let s2 = self.string_table.get(s2)?;

        self.set(out_ptr, s1 != s2)?;

        Ok(())
    }

    fn op_ne_ent(&mut self, e1_ptr: i16, e2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_ne::<EntityRef>(e1_ptr, e2_ptr, out_ptr)
    }

    fn op_ne_fnc(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_ne::<VmFunctionRef>(f1_ptr, f2_ptr, out_ptr)
    }

    // LE: Less than or equal to comparison
    fn op_le(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a <= b)
    }

    // GE: Greater than or equal to comparison
    fn op_ge(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a >= b)
    }

    // LT: Less than comparison
    fn op_lt(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a < b)
    }

    // GT: Greater than comparison
    fn op_gt(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a > b)
    }

    fn copy(&mut self, src_ptr: i16, dst_ptr: i16, _: i16) -> anyhow::Result<()> {
        self.set(dst_ptr, self.get_scalar::<_, VmScalar>(src_ptr)?)
    }

    // STORE_V
    fn op_store_v(&mut self, src_ofs: i16, dest_ofs: i16, unused: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_V"));
        }

        if (0..RESERVED_GLOBALS).contains(&dest_ofs) {
            // `qcc` uses `STORE_V` to copy function arguments of any type into the parameter slots,
            // so this has to be an untyped copy of three scalars.
            //
            // See https://github.com/id-Software/Quake-Tools/blob/master/qcc/pr_comp.c#L362
            for i in 0..3 {
                let value: VmScalar = self.get_scalar(src_ofs + i)?;
                self.set(dest_ofs + i, value)?;
            }
        } else {
            let value = self.get_vec3(src_ofs)?;
            self.set_vec3(dest_ofs, value)?;
        }

        Ok(())
    }

    fn not(&mut self, ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.set(out_ptr, self.get_scalar::<_, VmScalar>(ptr)?.is_false())
    }

    // NOT_V: Compare vec to { 0.0, 0.0, 0.0 }
    fn op_not_v(&mut self, v_id: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_V"));
        }

        self.set(out_ptr, self.get_vec3(v_id)? == Vec3::ZERO)
    }

    // NOT_S: Compare string to null string
    fn op_not_s(&mut self, s_ofs: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_S"));
        }

        let string_ref = self.get_scalar::<_, StringRef>(s_ofs)?;

        let is_null = string_ref.is_null() || self.string_table.get(string_ref)?.is_empty();

        self.set(out_ptr, is_null)
    }

    // NOT_F: Compare float to 0.0
    fn op_not_f(&mut self, f_id: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_F"));
        }

        self.not(f_id, out_ptr)
    }

    // NOT_FNC: Compare function to null function (0)
    fn op_not_fnc(&mut self, fnc_id_ptr: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_FNC"));
        }

        self.not(fnc_id_ptr, out_ptr)
    }

    // NOT_ENT: Compare entity to null entity (0)
    fn op_not_ent(&mut self, ent_ptr: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_ENT"));
        }

        self.not(ent_ptr, out_ptr)
    }

    // AND: Logical AND
    fn op_and(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: VmScalar, b: VmScalar| {
            !a.is_false() && !b.is_false()
        })
    }

    // OR: Logical OR
    fn op_or(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: VmScalar, b: VmScalar| {
            !a.is_false() || !b.is_false()
        })
    }

    // BIT_AND: Bitwise AND
    fn op_bit_and(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as i32 & b as i32) as f32
        })
    }

    // BIT_OR: Bitwise OR
    fn op_bit_or(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as i32 | b as i32) as f32
        })
    }

    fn op_state(
        &mut self,
        frame_id_addr: i16,
        think_function_addr: i16,
        _unused_c: i16,
    ) -> anyhow::Result<()> {
        let frame_id: f32 = self.get_scalar(frame_id_addr)?;
        let think_function_ref: VmFunctionRef = self.get_scalar(think_function_addr)?;

        let think_function = self.to_value(think_function_ref.into())?;

        self.context.dyn_state(frame_id, think_function)
    }
}
