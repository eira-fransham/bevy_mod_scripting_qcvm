use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use glam::Vec3;
use num_derive::FromPrimitive;
use std::{ffi::CStr, fmt, num::NonZeroIsize, sync::Arc};
use tracing::{debug, error};

use crate::{
    ArgAddr, ExecutionCtx, OpResult, QCMemory, ScopedAlloc, Type, function_args,
    progs::{
        EntityField, EntityRef, FieldPtr, StringRef, VmFunctionRef, VmScalar, VmValue,
        functions::{MAX_ARGS, Statement},
    },
    userdata::{ErasedContext, ErasedFunction, FnCall},
};

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
            error!(
                "Builtin called with wrong number of args. Proceeding, but this is probably a bug"
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
            .with_args(name, params, |ctx| {
                builtin.dyn_call(FnCall { execution: ctx })
            })?
            .into();

        Ok(vm_value.into())
    }
}

impl<Alloc> ExecutionCtx<'_, dyn ErasedContext, Alloc>
where
    Alloc: ScopedAlloc,
{
    pub(crate) fn execute_statement(&mut self, statement: Statement) -> anyhow::Result<OpResult> {
        use Opcode as O;

        let op = statement.opcode;
        let a = statement.arg1;
        let b = statement.arg2;
        let c = statement.arg3;

        debug!("{:<12} {:>5} {:>5} {:>5}", op.to_string(), a, b, c);
        println!("{:<12} {:>5} {:>5} {:>5}", op.to_string(), a, b, c);

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

                let result = match func_ref {
                    VmFunctionRef::Ptr(p) => {
                        match self.functions.get_by_index(p.0)?.clone().try_into_qc() {
                            Ok(quakec) => self.execute_def(&quakec)?,
                            Err(builtin) => self.enter_builtin(
                                &builtin.name,
                                &*self.context.dyn_builtin(&builtin)?,
                                num_args,
                            )?,
                        }
                    }
                    VmFunctionRef::Extern(func) => {
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

        let value =
            self.context
                .dyn_entity_get(ent.non_null()?.0, ptr.0.try_into()?, Type::AnyScalar)?;

        self.set(out_ptr, value)
    }

    // LOAD_V: load vector field from entity
    fn op_load_v(&mut self, entity: i16, field: i16, out_ptr: i16) -> anyhow::Result<()> {
        let ent: EntityRef = self.get_scalar(entity)?;
        let FieldPtr(ptr) = self.get_scalar(field)?;

        let value: Vec3 = self
            .context
            .dyn_entity_get(ent.non_null()?.0, ptr.0.try_into()?, Type::Vector)?
            .try_into()?;

        self.set_vector(out_ptr, value.into())
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
            .dyn_entity_set(entity.non_null()?.0, field.0.try_into()?, value)?;

        Ok(())
    }

    fn op_storep_v(&mut self, src_ptr: i16, out_ptr: i16, unused: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_f: nonzero arg3"));
        }

        let f = self.get_vec3(src_ptr)?;
        let EntityField { entity, field } = self.get_scalar(out_ptr)?;

        self.context
            .dyn_entity_set(entity.non_null()?.0, field.0.try_into()?, f.into())?;

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

        // log_op!(self; ne_ofs = NeS(s1, s2));

        self.set(out_ptr, s1 == s2)?;

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

        self.set_vec3(dest_ofs, self.get_vec3(src_ofs)?)?;

        Ok(())
    }

    fn not(&mut self, ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.set(out_ptr, self.get_scalar::<_, VmScalar>(ptr)?.is_null())
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
            !a.is_null() && !b.is_null()
        })
    }

    // OR: Logical OR
    fn op_or(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: VmScalar, b: VmScalar| {
            !a.is_null() || !b.is_null()
        })
    }

    // BIT_AND: Bitwise AND
    fn op_bit_and(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as usize & b as usize) as f32
        })
    }

    // BIT_OR: Bitwise OR
    fn op_bit_or(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as usize | b as usize) as f32
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

        let think_function: Arc<dyn ErasedFunction> =
            self.to_value(think_function_ref.into())?.try_into()?;

        self.context.dyn_state(frame_id, think_function)
    }
}

// Need `quake1` feature for tests.
#[cfg(test)]
mod test {
    use std::{ffi::CString, fmt, ops::Range, sync::Arc};

    use crate::{
        Address, EmptyAddress, HashMap, VectorField,
        userdata::{AddrError, ErasedEntityHandle},
    };
    use itertools::Itertools;
    use strum::VariantArray;

    use crate::{
        EntityRef, Type, Value,
        entity::EntityTypeDef,
        load::{LoadFn, Progs},
        ops::test::assembler::{FunctionBodyBuilder, Reg},
        progs::{
            FieldDef, FieldPtr, GlobalDef, GlobalPtr, Ptr, StringTable, VmScalarType, VmType,
            functions::{FunctionRegistry, MAX_ARGS, Statement},
            globals::GlobalRegistry,
        },
        userdata::{Context, EntityHandle, FnCall, Function, QCType},
    };

    impl AddrError<anyhow::Error> {
        /// Convert to anyhow, only used for tests
        pub(crate) fn into_arc_dyn_error(
            self,
        ) -> AddrError<Arc<dyn std::error::Error + Send + Sync + 'static>> {
            match self {
                Self::OutOfRange => AddrError::OutOfRange,
                Self::Other { error: e } => AddrError::Other {
                    error: e.into_boxed_dyn_error().into(),
                },
            }
        }
    }

    mod assembler {
        use std::{ffi::CStr, ops::Range, sync::Arc};

        use num::FromPrimitive;

        use crate::{
            Type, function_args,
            load::LoadFn,
            ops::Opcode,
            progs::{VmScalarType, VmType, functions::Statement},
        };

        /// Extremely bare-bones assembler for testing that makes no attempt to reuse registers.
        pub struct FunctionBodyBuilder<'a> {
            statements: &'a mut Vec<Statement>,
            arguments: Vec<Type>,
            arg_locals: Vec<Reg>,
            locals: Range<usize>,
            offset: usize,
            name: Arc<CStr>,
        }

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct Reg {
            offset: usize,
            ty: Type,
        }

        impl Reg {
            pub fn new(offset: usize, ty: Type) -> Self {
                Self { offset, ty }
            }
        }

        impl<'a> FunctionBodyBuilder<'a> {
            pub fn new(
                statements: &'a mut Vec<Statement>,
                locals_start: usize,
                name: Arc<CStr>,
                arguments: Vec<Type>,
            ) -> Self {
                let offset = statements.len();
                let mut cur_local = locals_start;
                let arg_locals = arguments
                    .iter()
                    .map(|arg_type| {
                        let start = cur_local;
                        let count = arg_type.arg_size();
                        cur_local += count as usize;
                        Reg {
                            offset: start,
                            ty: *arg_type,
                        }
                    })
                    .collect();

                Self {
                    statements,
                    arguments,
                    arg_locals,
                    locals: locals_start..cur_local,
                    offset,
                    name,
                }
            }

            pub fn build(self) -> LoadFn {
                LoadFn {
                    offset: self.offset as _,
                    name: self.name,
                    source: c"<test>".to_owned().into(),
                    locals: self.locals,
                    args: self
                        .arguments
                        .into_iter()
                        .map(|arg| arg.arg_size())
                        .collect(),
                }
            }
        }

        impl FunctionBodyBuilder<'_> {
            pub fn arg(&self, n: usize) -> Reg {
                self.arg_locals[n]
            }

            fn allocate_scalar(&mut self, ty: Type) -> Reg {
                let new_reg = self.locals.end;
                self.locals.end += ty.arg_size() as usize;
                Reg {
                    offset: new_reg,
                    ty,
                }
            }

            pub fn load_field(
                &mut self,
                entity_ref: i16,
                field_ref: i16,
                ty: impl Into<VmType>,
            ) -> Reg {
                let ty = ty.into();
                let opcode = match ty {
                    VmType::Vector => todo!(),
                    VmType::Scalar(VmScalarType::Float) => Opcode::LoadF,
                    VmType::Scalar(VmScalarType::String) => Opcode::LoadS,
                    VmType::Scalar(VmScalarType::Entity) => Opcode::LoadEnt,
                    VmType::Scalar(VmScalarType::FieldRef) => Opcode::LoadFld,
                    VmType::Scalar(VmScalarType::Function) => Opcode::LoadFnc,
                    _ => panic!("Unsupported field type!"),
                };

                let out = self.allocate_scalar(ty.try_into().unwrap());

                self.statements.extend([Statement {
                    opcode,
                    arg1: entity_ref,
                    arg2: field_ref,
                    arg3: out.offset as _,
                }]);

                out
            }

            pub fn mul(&mut self, in_a: Reg, in_b: Reg) -> Reg {
                let (ty, opcode) = match (in_a.ty, in_b.ty) {
                    (Type::Float | Type::AnyScalar, Type::Float | Type::AnyScalar) => {
                        (Type::Float, Opcode::MulF)
                    }
                    (Type::Vector, Type::Vector) => (Type::Vector, Opcode::MulV),
                    (Type::Vector, Type::Float | Type::AnyScalar) => (Type::Vector, Opcode::MulVF),
                    (Type::Float | Type::AnyScalar, Type::Vector) => (Type::Vector, Opcode::MulFV),
                    _ => panic!("Invalid args to `mul`"),
                };

                let out = self.allocate_scalar(ty);

                self.statements.extend([Statement {
                    opcode,
                    arg1: in_a.offset as _,
                    arg2: in_b.offset as _,
                    arg3: out.offset as _,
                }]);

                out
            }

            pub fn call(&mut self, idx: i16, args: &[Reg]) -> Reg {
                let opcode = Opcode::from_usize(Opcode::Call0 as usize + args.len()).unwrap();
                for (arg, dst) in args.iter().zip(function_args()) {
                    self.copy(*arg, Reg::new(dst.addr as usize, arg.ty));
                }

                self.statements.extend([Statement {
                    opcode,
                    arg1: idx,
                    arg2: 0,
                    arg3: 0,
                }]);

                self.ret_reg(Type::AnyScalar)
            }

            fn ret_reg(&self, ty: Type) -> Reg {
                Reg {
                    offset: crate::RETURN_ADDRS.start,
                    ty,
                }
            }

            fn copy(&mut self, src: Reg, dst: Reg) {
                let ty_to_check = if matches!(src.ty, Type::AnyScalar) {
                    dst.ty
                } else {
                    src.ty
                };

                let opcode = match ty_to_check {
                    Type::Vector => Opcode::StoreV,
                    Type::String => Opcode::StoreS,
                    Type::Function => Opcode::StoreF,
                    Type::Entity => Opcode::StoreEnt,
                    Type::Float => Opcode::StoreF,
                    Type::Void => return,
                    Type::AnyScalar => panic!("Can't copy any to any"),
                };

                self.statements.push(Statement {
                    opcode,
                    arg1: src.offset as _,
                    arg2: dst.offset as _,
                    arg3: 0,
                });
            }

            pub fn ret(&mut self, ofs: Reg) {
                self.statements.push(Statement {
                    opcode: Opcode::Return,
                    arg1: ofs.offset as _,
                    arg2: 0,
                    arg3: 0,
                });
            }
        }
    }

    #[derive(Debug)]
    struct DefinedFunction {
        /// For calling externally.
        function_id: Ptr,
        /// For use with `call` opcode.
        dynamic_id: Ptr,
    }

    struct HeaderBuilder {
        globals: Vec<GlobalDef>,
        global_idx: u16,
        global_values: Vec<u8>,
        fields: Vec<FieldDef>,
        field_idx: u16,
        strings: Vec<u8>,
        functions: Vec<LoadFn>,
        locals_start: usize,
        statements: Vec<Statement>,
        next_builtin: i32,
    }

    impl Default for HeaderBuilder {
        fn default() -> Self {
            const GLOBAL_RANGE: Range<usize> = 28..92;

            Self {
                globals: vec![],
                global_idx: GLOBAL_RANGE.start as u16,
                global_values: vec![0; GLOBAL_RANGE.start * std::mem::size_of::<u32>()],
                fields: vec![],
                field_idx: 0,
                strings: vec![],
                statements: vec![],
                functions: vec![],
                locals_start: GLOBAL_RANGE.end,
                next_builtin: -1,
            }
        }
    }

    impl HeaderBuilder {
        fn push_global(
            &mut self,
            name: impl AsRef<str>,
            ty: impl Into<VmType>,
            value: impl AsRef<[u8]>,
        ) -> GlobalPtr {
            let name = name.as_ref();
            let ty = ty.into();
            let value = value.as_ref();

            let ty_size = ty.num_elements();
            assert_eq!(value.len(), ty_size * 4);
            let offset = self.global_idx;
            self.global_idx = self.global_idx.checked_add(ty_size as u16).unwrap();
            self.globals.push(GlobalDef {
                save: false,
                type_: ty,
                offset,
                name: CString::new(name).unwrap().into(),
            });
            self.global_values.extend_from_slice(value);

            GlobalPtr(Ptr(offset as _))
        }

        fn push_field(&mut self, name: impl AsRef<str>, ty: impl Into<VmType>) -> FieldPtr {
            let name = name.as_ref();
            let ty = ty.into();

            let ty_size = ty.num_elements();
            let offset = self.field_idx;
            self.field_idx = self.field_idx.checked_add(ty_size as u16).unwrap();
            self.fields.push(FieldDef {
                type_: ty,
                offset,
                name: CString::new(name).unwrap().into(),
            });

            FieldPtr(Ptr(offset as _))
        }

        #[expect(dead_code)]
        fn push_str(&mut self, string: impl AsRef<str>) -> i32 {
            let string = string.as_ref();
            assert!(string.as_bytes().iter().all(|t| *t != 0));
            let out_idx = self.strings.len();
            self.strings.extend_from_slice(string.as_bytes());
            self.strings.push(0);
            out_idx.try_into().unwrap()
        }

        fn push_builtin(
            &mut self,
            name: impl AsRef<str>,
            args: impl IntoIterator<Item = Type>,
        ) -> DefinedFunction {
            let name = name.as_ref();
            let offset = self.next_builtin;
            self.next_builtin -= 1;

            self.functions.push(LoadFn {
                offset,
                name: CString::new(name).unwrap().into(),
                source: c"<builtin>".to_owned().into(),
                locals: 0..0,
                args: args.into_iter().map(|ty| ty.arg_size()).collect(),
            });

            DefinedFunction {
                function_id: Ptr(offset),
                dynamic_id: self
                    .push_global(
                        format!("{name}$addr"),
                        VmScalarType::Function,
                        offset.to_le_bytes(),
                    )
                    .0,
            }
        }

        fn push_function(
            &mut self,
            name: impl AsRef<str>,
            args: impl IntoIterator<Item = Type>,
            body: impl FnOnce(&mut FunctionBodyBuilder<'_>),
        ) -> DefinedFunction {
            let locals_start = self.locals_start;
            let name = name.as_ref();

            let mut builder = FunctionBodyBuilder::new(
                &mut self.statements,
                locals_start,
                CString::new(name).unwrap().into(),
                args.into_iter().collect(),
            );

            body(&mut builder);

            let load_fn = builder.build();

            self.locals_start = load_fn.locals.end;

            let function_id = load_fn.offset;

            self.functions.push(load_fn);

            DefinedFunction {
                function_id: Ptr(function_id),
                dynamic_id: self
                    .push_global(
                        format!("{name}$addr"),
                        VmScalarType::Function,
                        function_id.to_le_bytes(),
                    )
                    .0,
            }
        }

        fn build(self) -> Progs {
            Progs {
                globals: GlobalRegistry::new(dbg!(self.globals), &self.global_values).unwrap(),
                entity_def: EntityTypeDef::new(self.fields),
                string_table: StringTable::new(self.strings).unwrap(),
                functions: FunctionRegistry::new(self.statements.into(), &self.functions).unwrap(),
            }
        }
    }

    #[derive(Debug, PartialEq)]
    struct TestEnt {
        id: usize,
    }

    type TestFnBody = dyn Fn(&[Value]) -> Value + Send + Sync;

    struct TestFn {
        signature: &'static [Type],
        body: Box<TestFnBody>,
    }

    impl PartialEq for TestFn {
        fn eq(&self, other: &Self) -> bool {
            self.signature == other.signature && std::ptr::eq(&*self.body, &*other.body)
        }
    }

    impl fmt::Debug for TestFn {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TestFn")
                .field("signature", &self.signature)
                .field("body", &..)
                .finish()
        }
    }

    struct TestContext {
        fields: HashMap<(usize, &'static str), Value>,
    }

    #[derive(VariantArray, Debug, Copy, Clone, PartialEq, Eq)]
    enum FieldAddr {
        Float,
        Vector,
        VectorX,
        VectorY,
        VectorZ,
    }

    impl Address for FieldAddr {
        fn name(&self) -> &'static str {
            match self {
                FieldAddr::Float => "float_field",
                FieldAddr::Vector => "vector_field",
                FieldAddr::VectorX => "vector_field_x",
                FieldAddr::VectorY => "vector_field_y",
                FieldAddr::VectorZ => "vector_field_z",
            }
        }

        fn vector_field_or_scalar(&self) -> (Self, VectorField) {
            match self {
                FieldAddr::VectorX => (FieldAddr::Vector, VectorField::XOrScalar),
                FieldAddr::VectorY => (FieldAddr::Vector, VectorField::Y),
                FieldAddr::VectorZ => (FieldAddr::Vector, VectorField::Z),
                other => (*other, VectorField::XOrScalar),
            }
        }

        fn type_(&self) -> Type {
            match self {
                FieldAddr::Float => Type::Float,
                FieldAddr::Vector => Type::Vector,
                FieldAddr::VectorX => Type::Float,
                FieldAddr::VectorY => Type::Float,
                FieldAddr::VectorZ => Type::Float,
            }
        }

        fn from_u16_typed(val: u16, ty: Type) -> Option<Self> {
            let mut i = 0;
            while i < Self::VARIANTS.len() {
                let variant = Self::VARIANTS[i];

                if variant.to_u16() == val && variant.type_().typeck(&ty) {
                    return Some(variant);
                }

                i += 1;
            }

            None
        }

        fn to_u16(&self) -> u16 {
            match self {
                FieldAddr::Float => 0,
                FieldAddr::Vector | FieldAddr::VectorX => 1,
                FieldAddr::VectorY => 2,
                FieldAddr::VectorZ => 3,
            }
        }
    }

    const FIELDS: &[(&str, Value)] = &[
        ("float_field", Value::Float(100.)),
        ("vector_field", Value::Vector(glam::Vec3::new(0., 0., 0.))),
    ];
    const ENTITIES: &[usize] = &[0, 1, 2];

    impl Default for TestContext {
        fn default() -> Self {
            Self {
                fields: ENTITIES
                    .iter()
                    .cartesian_product(FIELDS)
                    .map(|(id, (name, val))| ((*id, *name), val.clone()))
                    .collect(),
            }
        }
    }

    impl QCType for TestEnt {
        fn type_(&self) -> Type {
            Type::Entity
        }

        fn is_null(&self) -> bool {
            false
        }
    }

    impl EntityHandle for TestEnt {
        type Context = TestContext;
        type Error = <TestContext as Context>::Error;
        type FieldAddr = FieldAddr;

        fn get(
            &self,
            context: &Self::Context,
            field: Self::FieldAddr,
        ) -> Result<Value, AddrError<Self::Error>> {
            let (field, offset) = field.vector_field_or_scalar();
            context
                .fields
                .get(&(self.id, field.name()))
                .and_then(|v| v.field(offset).ok())
                .ok_or_else(|| {
                    AddrError::Other {
                        error: anyhow::format_err!("No field with name {:?}", field.name()),
                    }
                    .into_arc_dyn_error()
                })
        }

        fn set(
            &self,
            context: &mut Self::Context,
            field: Self::FieldAddr,
            value: Value,
        ) -> Result<(), AddrError<Self::Error>> {
            let (field, offset) = field.vector_field_or_scalar();
            let existing_value: &mut Value = context
                .fields
                .get_mut(&(self.id, field.name()))
                .ok_or_else(|| {
                    AddrError::Other {
                        error: anyhow::format_err!("No field with name {:?}", field.name()),
                    }
                    .into_arc_dyn_error()
                })?;

            existing_value.set(offset, value).map_err(|e| {
                AddrError::Other {
                    error: anyhow::Error::from(e),
                }
                .into_arc_dyn_error()
            })?;

            Ok(())
        }

        fn from_erased_mut<F, O>(erased: u64, callback: F) -> Result<O, Self::Error>
        where
            F: FnOnce(&mut Self) -> O,
        {
            Ok(callback(&mut Self { id: erased as _ }))
        }

        fn to_erased(&self) -> u64 {
            self.id as _
        }
    }

    impl QCType for TestFn {
        fn type_(&self) -> Type {
            Type::Function
        }

        fn is_null(&self) -> bool {
            false
        }
    }

    impl Function for TestFn {
        type Context = TestContext;
        type Error = <TestContext as Context>::Error;

        fn signature(&self) -> Result<arrayvec::ArrayVec<Type, MAX_ARGS>, Self::Error> {
            Ok(self.signature.iter().copied().collect())
        }

        fn call(&self, context: FnCall<'_, Self::Context>) -> Result<Value, Self::Error> {
            let arguments = context.arguments(self.signature).collect::<Vec<_>>();

            Ok((self.body)(&arguments))
        }
    }

    impl Context for TestContext {
        type Entity = TestEnt;
        type Function = TestFn;
        type Error = Arc<dyn std::error::Error + Send + Sync>;
        type GlobalAddr = EmptyAddress;

        fn builtin(
            &self,
            def: &crate::progs::functions::BuiltinDef,
        ) -> Result<std::sync::Arc<Self::Function>, Self::Error> {
            match def.name.to_bytes() {
                b"mul" => Ok(Arc::new(TestFn {
                    signature: &[Type::Float, Type::Float],
                    body: Box::new(|args| {
                        let args: &[Value; 2] = args.try_into().unwrap();
                        let [a_f, b_f]: [f32; 2] =
                            args.each_ref().map(|v| v.clone().try_into().unwrap());

                        (a_f * b_f).into()
                    }),
                })),
                b"add" => Ok(Arc::new(TestFn {
                    signature: &[Type::Float, Type::Float],
                    body: Box::new(|args| {
                        let args: &[Value; 2] = args.try_into().unwrap();
                        let [a_f, b_f]: [f32; 2] =
                            args.each_ref().map(|v| v.clone().try_into().unwrap());

                        (a_f + b_f).into()
                    }),
                })),
                _ => Err(anyhow::format_err!(
                    "No builtin with name {}",
                    def.name.to_string_lossy()
                )
                .into_boxed_dyn_error()
                .into()),
            }
        }

        fn global(
            &self,
            _def: EmptyAddress,
        ) -> Result<Value, crate::userdata::AddrError<Self::Error>> {
            unreachable!()
        }

        fn set_global(
            &mut self,
            _def: EmptyAddress,
            _value: Value,
        ) -> Result<(), crate::userdata::AddrError<Self::Error>> {
            unreachable!()
        }
    }

    #[test]
    fn basic() {
        let mut header_builder = HeaderBuilder::default();

        let just_mul =
            header_builder.push_function("simple_mul_f", [Type::Float, Type::Float], |body| {
                let arg_0 = body.arg(0);
                let arg_1 = body.arg(1);
                let result = body.mul(arg_0, arg_1);
                body.ret(result);
            });

        let executor = crate::QCVm {
            progs: header_builder.build(),
        };

        let out: f32 = executor
            .run(
                &mut TestContext::default(),
                just_mul.function_id.0,
                (3f32, 4f32),
            )
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(out, 12.);
    }

    #[test]
    fn subcall() {
        let mut header_builder = HeaderBuilder::default();

        let constant =
            header_builder.push_global("my_const", VmScalarType::Float, 10f32.to_le_bytes());

        let mul_by_10 = header_builder.push_function("mul_by_10", [Type::Float], |body| {
            let arg_0 = body.arg(0);
            let arg_1 = Reg::new(constant.0.0 as _, Type::Float);
            let result = body.mul(arg_0, arg_1);
            body.ret(result);
        });

        let just_mul =
            header_builder.push_function("mul_two_by_10", [Type::Float, Type::Float], |body| {
                let arg_0 = body.arg(0);
                let arg_1 = body.arg(1);
                let ret = body.call(mul_by_10.dynamic_id.0 as _, &[arg_1]);
                let result = body.mul(arg_0, ret);
                body.ret(result);
            });

        let executor = crate::QCVm {
            progs: header_builder.build(),
        };

        let out: f32 = executor
            .run(
                &mut TestContext::default(),
                just_mul.function_id.0,
                (3f32, 4f32),
            )
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(out, 120.);
    }

    #[test]
    fn call_builtin() {
        let mut header_builder = HeaderBuilder::default();

        let mul_builtin = header_builder.push_builtin("mul", [Type::Float]);

        let just_mul = header_builder.push_function(
            "mul_three_with_builtin",
            [Type::Float, Type::Float, Type::Float],
            |body| {
                let arg_0 = body.arg(0);
                let arg_1 = body.arg(1);
                let arg_2 = body.arg(2);
                let ret = body.call(mul_builtin.dynamic_id.0 as _, &[arg_1, arg_2]);
                let result = body.mul(arg_0, ret);
                body.ret(result);
            },
        );

        let executor = crate::QCVm {
            progs: header_builder.build(),
        };

        let out: f32 = executor
            .run(
                &mut TestContext::default(),
                just_mul.function_id.0,
                (3f32, 4f32, 5f32),
            )
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(out, 60.);
    }

    #[test]
    fn ent_fields() {
        let mut header_builder = HeaderBuilder::default();

        let ent_global_0 =
            header_builder.push_global("ent_id_0", VmScalarType::Entity, 0i32.to_le_bytes());
        let ent_global_1 =
            header_builder.push_global("ent_id_1", VmScalarType::Entity, 0i32.to_le_bytes());
        let ent_global_2 =
            header_builder.push_global("ent_id_2", VmScalarType::Entity, 0i32.to_le_bytes());
        let float_field = header_builder.push_field("float_field", VmScalarType::Float);
        let field_ref = header_builder.push_global(
            "ent_field_float",
            VmScalarType::FieldRef,
            float_field.0.0.to_le_bytes(),
        );

        let _just_mul = header_builder.push_function("mul_three_from_fields", [], |body| {
            let ent_0_fld = body.load_field(
                ent_global_0.0.0 as _,
                field_ref.0.0 as _,
                VmScalarType::Float,
            );
            let ent_1_fld = body.load_field(
                ent_global_1.0.0 as _,
                field_ref.0.0 as _,
                VmScalarType::Float,
            );
            let ent_2_fld = body.load_field(
                ent_global_2.0.0 as _,
                field_ref.0.0 as _,
                VmScalarType::Float,
            );
            let mul_tmp_0 = body.mul(ent_0_fld, ent_1_fld);
            let result = body.mul(mul_tmp_0, ent_2_fld);
            body.ret(result);
        });

        let mut executor = crate::QCVm {
            progs: (header_builder.build()),
        };

        executor
            .progs
            .globals
            .get_mut(ent_global_0.0.0)
            .unwrap()
            .value = EntityRef::Entity(ErasedEntityHandle(0)).into();
        executor
            .progs
            .globals
            .get_mut(ent_global_1.0.0)
            .unwrap()
            .value = EntityRef::Entity(ErasedEntityHandle(1)).into();
        executor
            .progs
            .globals
            .get_mut(ent_global_2.0.0)
            .unwrap()
            .value = EntityRef::Entity(ErasedEntityHandle(2)).into();

        let out: f32 = executor
            .run(&mut TestContext::default(), c"mul_three_from_fields", ())
            .unwrap()
            .try_into()
            .unwrap();

        assert_eq!(out, 100f32.powi(3));
    }

    #[test]
    fn per_execution_globals() {
        // TODO: Test per-execution globals (e.g. self, other)
    }
}
