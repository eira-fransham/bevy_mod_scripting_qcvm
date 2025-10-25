use arrayvec::ArrayVec;
#[cfg(feature = "reflect")]
use bevy_reflect::Reflect;
use glam::Vec3;
use itertools::Either;
use num_derive::FromPrimitive;
use std::{ffi::CStr, fmt, num::NonZeroIsize};
use tracing::{debug, error};

use crate::{
    ARG_ADDRS, ArgType, ExecutionCtx, OpResult,
    progs::{
        EntityRef, FieldPtr, FunctionRef, StringRef, VmScalar, VmValue,
        functions::{MAX_ARGS, Statement},
    },
    userdata::{Context, Entity as _, ErasedFunction, FnCall, Function as _},
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

impl ExecutionCtx<'_> {
    pub(crate) fn enter_builtin(
        &mut self,
        name: &CStr,
        builtin: &dyn ErasedFunction,
        num_args: usize,
    ) -> anyhow::Result<[VmScalar; 3]> {
        let sig = builtin.signature()?;
        if sig.len() != num_args {
            error!(
                "Builtin called with wrong number of args. Proceeding, but this is probably a bug"
            );
        }

        let args = ARG_ADDRS
            .step_by(3)
            .take(num_args)
            .zip(sig)
            .map(|(addr, ty)| match ty {
                ArgType::Vector => Ok(self.get_vec3(addr)?.into()),
                ArgType::Any => Ok(self.get::<_, VmValue>(addr)?),
                ArgType::Entity => Ok(self.get::<_, EntityRef>(addr)?.into()),
                ArgType::Function => Ok(self.get::<_, FunctionRef>(addr)?.into()),
                ArgType::Float => Ok(self.get::<_, f32>(addr)?.into()),
                ArgType::String => Ok(self.get::<_, StringRef>(addr)?.into()),
                ArgType::Void => Ok(Default::default()),
            })
            .flat_map(|value| match value {
                Ok(value) => Either::Left(<[VmScalar; 3]>::from(value).map(Ok).into_iter()),
                Err(e) => Either::Right(std::iter::once(Err(e))),
            })
            .collect::<anyhow::Result<ArrayVec<VmScalar, MAX_ARGS>>>()?;

        let vm_value: VmValue = self
            .with_builtin_call_context(name, args, |ctx| {
                builtin.dyn_call(FnCall { execution: ctx })
            })?
            .into();

        Ok(vm_value.into())
    }

    pub(crate) fn execute_statement(&mut self, statement: Statement) -> anyhow::Result<OpResult> {
        use Opcode as O;

        let op = statement.opcode;
        let a = statement.arg1;
        let b = statement.arg2;
        let c = statement.arg3;

        debug!("{:<12} {:>5} {:>5} {:>5}", op.to_string(), a, b, c);

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
                let func_ref: FunctionRef = self.get(a)?;

                let num_args = op as usize - O::Call0 as usize;

                let result = match func_ref {
                    FunctionRef::Ptr(p) => match self.functions.get(p.0)?.clone().try_into_quakec()
                    {
                        Ok(quakec) => self.execute(&quakec)?,
                        Err(builtin) => self.enter_builtin(
                            &builtin.name,
                            &*self.context.builtin(&builtin)?,
                            num_args,
                        )?,
                    },
                    FunctionRef::Extern(func) => {
                        self.enter_builtin(c"{anonymous}", &*func, num_args)?
                    }
                };

                self.set_return(result);

                OpResult::Continue

                // let Ok(def) = self.functions.get(f_to_call) else {
                //     return Err(ProgsError::with_msg("NULL function").into());
                // };

                // debug!("Calling function {f_to_call}");

                // let called_with_args = op as usize - O::Call0 as usize;
                // if def.argc != called_with_args {
                //     /// Seemingly `droptofloor` is defined with 2 args in the quakec defs
                //     /// but every example I can find calls it with 0 args and the
                //     /// implementation ignores any extra args. To prevent spamming the
                //     /// console with warnings, we ignore arg count mismatches for this
                //     /// function.
                //     const HACK_IGNORE_MISMATCH: &[&[u8]] = &[b"droptofloor"];

                //     let func_name = self.string_table.get(name_id).unwrap();
                //     if !HACK_IGNORE_MISMATCH.contains(&&*func_name) {
                //         self.cx.print_backtrace(&self.string_table, false);
                //         warn!(
                //             "Arg count mismatch calling {}: expected {}, found {}",
                //             func_name, def.argc, called_with_args,
                //         );
                //     }
                // }

                // if let FunctionKind::BuiltIn(b) = def.kind {
                //     self.enter_builtin(b, registry.reborrow(), vfs)?;
                //     debug!(
                //         "Returning from built-in function {}",
                //         self.string_table.get(name_id).unwrap()
                //     );
                // } else {
                //     self.enter_function(&self.string_table, &mut self.globals, f_to_call)?;
                //     continue;
                // }
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
            O::LoadF => self.op_load_f(a, b, c)?.into(),
            O::LoadV => self.op_load_v(a, b, c)?.into(),
            O::LoadS => self.op_load_s(a, b, c)?.into(),
            O::LoadEnt => self.op_load_ent(a, b, c)?.into(),
            O::LoadFld => panic!("load_fld not implemented"),
            O::LoadFnc => self.op_load_fnc(a, b, c)?.into(),
            O::Address => self.op_address(a, b, c)?.into(),
            O::StoreF => self.op_store_f(a, b, c)?.into(),
            O::StoreV => self.op_store_v(a, b, c)?.into(),
            O::StoreS => self.op_store_s(a, b, c)?.into(),
            O::StoreEnt => self.op_store_ent(a, b, c)?.into(),
            O::StoreFld => self.op_store_fld(a, b, c)?.into(),
            O::StoreFnc => self.op_store_fnc(a, b, c)?.into(),
            O::StorePF => self.op_storep_f(a, b, c)?.into(),
            O::StorePV => self.op_storep_v(a, b, c)?.into(),
            O::StorePS => self.op_storep_s(a, b, c)?.into(),
            O::StorePEnt => self.op_storep_ent(a, b, c)?.into(),
            O::StorePFld => panic!("storep_fld not implemented"),
            O::StorePFnc => self.op_storep_fnc(a, b, c)?.into(),
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

    pub(crate) fn op_if(&mut self, a: i16, b: i16, _c: i16) -> anyhow::Result<OpResult> {
        let cond = !self.get::<_, VmScalar>(a)?.is_null();
        // debug!("{op}: cond == {cond}");

        if cond {
            Ok(OpResult::Jump(NonZeroIsize::new(b as isize).ok_or_else(
                || anyhow::Error::msg("Tried to jump with an offset of 0"),
            )?))
        } else {
            Ok(OpResult::Continue)
        }
    }

    pub(crate) fn op_if_not(&mut self, a: i16, b: i16, _c: i16) -> anyhow::Result<OpResult> {
        let cond = !self.get::<_, VmScalar>(a)?.is_null();
        // debug!("{op}: cond == {cond}");

        if cond {
            Ok(OpResult::Continue)
        } else {
            Ok(OpResult::Jump(NonZeroIsize::new(b as isize).ok_or_else(
                || anyhow::Error::msg("Tried to jump with an offset of 0"),
            )?))
        }
    }

    pub(crate) fn op_goto(&mut self, a: i16, _b: i16, _c: i16) -> anyhow::Result<OpResult> {
        Ok(OpResult::Jump(NonZeroIsize::new(a as isize).ok_or_else(
            || anyhow::Error::msg("Tried to jump with an offset of 0"),
        )?))
    }

    pub(crate) fn op_return(&mut self, a: i16, b: i16, c: i16) -> anyhow::Result<[VmScalar; 3]> {
        let val1: VmScalar = self.get(a).unwrap_or_default();
        let val2: VmScalar = self.get(b).unwrap_or_default();
        let val3: VmScalar = self.get(c).unwrap_or_default();

        Ok([val1, val2, val3])
    }

    fn load_scalar(&mut self, entity: i16, field: i16, out_ptr: i16) -> anyhow::Result<()> {
        let ent: EntityRef = self.get(entity)?;
        let FieldPtr(ptr) = self.get(field)?;
        let field = self.entity_def.get_scalar(ptr)?;

        let value = ent.non_null()?.get_scalar(&*self.context, field)?;

        self.set(out_ptr, value)
    }

    // LOAD_F: load float field from entity
    pub(crate) fn op_load_f(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.load_scalar(entity, field, out_ptr)
    }

    // LOAD_V: load vector field from entity
    pub(crate) fn op_load_v(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        let ent: EntityRef = self.get(entity)?;
        let FieldPtr(ptr) = self.get(field)?;
        let field = self.entity_def.get_vector(ptr)?;

        let value: [VmScalar; 3] = ent.non_null()?.get(&*self.context, &field)?.into();

        self.set_vector(out_ptr, value)
    }

    pub(crate) fn op_load_s(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.load_scalar(entity, field, out_ptr)
    }

    pub(crate) fn op_load_ent(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.load_scalar(entity, field, out_ptr)
    }

    pub(crate) fn op_load_fnc(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.load_scalar(entity, field, out_ptr)
    }

    pub(crate) fn op_address(
        &mut self,
        entity: i16,
        field: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        let _ent_id: EntityRef = self.get(entity)?;
        let _fld_addr: FieldPtr = self.get(field)?;
        let _out_ptr = out_ptr;

        // Should be (a facsimile of) the byte offset of the field in the full entity array.
        todo!();

        // Ok(())
    }

    pub(crate) fn op_storep_f(
        &mut self,
        src_float_addr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_f: nonzero arg3"));
        }

        let _f: f32 = self.get(src_float_addr)?;
        let _out_ptr = out_ptr;

        todo!();

        // Ok(())
    }

    pub(crate) fn op_storep_v(
        &mut self,
        src_vector_addr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_v: nonzero arg3"));
        }

        let _v = self.get_vec3(src_vector_addr)?;
        let _out_ptr = out_ptr;

        todo!();

        // Ok(())
    }

    pub(crate) fn op_storep_s(
        &mut self,
        src_string_id_addr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_s: nonzero arg3"));
        }

        let _s: StringRef = self.get(src_string_id_addr)?;
        let _out_ptr = out_ptr;

        todo!();

        // Ok(())
    }

    pub(crate) fn op_storep_ent(
        &mut self,
        src_entity_id_addr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("storep_ent: nonzero arg3"));
        }

        let _e: EntityRef = self.get(src_entity_id_addr)?;
        let _out_ptr = out_ptr;

        todo!();

        // Ok(())
    }

    pub(crate) fn op_storep_fnc(
        &mut self,
        src_function_id_addr: i16,
        out_ptr: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg(format!(
                "storep_fnc: nonzero arg3 ({unused})"
            )));
        }

        let _f: FunctionRef = self.get(src_function_id_addr)?;
        let _out_ptr = out_ptr;

        todo!();

        // Ok(())
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
        let val1: T = self.get(ptr1)?;
        let val2: T = self.get(ptr2)?;

        self.set(out, map(val1, val2))
    }

    // MUL_F: Float multiplication
    pub(crate) fn op_mul_f(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a * b)
    }

    // MUL_V: Vector dot-product
    pub(crate) fn op_mul_v(&mut self, v1_id: i16, v2_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_id)?;
        let v2 = self.get_vec3(v2_id)?;

        // log_op!(self; dot_id = MulV(v1, v2));

        self.set(out_ptr, v1.dot(v2))?;

        Ok(())
    }

    // MUL_FV: Component-wise multiplication of vector by scalar
    pub(crate) fn op_mul_fv(&mut self, f_id: i16, v_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let f: f32 = self.get(f_id)?;
        let v = self.get_vec3(v_id)?;

        // log_op!(self; prod_id = MulFV(f, v));

        self.set_vec3(out_ptr, f * v)?;

        Ok(())
    }

    // MUL_VF: Component-wise multiplication of vector by scalar
    pub(crate) fn op_mul_vf(&mut self, v_id: i16, f_id: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v = self.get_vec3(v_id)?;
        let f: f32 = self.get(f_id)?;

        // log_op!(self; prod_id = MulVF(v, f));

        self.set_vec3(out_ptr, f * v)?;

        Ok(())
    }

    // DIV: Float division
    pub(crate) fn op_div(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a / b)
    }

    // ADD_F: Float addition
    pub(crate) fn op_add_f(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a + b)
    }

    // ADD_V: Vector addition
    pub(crate) fn op_add_v(
        &mut self,
        v1_ptr: i16,
        v2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ptr)?;
        let v2 = self.get_vec3(v2_ptr)?;

        // log_op!(self; sum_id = AddV(v1, v2));

        self.set_vec3(out_ptr, v1 + v2)?;

        Ok(())
    }

    // SUB_F: Float subtraction
    pub(crate) fn op_sub_f(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a - b)
    }

    // SUB_V: Vector subtraction
    pub(crate) fn op_sub_v(&mut self, v1_id: i16, v2_id: i16, diff_id: i16) -> anyhow::Result<()> {
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
    pub(crate) fn op_eq_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_eq::<f32>(f1_ptr, f2_ptr, out_ptr)
    }

    // EQ_V: Test equality of two vectors
    pub(crate) fn op_eq_v(&mut self, v1_ptr: i16, v2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ptr)?;
        let v2 = self.get_vec3(v2_ptr)?;

        // log_op!(self; eq_id = EqV(v1, v2));

        self.set(out_ptr, v1 == v2)?;

        Ok(())
    }

    // EQ_S: Test equality of two strings
    pub(crate) fn op_eq_s(&mut self, s1_ptr: i16, s2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let s1: StringRef = self.get(s1_ptr)?;
        let s2: StringRef = self.get(s2_ptr)?;

        let s1 = self.string_table.get(s1)?;
        let s2 = self.string_table.get(s2)?;

        // log_op!(self; ne_ofs = NeS(s1, s2));

        self.set(out_ptr, s1 == s2)?;

        Ok(())
    }

    // EQ_ENT: Test equality of two entities (by identity)
    pub(crate) fn op_eq_ent(
        &mut self,
        e1_ptr: i16,
        e2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_eq::<EntityRef>(e1_ptr, e2_ptr, out_ptr)
    }

    // EQ_FNC: Test equality of two functions (by identity)
    pub(crate) fn op_eq_fnc(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_eq::<FunctionRef>(f1_ptr, f2_ptr, out_ptr)
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
    pub(crate) fn op_ne_f(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_ne::<f32>(f1_ptr, f2_ptr, out_ptr)
    }

    // NE_V: Test inequality of two vectors
    pub(crate) fn op_ne_v(&mut self, v1_ofs: i16, v2_ofs: i16, out_ptr: i16) -> anyhow::Result<()> {
        let v1 = self.get_vec3(v1_ofs)?;
        let v2 = self.get_vec3(v2_ofs)?;

        // log_op!(self; ne_ofs = NeV(v1, v2));

        self.set(out_ptr, v1 != v2)?;

        Ok(())
    }

    // NE_S: Test inequality of two strings
    pub(crate) fn op_ne_s(&mut self, s1_ptr: i16, s2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        let s1: StringRef = self.get(s1_ptr)?;
        let s2: StringRef = self.get(s2_ptr)?;

        let s1 = self.string_table.get(s1)?;
        let s2 = self.string_table.get(s2)?;

        // log_op!(self; ne_ofs = NeS(s1, s2));

        self.set(out_ptr, s1 == s2)?;

        Ok(())
    }

    pub(crate) fn op_ne_ent(
        &mut self,
        e1_ptr: i16,
        e2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_ne::<EntityRef>(e1_ptr, e2_ptr, out_ptr)
    }

    pub(crate) fn op_ne_fnc(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_ne::<FunctionRef>(f1_ptr, f2_ptr, out_ptr)
    }

    // LE: Less than or equal to comparison
    pub(crate) fn op_le(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a <= b)
    }

    // GE: Greater than or equal to comparison
    pub(crate) fn op_ge(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a >= b)
    }

    // LT: Less than comparison
    pub(crate) fn op_lt(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a < b)
    }

    // GT: Greater than comparison
    pub(crate) fn op_gt(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| a > b)
    }

    fn copy(&mut self, src_ptr: i16, dst_ptr: i16) -> anyhow::Result<()> {
        self.set(dst_ptr, self.get::<_, VmScalar>(src_ptr)?)
    }

    // STORE_F
    pub(crate) fn op_store_f(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_F"));
        }

        self.copy(src_ofs, dest_ofs)
    }

    // STORE_V
    pub(crate) fn op_store_v(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_V"));
        }

        self.set_vec3(dest_ofs, self.get_vector(src_ofs)?)?;

        Ok(())
    }

    pub(crate) fn op_store_s(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_S"));
        }

        self.copy(src_ofs, dest_ofs)
    }

    pub(crate) fn op_store_ent(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_ENT"));
        }

        self.copy(src_ofs, dest_ofs)
    }

    pub(crate) fn op_store_fld(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_FLD"));
        }

        self.copy(src_ofs, dest_ofs)
    }

    pub(crate) fn op_store_fnc(
        &mut self,
        src_ofs: i16,
        dest_ofs: i16,
        unused: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg3 to STORE_FNC"));
        }

        self.copy(src_ofs, dest_ofs)
    }

    fn not(&mut self, ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.set(out_ptr, self.get::<_, VmScalar>(ptr)?.is_null())
    }

    // NOT_V: Compare vec to { 0.0, 0.0, 0.0 }
    pub(crate) fn op_not_v(&mut self, v_id: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_V"));
        }

        self.set(out_ptr, self.get_vec3(v_id)? == Vec3::ZERO)
    }

    // NOT_S: Compare string to null string
    pub(crate) fn op_not_s(&mut self, s_ofs: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_S"));
        }

        let string_ref = self.get::<_, StringRef>(s_ofs)?;

        let is_null = string_ref.is_null() || self.string_table.get(string_ref)?.is_empty();

        self.set(out_ptr, is_null)
    }

    // NOT_F: Compare float to 0.0
    pub(crate) fn op_not_f(&mut self, f_id: i16, unused: i16, out_ptr: i16) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_F"));
        }

        self.not(f_id, out_ptr)
    }

    // NOT_FNC: Compare function to null function (0)
    pub(crate) fn op_not_fnc(
        &mut self,
        fnc_id_ptr: i16,
        unused: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_FNC"));
        }

        self.not(fnc_id_ptr, out_ptr)
    }

    // NOT_ENT: Compare entity to null entity (0)
    pub(crate) fn op_not_ent(
        &mut self,
        ent_ptr: i16,
        unused: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        if unused != 0 {
            return Err(anyhow::Error::msg("Nonzero arg2 to NOT_ENT"));
        }

        self.not(ent_ptr, out_ptr)
    }

    // AND: Logical AND
    pub(crate) fn op_and(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: VmScalar, b: VmScalar| {
            !a.is_null() && !b.is_null()
        })
    }

    // OR: Logical OR
    pub(crate) fn op_or(&mut self, f1_ptr: i16, f2_ptr: i16, out_ptr: i16) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: VmScalar, b: VmScalar| {
            !a.is_null() || !b.is_null()
        })
    }

    // BIT_AND: Bitwise AND
    pub(crate) fn op_bit_and(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as usize & b as usize) as f32
        })
    }

    // BIT_OR: Bitwise OR
    pub(crate) fn op_bit_or(
        &mut self,
        f1_ptr: i16,
        f2_ptr: i16,
        out_ptr: i16,
    ) -> anyhow::Result<()> {
        self.scalar_binop(f1_ptr, f2_ptr, out_ptr, |a: f32, b: f32| {
            (a as usize | b as usize) as f32
        })
    }

    pub(crate) fn op_state(
        &mut self,
        _frame_id_addr: i16,
        _think_function_addr: i16,
        unused_c: i16,
    ) -> anyhow::Result<()> {
        if unused_c != 0 {
            return Err(anyhow::Error::msg(format!(
                "state: nonzero arg3 ({unused_c})"
            )));
        }

        // `state` is the only opcode that makes assumptions about the entity layout.
        // We should make it configurable by the consumer.
        todo!("`OP_STATE` not implemented - should be a builtin");
    }
}
