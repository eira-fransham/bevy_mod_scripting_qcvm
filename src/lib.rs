use std::{backtrace::Backtrace, borrow::Cow};

use bevy_ecs::component::Component;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{FunctionCallContext, WorldGuard};

use crate::{
    entity::EntityTypeDef,
    progs::{ExecutionContext, Functions, Globals, Opcode, ProgsError, StringTable},
};

pub mod entity;
pub mod progs;

const FUNCTION_CALL_CONTEXT: FunctionCallContext =
    FunctionCallContext::new(Language::External(Cow::Borrowed("quakec")));

/// Server-side level state.
#[derive(Component, Debug)]
pub struct QuakeCVm {
    /// Global values for QuakeC bytecode.
    globals: Globals,

    entity_def: EntityTypeDef,

    string_table: StringTable,

    /// Function definitions and bodies.
    functions: Functions,
}

#[derive(Debug)]
struct QuakeCVmRef<'a> {
    globals: &'a mut Globals,
    entity_def: &'a EntityTypeDef,
    string_table: &'a StringTable,
    functions: &'a Functions,
}

impl QuakeCVmRef<'_> {
    fn reborrow(&mut self) -> QuakeCVmRef<'_> {
        QuakeCVmRef {
            globals: self.globals,
            entity_def: self.entity_def,
            string_table: self.string_table,
            functions: self.functions,
        }
    }
}

impl QuakeCVm {
    /// Execute a QuakeC function in the VM.
    pub fn execute_program(&mut self, world_guard: WorldGuard<'_>) -> anyhow::Result<()> {
        use Opcode as O;

        let mut runaway = 10000;

        let ctx = ExecutionContext::new(&self.functions, &mut *self.globals, function);

        while self.cx.call_stack_depth() != exit_depth {
            runaway -= 1;

            if runaway == 0 {
                self.cx.print_backtrace(&self.string_table, false);
                return Err(ProgsError::LocalStackOverflow {
                    backtrace: Backtrace::capture(),
                }
                .into());
            }

            let statement = self.cx.load_statement();
            let op = statement.opcode;
            let a = statement.arg1;
            let b = statement.arg2;
            let c = statement.arg3;

            debug!("{:<12} {:>5} {:>5} {:>5}", op.to_string(), a, b, c);

            match op {
                // Control flow ================================================
                O::If => {
                    let cond = self.globals.get_int(a)? != 0;
                    debug!("{op}: cond == {cond}");

                    if cond {
                        self.cx.jump_relative(b);
                        continue;
                    }
                }

                O::IfNot => {
                    let cond = self.globals.get_int(a)? != 0;
                    debug!("{op}: cond != {cond}");

                    if !cond {
                        self.cx.jump_relative(b);
                        continue;
                    }
                }

                O::Goto => {
                    self.cx.jump_relative(a);
                    continue;
                }

                O::Call0
                | O::Call1
                | O::Call2
                | O::Call3
                | O::Call4
                | O::Call5
                | O::Call6
                | O::Call7
                | O::Call8 => {
                    let f_to_call = self.globals.get_function_id(a)?;

                    if f_to_call.0 == 0 {
                        return Err(ProgsError::with_msg("NULL function"));
                    }

                    let Ok(def) = self.cx.function_def(f_to_call) else {
                        return Err(ProgsError::with_msg("NULL function"));
                    };

                    let name_id = def.name_id;

                    debug!(
                        "Calling function {} ({:?})",
                        self.string_table.get(name_id).unwrap(),
                        f_to_call
                    );

                    let called_with_args = op as usize - Call0 as usize;
                    if def.argc != called_with_args {
                        /// Seemingly `droptofloor` is defined with 2 args in the quakec defs
                        /// but every example I can find calls it with 0 args and the
                        /// implementation ignores any extra args. To prevent spamming the
                        /// console with warnings, we ignore arg count mismatches for this
                        /// function.
                        const HACK_IGNORE_MISMATCH: &[&[u8]] = &[b"droptofloor"];

                        let func_name = self.string_table.get(name_id).unwrap();
                        if !HACK_IGNORE_MISMATCH.contains(&&*func_name) {
                            self.cx.print_backtrace(&self.string_table, false);
                            warn!(
                                "Arg count mismatch calling {}: expected {}, found {}",
                                func_name, def.argc, called_with_args,
                            );
                        }
                    }

                    if let FunctionKind::BuiltIn(b) = def.kind {
                        self.enter_builtin(b, registry.reborrow(), vfs)?;
                        debug!(
                            "Returning from built-in function {}",
                            self.string_table.get(name_id).unwrap()
                        );
                    } else {
                        self.cx
                            .enter_function(&self.string_table, &mut self.globals, f_to_call)?;
                        continue;
                    }
                }

                O::Done | O::Return => self.op_return(a, b, c)?,

                O::MulF => self.globals.op_mul_f(a, b, c)?,
                O::MulV => self.globals.op_mul_v(a, b, c)?,
                O::MulFV => self.globals.op_mul_fv(a, b, c)?,
                O::MulVF => self.globals.op_mul_vf(a, b, c)?,
                O::Div => self.globals.op_div(a, b, c)?,
                O::AddF => self.globals.op_add_f(a, b, c)?,
                O::AddV => self.globals.op_add_v(a, b, c)?,
                O::SubF => self.globals.op_sub_f(a, b, c)?,
                O::SubV => self.globals.op_sub_v(a, b, c)?,
                O::EqF => self.globals.op_eq_f(a, b, c)?,
                O::EqV => self.globals.op_eq_v(a, b, c)?,
                O::EqS => self.globals.op_eq_s(&self.string_table, a, b, c)?,
                O::EqEnt => self.globals.op_eq_ent(a, b, c)?,
                O::EqFnc => self.globals.op_eq_fnc(a, b, c)?,
                O::NeF => self.globals.op_ne_f(a, b, c)?,
                O::NeV => self.globals.op_ne_v(a, b, c)?,
                O::NeS => self.globals.op_ne_s(&self.string_table, a, b, c)?,
                O::NeEnt => self.globals.op_ne_ent(a, b, c)?,
                O::NeFnc => self.globals.op_ne_fnc(a, b, c)?,
                O::Le => self.globals.op_le(a, b, c)?,
                O::Ge => self.globals.op_ge(a, b, c)?,
                O::Lt => self.globals.op_lt(a, b, c)?,
                O::Gt => self.globals.op_gt(a, b, c)?,
                O::LoadF => self.op_load_f(a, b, c)?,
                O::LoadV => self.op_load_v(a, b, c)?,
                O::LoadS => self.op_load_s(a, b, c)?,
                O::LoadEnt => self.op_load_ent(a, b, c)?,
                O::LoadFld => panic!("load_fld not implemented"),
                O::LoadFnc => self.op_load_fnc(a, b, c)?,
                O::Address => self.op_address(a, b, c)?,
                O::StoreF => self.globals.op_store_f(a, b, c)?,
                O::StoreV => self.globals.op_store_v(a, b, c)?,
                O::StoreS => self.globals.op_store_s(a, b, c)?,
                O::StoreEnt => self.globals.op_store_ent(a, b, c)?,
                O::StoreFld => self.globals.op_store_fld(a, b, c)?,
                O::StoreFnc => self.globals.op_store_fnc(a, b, c)?,
                O::StorePF => self.op_storep_f(a, b, c)?,
                O::StorePV => self.op_storep_v(a, b, c)?,
                O::StorePS => self.op_storep_s(a, b, c)?,
                O::StorePEnt => self.op_storep_ent(a, b, c)?,
                O::StorePFld => panic!("storep_fld not implemented"),
                O::StorePFnc => self.op_storep_fnc(a, b, c)?,
                O::NotF => self.globals.op_not_f(a, b, c)?,
                O::NotV => self.globals.op_not_v(a, b, c)?,
                O::NotS => self.globals.op_not_s(a, b, c)?,
                O::NotEnt => self.globals.op_not_ent(a, b, c)?,
                O::NotFnc => self.globals.op_not_fnc(a, b, c)?,
                O::And => self.globals.op_and(a, b, c)?,
                O::Or => self.globals.op_or(a, b, c)?,
                O::BitAnd => self.globals.op_bit_and(a, b, c)?,
                O::BitOr => self.globals.op_bit_or(a, b, c)?,

                O::State => self.op_state(a, b, c)?,
            }

            // Increment program counter.
            self.cx.jump_relative(1);
        }

        Ok(())
    }

    // QuakeC instructions ====================================================

    pub fn op_return(&mut self, a: i16, b: i16, c: i16) -> progs::Result<()> {
        let val1 = self.globals.get_bytes(a)?;
        let val2 = self.globals.get_bytes(b)?;
        let val3 = self.globals.get_bytes(c)?;

        self.globals.put_bytes(val1, GLOBAL_ADDR_RETURN as i16)?;
        self.globals
            .put_bytes(val2, GLOBAL_ADDR_RETURN as i16 + 1)?;
        self.globals
            .put_bytes(val3, GLOBAL_ADDR_RETURN as i16 + 2)?;

        debug!(
            "Returning from quakec function {}",
            self.string_table
                .get(
                    self.cx
                        .function_def(self.cx.current_function())
                        .unwrap()
                        .name_id
                )
                .unwrap()
        );

        self.cx
            .leave_function(&self.string_table, &mut self.globals)?;

        Ok(())
    }

    // LOAD_F: load float field from entity
    pub fn op_load_f(&mut self, e_ofs: i16, e_f: i16, dest_ofs: i16) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(e_ofs)?;

        let fld_ofs = self.globals.get_field_addr(e_f)?;

        let f = self.world.get(ent_id)?.get_float(fld_ofs.0 as i16)?;
        if let Some(field) = FieldAddrFloat::from_usize(fld_ofs.0) {
            debug!("{:?}.{:?} = {}", ent_id, field, f);
        }
        self.globals.put_float(f, dest_ofs)?;

        Ok(())
    }

    // LOAD_V: load vector field from entity
    pub fn op_load_v(
        &mut self,
        ent_id_addr: i16,
        ent_vector_addr: i16,
        dest_addr: i16,
    ) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(ent_id_addr)?;
        let ent_vector = self.globals.get_field_addr(ent_vector_addr)?;
        let v = self.world.get(ent_id)?.get_vector(ent_vector.0 as i16)?;
        self.globals.put_vector(v, dest_addr)?;

        Ok(())
    }

    pub fn op_load_s(
        &mut self,
        ent_id_addr: i16,
        ent_string_id_addr: i16,
        dest_addr: i16,
    ) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(ent_id_addr)?;
        let ent_string_id = self.globals.get_field_addr(ent_string_id_addr)?;
        let s = self.world.get(ent_id)?.string_id(ent_string_id.0 as i16)?;
        self.globals.put_string_id(s, dest_addr)?;

        Ok(())
    }

    pub fn op_load_ent(
        &mut self,
        ent_id_addr: i16,
        ent_entity_id_addr: i16,
        dest_addr: i16,
    ) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(ent_id_addr)?;
        let ent_entity_id = self.globals.get_field_addr(ent_entity_id_addr)?;
        let e = self.world.get(ent_id)?.entity_id(ent_entity_id.0 as i16)?;
        self.globals.put_entity_id(e, dest_addr)?;

        Ok(())
    }

    pub fn op_load_fnc(
        &mut self,
        ent_id_addr: i16,
        ent_function_id_addr: i16,
        dest_addr: i16,
    ) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(ent_id_addr)?;
        let fnc_function_id = self.globals.get_field_addr(ent_function_id_addr)?;
        let f = self
            .world
            .get(ent_id)?
            .function_id(fnc_function_id.0 as i16)?;
        self.globals.put_function_id(f, dest_addr)?;

        Ok(())
    }

    pub fn op_address(
        &mut self,
        ent_id_addr: i16,
        fld_addr_addr: i16,
        dest_addr: i16,
    ) -> progs::Result<()> {
        let ent_id = self.globals.get_entity_id(ent_id_addr)?;
        let fld_addr = self.globals.get_field_addr(fld_addr_addr)?;
        self.globals.put_entity_field(
            self.world.ent_fld_addr_to_i32(EntityFieldAddr {
                entity_id: ent_id,
                field_addr: fld_addr,
            }),
            dest_addr,
        )?;

        Ok(())
    }

    pub fn op_storep_f(
        &mut self,
        src_float_addr: i16,
        dst_ent_fld_addr: i16,
        unused: i16,
    ) -> progs::Result<()> {
        if unused != 0 {
            return Err(ProgsError::with_msg("storep_f: nonzero arg3"));
        }

        let f = self.globals.get_float(src_float_addr)?;
        let ent_fld_addr = self
            .world
            .ent_fld_addr_from_i32(self.globals.get_entity_field(dst_ent_fld_addr)?);

        self.world
            .get_mut(ent_fld_addr.entity_id)?
            .put_float(f, ent_fld_addr.field_addr.0 as i16)?;

        Ok(())
    }

    pub fn op_storep_v(
        &mut self,
        src_vector_addr: i16,
        dst_ent_fld_addr: i16,
        unused: i16,
    ) -> progs::Result<()> {
        if unused != 0 {
            return Err(ProgsError::with_msg("storep_v: nonzero arg3"));
        }

        let v = self.globals.get_vector(src_vector_addr)?;
        let ent_fld_addr = self
            .world
            .ent_fld_addr_from_i32(self.globals.get_entity_field(dst_ent_fld_addr)?);
        self.world
            .get_mut(ent_fld_addr.entity_id)?
            .put_vector(v, ent_fld_addr.field_addr.0 as i16)?;

        Ok(())
    }

    pub fn op_storep_s(
        &mut self,
        src_string_id_addr: i16,
        dst_ent_fld_addr: i16,
        unused: i16,
    ) -> progs::Result<()> {
        if unused != 0 {
            return Err(ProgsError::with_msg("storep_s: nonzero arg3"));
        }

        let s = self.globals.string_id(src_string_id_addr)?;
        let ent_fld_addr = self
            .world
            .ent_fld_addr_from_i32(self.globals.get_entity_field(dst_ent_fld_addr)?);
        self.world
            .get_mut(ent_fld_addr.entity_id)?
            .put_string_id(s, ent_fld_addr.field_addr.0 as i16)?;

        Ok(())
    }

    pub fn op_storep_ent(
        &mut self,
        src_entity_id_addr: i16,
        dst_ent_fld_addr: i16,
        unused: i16,
    ) -> progs::Result<()> {
        if unused != 0 {
            return Err(ProgsError::with_msg("storep_ent: nonzero arg3"));
        }

        let e = self.globals.get_entity_id(src_entity_id_addr)?;
        let ent_fld_addr = self
            .world
            .ent_fld_addr_from_i32(self.globals.get_entity_field(dst_ent_fld_addr)?);
        self.world
            .get_mut(ent_fld_addr.entity_id)?
            .put_entity_id(e, ent_fld_addr.field_addr.0 as i16)?;

        Ok(())
    }

    pub fn op_storep_fnc(
        &mut self,
        src_function_id_addr: i16,
        dst_ent_fld_addr: i16,
        unused: i16,
    ) -> progs::Result<()> {
        if unused != 0 {
            return Err(ProgsError::with_msg(format!(
                "storep_fnc: nonzero arg3 ({unused})"
            )));
        }

        let f = self.globals.get_function_id(src_function_id_addr)?;
        let ent_fld_addr = self
            .world
            .ent_fld_addr_from_i32(self.globals.get_entity_field(dst_ent_fld_addr)?);
        self.world
            .get_mut(ent_fld_addr.entity_id)?
            .put_function_id(f, ent_fld_addr.field_addr.0 as i16)?;

        Ok(())
    }

    pub fn op_state(
        &mut self,
        frame_id_addr: i16,
        think_function_addr: i16,
        unused_c: i16,
    ) -> progs::Result<()> {
        if unused_c != 0 {
            return Err(ProgsError::with_msg(format!(
                "state: nonzero arg3 ({unused_c})"
            )));
        }

        let self_id = self.globals.get_entity_id(GlobalAddrEntity::Self_ as i16)?;
        let mut self_ent = self.world.get_mut(self_id)?;
        let next_think_time = self.globals.get_float(GlobalAddrFloat::Time as i16)? + 0.1;

        self_ent.put_float(next_think_time, FieldAddrFloat::NextThink as i16)?;

        let frame_id = self.globals.get_float(frame_id_addr)?;
        self_ent.put_float(frame_id, FieldAddrFloat::FrameId as i16)?;

        let think_func = self.globals.get_function_id(think_function_addr)?;
        self_ent.put_function_id(think_func, FieldAddrFunctionId::Think as _)?;

        Ok(())
    }
}
