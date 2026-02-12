//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate, work will continue here when that is done.

use std::{any::TypeId, borrow::Cow, ffi::CString, str::FromStr as _, sync::Arc};

use bevy_ecs::world::WorldId;
use bevy_log::error;
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    DynamicScriptFunction, DynamicScriptFunctionMut, ExternalError, FunctionCallContext,
    InteropError, ReflectBase, ReflectBaseType, ReflectReference, ScriptValue,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin,
    config::{GetPluginThreadConfig, ScriptingPluginConfiguration},
    event::CallbackLabel,
    make_plugin_config_static,
};
use bevy_mod_scripting_script::ScriptAttachment;
use qcvm::{FieldDef, QuakeCArgs, arrayvec::ArrayVec, userdata::QuakeCType};

pub struct QcScriptingPlugin {
    pub scripting_plugin: ScriptingPlugin<QcScriptingPlugin>,
}

#[derive(Debug)]
struct BevyScriptContext {
    // TODO
    #[expect(dead_code)]
    world_id: WorldId,
}

#[derive(Debug, PartialEq, Eq)]
struct BevyEntityHandle(bevy_ecs::entity::Entity);

impl QuakeCType for BevyEntityHandle {
    fn type_(&self) -> qcvm::Type {
        qcvm::Type::Entity
    }

    fn is_null(&self) -> bool {
        // TODO: Should be possible to reference worldspawn(?)
        false
    }
}

impl qcvm::userdata::EntityHandle for BevyEntityHandle {
    type Context = BevyScriptContext;
    type Error = InteropError;

    fn get(&self, _: &Self::Context, _: &FieldDef) -> Result<qcvm::Value, Self::Error> {
        todo!()
    }

    fn set(
        &self,
        _context: &mut Self::Context,
        _field: &FieldDef,
        _offset: Option<qcvm::VectorField>,
        _value: qcvm::Value,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl qcvm::userdata::Context for BevyScriptContext {
    fn builtin(
        &self,
        _def: &qcvm::BuiltinDef,
    ) -> Result<std::sync::Arc<Self::Function>, Self::Error> {
        todo!()
    }

    type Entity = BevyEntityHandle;

    type Function = BevyBuiltin;

    type Error = InteropError;
}

#[derive(Debug)]
struct BevyScriptArgs(Vec<ScriptValue>);

// TODO
#[expect(dead_code)]
#[derive(Debug, PartialEq)]
enum BevyBuiltin {
    Ref(DynamicScriptFunction),
    Mut(DynamicScriptFunctionMut),
}

impl QuakeCType for BevyBuiltin {
    fn type_(&self) -> qcvm::Type {
        qcvm::Type::Function
    }

    fn is_null(&self) -> bool {
        false
    }
}

const FUNCTION_CALL_CONTEXT: FunctionCallContext =
    FunctionCallContext::new(<QcScriptingPlugin as IntoScriptPluginParams>::LANGUAGE);

impl qcvm::userdata::Function for BevyBuiltin {
    type Context = BevyScriptContext;

    type Error = InteropError;

    fn signature(&self) -> Result<ArrayVec<qcvm::Type, { qcvm::MAX_ARGS }>, Self::Error> {
        let arg_info = match self {
            BevyBuiltin::Ref(func) => &func.info.arg_info,
            BevyBuiltin::Mut(func) => &func.info.arg_info,
        };

        Ok(arg_info
            .iter()
            .map(|arg| {
                // TODO: Do we need to consider `Vec<Vec3>` or other `Vec3`-like types?
                if arg.type_id == TypeId::of::<Vec3>() {
                    qcvm::Type::Vector
                } else {
                    qcvm::Type::AnyScalar
                }
            })
            .collect())
    }

    fn call(
        &self,
        _context: qcvm::userdata::FnCall<'_, Self::Context>,
    ) -> Result<qcvm::Value, Self::Error> {
        match self {
            BevyBuiltin::Ref(func) => {
                // TODO
                let args = [];
                func.call(args, FUNCTION_CALL_CONTEXT)
                    .and_then(|val| bms_script_value_to_qcvm_script_value(&val))
            }
            BevyBuiltin::Mut(func) => {
                // TODO
                let args = [];
                func.call(args, FUNCTION_CALL_CONTEXT)
                    .and_then(|val| bms_script_value_to_qcvm_script_value(&val))
            }
        }
    }
}

fn bms_script_value_to_qcvm_script_value(
    bms_value: &ScriptValue,
) -> Result<qcvm::Value, InteropError> {
    match bms_value {
        ScriptValue::Unit => Ok(qcvm::Value::Void),
        &ScriptValue::Bool(b) => Ok(b.into()),
        &ScriptValue::Integer(i) => Ok(i.into()),
        &ScriptValue::Float(f) => Ok(f.into()),
        ScriptValue::String(cow) => Ok(qcvm::Value::try_from(&cow[..])
            .map_err(|e| InteropError::External(ExternalError(e.into_boxed_dyn_error().into())))?),
        ScriptValue::List(script_values) => match &script_values[..] {
            [
                ScriptValue::Float(x),
                ScriptValue::Float(y),
                ScriptValue::Float(z),
            ] => Ok(qcvm::Value::Vector([*x, *y, *z].map(|v| v as f32).into())),
            _ => Err(InteropError::NotImplemented),
        },
        ScriptValue::Reference(ReflectReference {
            base:
                ReflectBaseType {
                    // TODO: We should be using a specific marker component ID
                    base_id: ReflectBase::Component(ent, _),
                    ..
                },
            ..
        }) => Ok(qcvm::Value::Entity(qcvm::EntityRef::Entity(Arc::new(
            BevyEntityHandle(*ent),
        )))),
        ScriptValue::FunctionMut(_) => todo!(),
        ScriptValue::Function(_) => todo!(),
        ScriptValue::Error(e) => Err(e.clone()),
        _ => Err(InteropError::NotImplemented),
    }
}

fn qcvm_script_value_to_bms_script_value(
    qcvm_value: &qcvm::Value,
) -> Result<ScriptValue, InteropError> {
    match qcvm_value {
        qcvm::Value::Void => Ok(ScriptValue::Unit),
        qcvm::Value::Entity(_) => todo!(),
        qcvm::Value::Function(_) => todo!(),
        qcvm::Value::Float(f) => Ok(ScriptValue::Float(*f as f64)),
        qcvm::Value::Vector(vec3) => Ok(ScriptValue::List(vec![
            ScriptValue::Float(vec3.x as f64),
            ScriptValue::Float(vec3.y as f64),
            ScriptValue::Float(vec3.z as f64),
        ])),
        qcvm::Value::String(cstr) => Ok(ScriptValue::String(
            cstr.to_string_lossy().into_owned().into(),
        )),
    }
}

impl QuakeCArgs for BevyScriptArgs {
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, Self::Error> {
        let arg = self
            .0
            .get(index)
            .ok_or(InteropError::ArgumentCountMismatch {
                expected: index,
                got: self.0.len(),
            })?;

        bms_script_value_to_qcvm_script_value(arg)
    }
}

fn qc_interop_error(e: qcvm::Error) -> InteropError {
    InteropError::External(ExternalError(e.into_boxed_dyn_error().into()))
}

impl IntoScriptPluginParams for QcScriptingPlugin {
    const LANGUAGE: Language = Language::External {
        name: Cow::Borrowed("QC"),
        one_indexed: false,
    };

    type C = qcvm::QuakeCVm;
    type R = ();

    fn build_runtime() -> Self::R {}

    fn handler() -> bevy_mod_scripting_core::handler::HandlerFn<Self> {
        fn qc_callback_handler(
            args: Vec<ScriptValue>,
            context_key: &ScriptAttachment,
            callback: &CallbackLabel,
            context: &mut qcvm::QuakeCVm,
            world_id: WorldId,
        ) -> Result<ScriptValue, InteropError> {
            if matches!(context_key, ScriptAttachment::StaticScript(_)) {
                error!(
                    "QuakeC script used as static script - it should be attached to the worldspawn entity"
                );
                return Err(InteropError::NotImplemented);
            }

            let callback_cstr = CString::from_str(callback.as_ref()).map_err(|_| {
                InteropError::Invariant(Box::new(format!(
                    "Callback name string contained internal NUL"
                )))
            })?;

            let val = context
                .run(
                    &mut BevyScriptContext { world_id },
                    callback_cstr,
                    BevyScriptArgs(args),
                )
                .map_err(qc_interop_error)?;

            qcvm_script_value_to_bms_script_value(&val)
        }

        qc_callback_handler
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
        fn qc_context_loader(
            _context_key: &ScriptAttachment,
            content: &[u8],
            _world_id: WorldId,
        ) -> Result<qcvm::QuakeCVm, InteropError> {
            Ok(qcvm::QuakeCVm::load(std::io::Cursor::new(content)).map_err(qc_interop_error)?)
        }

        qc_context_loader
    }

    fn context_reloader() -> bevy_mod_scripting_core::context::ContextReloadFn<Self> {
        todo!()
    }
}

make_plugin_config_static!(QcScriptingPlugin);

impl AsMut<ScriptingPlugin<Self>> for QcScriptingPlugin {
    fn as_mut(&mut self) -> &mut ScriptingPlugin<Self> {
        &mut self.scripting_plugin
    }
}
