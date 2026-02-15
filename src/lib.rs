//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate, work will continue here when that is done.

use std::{any::TypeId, borrow::Cow, ffi::CString, str::FromStr as _, sync::Arc};

use bevy_ecs::{component::Component, entity::Entity, world::WorldId};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    DynamicScriptFunction, DynamicScriptFunctionMut, ExternalError, FunctionCallContext,
    InteropError, IntoNamespace, Namespace, ReflectAccessId, ReflectBase, ReflectBaseType,
    ReflectReference, ScriptValue, ThreadWorldContainer,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin,
    config::{GetPluginThreadConfig, ScriptingPluginConfiguration},
    event::CallbackLabel,
    make_plugin_config_static,
};
use bevy_mod_scripting_script::ScriptAttachment;
use qcvm::{
    ArgError, FieldDef, GlobalDef, QCParams, VectorField, anyhow,
    arrayvec::ArrayVec,
    userdata::{AddrErr, QuakeCType},
};

pub struct QcScriptingPlugin {
    pub scripting_plugin: ScriptingPlugin<QcScriptingPlugin>,
}

#[derive(Debug)]
struct BevyScriptContext {
    worldspawn: Entity,
    special_args: hashbrown::HashMap<CString, qcvm::Value>,
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

#[derive(Component)]
pub struct QCEntity;

#[derive(Component)]
pub struct QCWorldspawn;

impl qcvm::userdata::EntityHandle for BevyEntityHandle {
    type Context = BevyScriptContext;
    type Error = InteropError;

    fn get(&self, _: &Self::Context, field: &FieldDef) -> Result<qcvm::Value, Self::Error> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let get = function_registry.read().magic_functions.get;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(ReflectAccessId::for_global(), |world| {
                // TODO: We can probably use the `quake1` module defs somehow here(?)
                let key = if field.name.is_empty() {
                    ScriptValue::Integer(field.offset as i64)
                } else {
                    ScriptValue::String(field.name.to_string_lossy().into_owned().into())
                };

                get(
                    FUNCTION_CALL_CONTEXT,
                    ReflectReference::new_component_ref::<QCEntity>(self.0, world.clone())?,
                    key,
                )
                .and_then(|v| bms_script_value_to_qcvm_script_value(&v))
            })
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn set(
        &self,
        _context: &mut Self::Context,
        field: &FieldDef,
        offset: Option<qcvm::VectorField>,
        value: qcvm::Value,
    ) -> Result<(), Self::Error> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let (get, set) = {
            let function_registry = function_registry.read();

            (
                function_registry.magic_functions.get,
                function_registry.magic_functions.set,
            )
        };

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(ReflectAccessId::for_global(), |world| {
                let key = if field.name.is_empty() {
                    ScriptValue::Integer(field.offset as i64)
                } else {
                    ScriptValue::String(field.name.to_string_lossy().into_owned().into())
                };

                // TODO: This whole process is horribly wasteful for the very-common case of setting a single field of a vector.
                //       While it is built for flexibility over speed, this feels like a bit much.
                let new_value = if let Some(offset) = offset {
                    // Get old value
                    let mut old = get(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCEntity>(self.0, world.clone())?,
                        key.clone(),
                    )
                    .and_then(|v| bms_script_value_to_qcvm_script_value(&v))?;

                    old.set(Some(offset), value)
                        // TODO
                        .map_err(|_| InteropError::NotImplemented)?;

                    old
                } else {
                    value
                };

                let new_value = qcvm_script_value_to_bms_script_value(&new_value)?;

                set(
                    FUNCTION_CALL_CONTEXT,
                    ReflectReference::new_component_ref::<QCEntity>(self.0, world.clone())?,
                    key,
                    new_value,
                )
            })
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn from_erased_mut<F, O>(erased: u64, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&mut Self) -> O,
    {
        let ent_id = bevy_ecs::entity::Entity::from_bits(erased);

        Ok(callback(&mut Self(ent_id)))
    }

    fn to_erased(&self) -> u64 {
        self.0.to_bits()
    }
}

pub struct QCBuiltin;

impl qcvm::userdata::Context for BevyScriptContext {
    type Entity = BevyEntityHandle;
    type Function = BevyBuiltin;
    type Error = InteropError;

    fn builtin(
        &self,
        def: &qcvm::BuiltinDef,
    ) -> Result<std::sync::Arc<Self::Function>, Self::Error> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient (especially
        //       when we need to look at both builtin and global namespaces).
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let function_registry = function_registry.read();

        // TODO: This is a wasteful way to do lookup, but the API of `bevy_mod_scripting` doesn't really give us a choice
        let def_name = def.name.to_string_lossy().into_owned();

        // TODO: Just in case some progs.dats use renamed builtins, we should have some way of looking up via
        //       index, not just name.
        if let Ok(func) =
            function_registry.get_function(QCBuiltin::into_namespace(), def_name.clone())
        {
            Ok(Arc::new(BevyBuiltin::Ref(func.clone())))
        } else if let Ok(func) = function_registry.get_function(Namespace::Global, def_name) {
            Ok(Arc::new(BevyBuiltin::Ref(func.clone())))
        } else {
            Err(InteropError::NotImplemented)
        }
    }

    fn global(&self, def: &GlobalDef) -> Result<qcvm::Value, AddrErr<InteropError>> {
        if !qcvm::quake1::GLOBALS_RANGE.contains(&(def.offset as usize)) {
            return Err(AddrErr::OutOfRange);
        }

        if let Some(val) = self.special_args.get(&*def.name) {
            return Ok(val.clone());
        }

        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let get = function_registry.read().magic_functions.get;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(
                ReflectAccessId::for_global(),
                |world| -> Result<_, AddrErr<InteropError>> {
                    // TODO: We can probably use the `quake1` module defs somehow here(?)
                    let key = if def.name.is_empty() {
                        ScriptValue::Integer(def.offset as i64)
                    } else {
                        ScriptValue::String(def.name.to_string_lossy().into_owned().into())
                    };

                    Ok(get(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCWorldspawn>(
                            self.worldspawn,
                            world.clone(),
                        )?,
                        key,
                    )
                    .and_then(|v| bms_script_value_to_qcvm_script_value(&v))?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn set_global(
        &mut self,
        def: &GlobalDef,
        offset: Option<VectorField>,
        value: qcvm::Value,
    ) -> Result<(), AddrErr<InteropError>> {
        if !qcvm::quake1::GLOBALS_RANGE.contains(&(def.offset as usize)) {
            return Err(AddrErr::OutOfRange);
        }

        if let Some(val) = self.special_args.get_mut(&*def.name) {
            val.set(offset, value)
                .map_err(|e| InteropError::External(ExternalError(Arc::new(e))))?;
            return Ok(());
        }

        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let (get, set) = {
            let function_registry = function_registry.read();

            (
                function_registry.magic_functions.get,
                function_registry.magic_functions.set,
            )
        };

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(
                ReflectAccessId::for_global(),
                |world| -> Result<_, AddrErr<InteropError>> {
                    let key = if def.name.is_empty() {
                        ScriptValue::Integer(def.offset as i64)
                    } else {
                        ScriptValue::String(def.name.to_string_lossy().into_owned().into())
                    };

                    // TODO: This whole process is horribly wasteful for the very-common case of setting a single def of a vector.
                    //       While it is built for flexibility over speed, this feels like a bit much.
                    let new_value = if let Some(offset) = offset {
                        // Get old value
                        let mut old = get(
                            FUNCTION_CALL_CONTEXT,
                            ReflectReference::new_component_ref::<QCWorldspawn>(
                                self.worldspawn,
                                world.clone(),
                            )?,
                            key.clone(),
                        )
                        .and_then(|v| bms_script_value_to_qcvm_script_value(&v))?;

                        old.set(Some(offset), value)
                            // TODO
                            .map_err(|_| InteropError::NotImplemented)?;

                        old
                    } else {
                        value
                    };

                    let new_value = qcvm_script_value_to_bms_script_value(&new_value)?;

                    Ok(set(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCWorldspawn>(
                            self.worldspawn,
                            world.clone(),
                        )?,
                        key,
                        new_value,
                    )?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
    }
}

#[derive(Debug)]
struct BevyScriptArgs<'a>(&'a [ScriptValue]);

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
        context: qcvm::userdata::FnCall<'_, Self::Context>,
    ) -> Result<qcvm::Value, Self::Error> {
        let sig = self.signature()?;
        let args = context.arguments(&sig);
        let args = args
            .map(|val| qcvm_script_value_to_bms_script_value(&val))
            .collect::<Result<ArrayVec<_, { qcvm::MAX_ARGS }>, _>>()?;
        match self {
            BevyBuiltin::Ref(func) => func
                .call(args, FUNCTION_CALL_CONTEXT)
                .and_then(|val| bms_script_value_to_qcvm_script_value(&val)),
            BevyBuiltin::Mut(func) => func
                .call(args, FUNCTION_CALL_CONTEXT)
                .and_then(|val| bms_script_value_to_qcvm_script_value(&val)),
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
        }) => Ok(qcvm::Value::Entity(qcvm::EntityRef::new(BevyEntityHandle(
            *ent,
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

impl QCParams for BevyScriptArgs<'_> {
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, ArgError<Self::Error>> {
        let arg = self.0.get(index).ok_or_else(|| ArgError::ArgOutOfRange {
            len: self.count(),
            index,
        })?;

        Ok(bms_script_value_to_qcvm_script_value(arg)?)
    }

    fn count(&self) -> usize {
        self.0.len()
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
            _world_id: WorldId,
        ) -> Result<ScriptValue, InteropError> {
            let worldspawn = match context_key {
                ScriptAttachment::EntityScript(entity, _) => *entity,
                ScriptAttachment::StaticScript(_) => {
                    return Err(qc_interop_error(anyhow::format_err!(
                        "progs.dat must be attached to a world entity"
                    )));
                }
            };

            let mut args_ref = &args[..];

            let special = match args_ref.get(0) {
                // QuakeC does not support structs, so if any parameter is a struct then we can treat it as
                // special args. We only support passing special args in the first arg slot though, for
                // consistency.
                Some(ScriptValue::Map(map)) => {
                    args_ref = &args_ref[1..];
                    map.into_iter()
                        .map(|(k, v)| {
                            Ok((
                                CString::new(&k[..]).map_err(|e| {
                                    InteropError::External(ExternalError(Arc::new(e)))
                                })?,
                                bms_script_value_to_qcvm_script_value(v)?,
                            ))
                        })
                        .collect::<Result<_, InteropError>>()?
                }

                _ => Default::default(),
            };

            let callback_cstr = CString::from_str(callback.as_ref()).map_err(|_| {
                InteropError::Invariant(Box::new(
                    "Callback name string contained internal NUL".to_string(),
                ))
            })?;

            let val = context
                .run(
                    // TODO: Implement special args (should not be part of callargs)
                    &mut BevyScriptContext {
                        worldspawn,
                        special_args: special,
                    },
                    callback_cstr,
                    qcvm::CallArgs {
                        params: BevyScriptArgs(args_ref),
                        // special,
                    },
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
            qcvm::QuakeCVm::load(std::io::Cursor::new(content)).map_err(qc_interop_error)
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
