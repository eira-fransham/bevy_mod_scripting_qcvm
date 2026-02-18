//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate. This is just bindings for `bevy_mod_scripting`.
//!
//! To add builtins to the engine, add callbacks to the [`QCBuiltin`] namespace. To add entity field getters and setters, use
//! the [`QCEntity`] namespace. To add global getters and setters, use the `QCWorldspawn` namespace.

#![deny(missing_docs)]

use std::{
    any::TypeId, borrow::Cow, ffi::CString, fmt, marker::PhantomData, str::FromStr as _, sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, world::WorldId};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    DynamicScriptFunction, ExternalError, FunctionCallContext, InteropError, IntoNamespace,
    Namespace, ReflectAccessId, ReflectBase, ReflectBaseType, ReflectReference, ScriptValue,
    ThreadWorldContainer,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin, config::GetPluginThreadConfig, event::CallbackLabel,
};
use bevy_mod_scripting_script::ScriptAttachment;
use qcvm::{
    Address, ArgError, EmptyAddress, QCParams, anyhow,
    arrayvec::ArrayVec,
    userdata::{AddrError, QCType},
};

pub use qcvm;

/// The main "entry point" plugin for adding QC Scripting
pub struct QCScriptingPlugin<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address + 'static,
    FieldAddr: Address + 'static,
    Self: GetPluginThreadConfig<Self>,
{
    /// The inner config for the plugin
    pub scripting_plugin: ScriptingPlugin<QCScriptingPlugin<GlobalAddr, FieldAddr>>,
}

// HACK: This should be configurable elsewhere, but this is ok for now
const SPECIAL_ARGS_ARE_IMMUTABLE: bool = true;

#[derive(Debug)]
struct BevyScriptContext<GlobalAddr, FieldAddr> {
    worldspawn: Entity,
    /// Per-execution globals such as `self` and `other`
    special_args: hashbrown::HashMap<u16, qcvm::Value>,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

#[derive(PartialEq, Eq)]
struct BevyEntityHandle<GlobalAddr, FieldAddr> {
    id: bevy_ecs::entity::Entity,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

impl<G, F> fmt::Debug for BevyEntityHandle<G, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BevyEntityHandle").field(&self.id).finish()
    }
}

impl<G, F> QCType for BevyEntityHandle<G, F> {
    fn type_(&self) -> qcvm::Type {
        qcvm::Type::Entity
    }

    fn is_null(&self) -> bool {
        // TODO: Should be possible to reference worldspawn(?)
        false
    }
}

/// Marker component for individual entities controlled by QC
#[derive(Component)]
pub struct QCEntity;

/// Marker component for the worldspawn entity, where globals will be stored
#[derive(Component)]
pub struct QCWorldspawn;

impl<GlobalAddr, FieldAddr> qcvm::userdata::EntityHandle for BevyEntityHandle<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address,
    FieldAddr: Address,
{
    type Context = BevyScriptContext<GlobalAddr, FieldAddr>;
    type Error = InteropError;
    type FieldAddr = FieldAddr;

    fn get(
        &self,
        _: &Self::Context,
        field: Self::FieldAddr,
    ) -> Result<qcvm::Value, AddrError<Self::Error>> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let get = function_registry.read().magic_functions.get;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(ReflectAccessId::for_global(), |world| {
                let key = ScriptValue::String(field.name().into());

                get(
                    FUNCTION_CALL_CONTEXT,
                    ReflectReference::new_component_ref::<QCEntity>(self.id, world.clone())?,
                    key,
                )
                .or_else(|e| match e {
                    InteropError::InvalidIndex { .. } => Ok(ScriptValue::Unit),
                    _ => Err(e),
                })
                .and_then(|v| bms_script_value_to_qcvm_script_value(&v))
                .map_err(Into::into)
            })
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn set(
        &self,
        _context: &mut Self::Context,
        field: Self::FieldAddr,
        value: qcvm::Value,
    ) -> Result<(), AddrError<Self::Error>> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let set = function_registry.read().magic_functions.set;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(ReflectAccessId::for_global(), |world| {
                let key = ScriptValue::String(field.name().into());

                let new_value = qcvm_script_value_to_bms_script_value(&value)?;

                set(
                    FUNCTION_CALL_CONTEXT,
                    ReflectReference::new_component_ref::<QCEntity>(self.id, world.clone())?,
                    key,
                    new_value,
                )
                .map_err(Into::into)
            })
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn from_erased_mut<F, O>(erased: u64, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&mut Self) -> O,
    {
        let id = bevy_ecs::entity::Entity::from_bits(erased);

        Ok(callback(&mut Self {
            id,
            _phantom: PhantomData,
        }))
    }

    fn to_erased(&self) -> u64 {
        self.id.to_bits()
    }
}

fn interop_error_to_addr_error<T>(
    res: Result<T, InteropError>,
) -> Result<T, AddrError<InteropError>> {
    res.map_err(|e| match e {
        InteropError::LengthMismatch { .. } | InteropError::InvalidIndex { .. } => {
            AddrError::OutOfRange
        }
        other => AddrError::Other { error: other },
    })
}

/// Namespace for QC builtins
pub struct QCBuiltin;

impl<GlobalAddr, FieldAddr> qcvm::userdata::Context for BevyScriptContext<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address,
    FieldAddr: Address,
{
    type Entity = BevyEntityHandle<GlobalAddr, FieldAddr>;
    type Function = BevyBuiltin<GlobalAddr, FieldAddr>;
    type Error = InteropError;
    type GlobalAddr = GlobalAddr;

    fn builtin(
        &self,
        def: &qcvm::BuiltinDef,
    ) -> Result<std::sync::Arc<Self::Function>, Self::Error> {
        // TODO: We could do this just once when initialising the context, which would be significantly more efficient (especially
        //       when we need to look at both builtin and global namespaces).
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let function_registry = function_registry.read();

        // TODO: This is a wasteful way to do lookup, maybe we should have a `BuiltinAddr` like for globals and fields?
        let def_name = def.name.to_string_lossy().into_owned();

        // TODO: Just in case some progs.dats use renamed builtins, we should have some way of looking up via
        //       index, not just name.
        if let Ok(func) =
            function_registry.get_function(QCBuiltin::into_namespace(), def_name.clone())
        {
            Ok(Arc::new(BevyBuiltin {
                func: func.clone(),
                _phantom: PhantomData,
            }))
        } else if let Ok(func) = function_registry.get_function(Namespace::Global, def_name) {
            Ok(Arc::new(BevyBuiltin {
                func: func.clone(),
                _phantom: PhantomData,
            }))
        } else {
            Err(InteropError::NotImplemented)
        }
    }

    fn global(&self, addr: Self::GlobalAddr) -> Result<qcvm::Value, AddrError<InteropError>> {
        let (root_addr, field_addr) = addr.vector_field_or_scalar();
        if let Some(val) = self.special_args.get(&root_addr.to_u16()) {
            return Ok(val
                .field(field_addr)
                .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?);
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
                |world| -> Result<_, AddrError<InteropError>> {
                    let key = ScriptValue::String(addr.name().into());

                    Ok(interop_error_to_addr_error(
                        get(
                            FUNCTION_CALL_CONTEXT,
                            ReflectReference::new_component_ref::<QCWorldspawn>(
                                self.worldspawn,
                                world.clone(),
                            )?,
                            key,
                        )
                        .and_then(|v| bms_script_value_to_qcvm_script_value(&v)),
                    )?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
    }

    fn set_global(
        &mut self,
        addr: Self::GlobalAddr,
        value: qcvm::Value,
    ) -> Result<(), AddrError<InteropError>> {
        let (root_addr, field_addr) = addr.vector_field_or_scalar();
        if let Some(val) = self.special_args.get_mut(&root_addr.to_u16()) {
            if SPECIAL_ARGS_ARE_IMMUTABLE {
                let arg_name = addr.name();
                return Err(AddrError::Other {
                    error: qc_interop_error(anyhow::format_err!(
                        "Special argument {arg_name} is immutable"
                    )),
                });
            } else {
                val.set(field_addr, value)
                    .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?;
                return Ok(());
            }
        }

        // TODO: We could do this just once when initialising the context, which would be significantly more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let set = function_registry.read().magic_functions.set;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world
            .with_read_access(
                ReflectAccessId::for_global(),
                |world| -> Result<_, AddrError<InteropError>> {
                    let key = ScriptValue::String(addr.name().into());

                    Ok(interop_error_to_addr_error(set(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCWorldspawn>(
                            self.worldspawn,
                            world.clone(),
                        )?,
                        key,
                        qcvm_script_value_to_bms_script_value(&value)?,
                    ))?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
    }
}

#[derive(Debug)]
struct BevyScriptArgs<'a>(&'a [ScriptValue]);

struct BevyBuiltin<GlobalAddr, FieldAddr> {
    func: DynamicScriptFunction,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

impl<G, F> PartialEq for BevyBuiltin<G, F> {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl<G, F> fmt::Debug for BevyBuiltin<G, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BevyBuiltin").field(&self.func).finish()
    }
}

impl<G, F> QCType for BevyBuiltin<G, F> {
    fn type_(&self) -> qcvm::Type {
        qcvm::Type::Function
    }

    fn is_null(&self) -> bool {
        false
    }
}

const FUNCTION_CALL_CONTEXT: FunctionCallContext = FunctionCallContext::new(
    <QCScriptingPlugin<EmptyAddress, EmptyAddress> as IntoScriptPluginParams>::LANGUAGE,
);

impl<GlobalAddr, FieldAddr> qcvm::userdata::Function for BevyBuiltin<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address,
    FieldAddr: Address,
{
    type Context = BevyScriptContext<GlobalAddr, FieldAddr>;
    type Error = InteropError;

    fn signature(&self) -> Result<ArrayVec<qcvm::Type, { qcvm::MAX_ARGS }>, Self::Error> {
        Ok(self
            .func
            .info
            .arg_info
            .iter()
            .map(|arg| {
                // TODO: Do we need to consider `[f32; 3]` or other `Vec3`-like types?
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
        self.func
            .call(args, FUNCTION_CALL_CONTEXT)
            .and_then(|val| bms_script_value_to_qcvm_script_value(&val))
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
        }) => Ok(qcvm::Value::Entity(qcvm::EntityRef::new(
            // The addresses do not matter for this usecase.
            //
            // TODO: Maybe we can make this more ergonomic somehow? Maybe remove the `EntityHandle`
            //       trait entirely?
            BevyEntityHandle::<EmptyAddress, EmptyAddress> {
                id: *ent,
                _phantom: PhantomData,
            },
        ))),
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

impl<GlobalAddr, FieldAddr> IntoScriptPluginParams for QCScriptingPlugin<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address + 'static,
    FieldAddr: Address + 'static,
    Self: GetPluginThreadConfig<Self>,
{
    const LANGUAGE: Language = Language::External {
        name: Cow::Borrowed("QC"),
        one_indexed: false,
    };

    type C = qcvm::QCVm;
    type R = ();

    fn build_runtime() -> Self::R {}

    fn handler() -> bevy_mod_scripting_core::handler::HandlerFn<Self> {
        fn qc_callback_handler<GlobalAddr, FieldAddr>(
            mut args: Vec<ScriptValue>,
            context_key: &ScriptAttachment,
            callback: &CallbackLabel,
            context: &mut qcvm::QCVm,
            _world_id: WorldId,
        ) -> Result<ScriptValue, InteropError>
        where
            GlobalAddr: Address,
            FieldAddr: Address,
        {
            let worldspawn = match context_key {
                ScriptAttachment::EntityScript(entity, _) => *entity,
                ScriptAttachment::StaticScript(_) => {
                    return Err(qc_interop_error(anyhow::format_err!(
                        "progs.dat must be attached to a world entity"
                    )));
                }
            };

            let (special_args, args_ref) = match args.get_mut(0) {
                // QuakeC does not support structs, so if any parameter is a struct then we can treat it as
                // special args. We only support passing special args in the first arg slot though, for
                // consistency.
                Some(ScriptValue::Map(map)) => (
                    map.drain()
                        .map(|(k, v)| {
                            let value = bms_script_value_to_qcvm_script_value(&v)?;
                            let key = GlobalAddr::from_name(&k).ok_or_else(|| {
                                let reason = format!("No global with name {k}");
                                InteropError::InvalidIndex {
                                    index: Box::new(k.into()),
                                    reason: Box::new(reason),
                                }
                            })?;

                            Ok((key.to_u16(), value))
                        })
                        .collect::<Result<_, InteropError>>()?,
                    &args[1..],
                ),

                _ => (Default::default(), &args[..]),
            };

            let callback_cstr = CString::from_str(callback.as_ref()).map_err(|_| {
                InteropError::Invariant(Box::new(
                    "Callback name string contained internal NUL".to_string(),
                ))
            })?;

            let val = context
                .run(
                    // TODO: Implement special args (should not be part of callargs)
                    &mut BevyScriptContext::<GlobalAddr, FieldAddr> {
                        worldspawn,
                        special_args,
                        _phantom: PhantomData,
                    },
                    callback_cstr,
                    BevyScriptArgs(args_ref),
                )
                .map_err(qc_interop_error)?;

            qcvm_script_value_to_bms_script_value(&val)
        }

        qc_callback_handler::<GlobalAddr, FieldAddr>
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
        // TODO: We should set all globals on the script attachment when loading.
        fn qc_context_loader(
            _context_key: &ScriptAttachment,
            content: &[u8],
            _world_id: WorldId,
        ) -> Result<qcvm::QCVm, InteropError> {
            qcvm::QCVm::load(std::io::Cursor::new(content)).map_err(qc_interop_error)
        }

        qc_context_loader
    }

    fn context_reloader() -> bevy_mod_scripting_core::context::ContextReloadFn<Self> {
        // TODO: We should diff the `progs.dat` global values between the old and new contexts and set globals
        //       on the script attachment.
        fn qc_context_reloader(
            _context_key: &ScriptAttachment,
            content: &[u8],
            previous_context: &mut qcvm::QCVm,
            _world_id: WorldId,
        ) -> Result<(), InteropError> {
            *previous_context =
                qcvm::QCVm::load(std::io::Cursor::new(content)).map_err(qc_interop_error)?;

            Ok(())
        }

        qc_context_reloader
    }
}

mod config_impl_empty {
    use bevy_mod_scripting_core::{
        config::{GetPluginThreadConfig, ScriptingPluginConfiguration},
        make_plugin_config_static,
    };
    use qcvm::EmptyAddress;

    use crate::QCScriptingPlugin;

    make_plugin_config_static!(QCScriptingPlugin<EmptyAddress, EmptyAddress>);
}

#[cfg(feature = "quake1")]
mod config_impl_quake1 {
    use bevy_mod_scripting_core::{
        config::{GetPluginThreadConfig, ScriptingPluginConfiguration},
        make_plugin_config_static,
    };

    use crate::QCScriptingPlugin;

    make_plugin_config_static!(QCScriptingPlugin<qcvm::quake1::globals::GlobalAddr, qcvm::quake1::fields::FieldAddr>);
}

impl<GlobalAddr, FieldAddr> AsMut<ScriptingPlugin<Self>>
    for QCScriptingPlugin<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address + 'static,
    FieldAddr: Address + 'static,
    Self: GetPluginThreadConfig<Self>,
{
    fn as_mut(&mut self) -> &mut ScriptingPlugin<Self> {
        &mut self.scripting_plugin
    }
}
