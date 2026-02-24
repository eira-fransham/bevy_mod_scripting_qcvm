//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate. This is just bindings for `bevy_mod_scripting`.
//!
//! To add builtins to the engine, add callbacks to the [`QCWorldspawn`] namespace. Fields are
//! accessed using the `get` and `set` of `MagicFunctions`, where globals are accessed using
//! `QCWorldspawn` reflect references and fields are accessed using `QCEntity`. In the rare
//! case that fields need to be accessed on worldspawn (which, unfortunately, is allowed by
//! Quake), `QCEntity` is still used.

#![deny(missing_docs)]

use std::{
    any::{Any, TypeId},
    borrow::Cow,
    collections::VecDeque,
    ffi::CString,
    fmt,
    marker::PhantomData,
    str::FromStr as _,
    sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, world::WorldId};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    DynamicScriptFunction, ExternalError, FunctionCallContext, InteropError, IntoNamespace,
    ReflectAccessId, ReflectBase, ReflectBaseType, ReflectReference, ScriptValue,
    ThreadScriptContext, ThreadWorldContainer,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin, config::GetPluginThreadConfig, event::CallbackLabel,
};
use bevy_mod_scripting_script::ScriptAttachment;
use bevy_reflect::PartialReflect;
use qcvm::{
    Address, ArgError, EmptyAddress, EntityRef, ErasedEntityHandle, QCParams, anyhow,
    arrayvec::ArrayVec,
    userdata::{AddrError, EntityHandle, QCType},
};

pub use qcvm;

/// A function converting an arbitrary `PartialReflect` to a QuakeC value
pub type ReflectToValue = fn(&dyn PartialReflect) -> Result<qcvm::Value, InteropError>;

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
    /// Per-execution globals such as `self` and `other`
    special_args: hashbrown::HashMap<u16, qcvm::Value>,
    context: QCVmContext,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

#[derive(PartialEq, Eq)]
struct BevyEntityHandle<GlobalAddr, FieldAddr> {
    id: bevy_ecs::entity::Entity,
    /// Used to sanity check check that the ID bits are not all-zero when converting back to u64
    is_worldspawn: bool,
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
        // TODO: Maybe worldspawn should be handled using `BevyEntityHandle` since
        // Quake 1 allows fields to be stored on worldspawn
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
        ctx: &Self::Context,
        field: Self::FieldAddr,
    ) -> Result<qcvm::Value, AddrError<Self::Error>> {
        // TODO: We could do this just once when initialising the context, which may be more efficient
        let world_ctx = ThreadWorldContainer.try_get_context()?;
        let world = &world_ctx.world;

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
            })
            .map_err(|()| InteropError::MissingWorld)?
            .and_then(|v| {
                bms_script_value_to_qcvm_script_value::<GlobalAddr, FieldAddr>(
                    &v,
                    &world_ctx,
                    &ctx.context,
                )
            })
            .map_err(Into::into)
    }

    fn set(
        &self,
        context: &mut Self::Context,
        field: Self::FieldAddr,
        value: qcvm::Value,
    ) -> Result<(), AddrError<Self::Error>> {
        // TODO: We could do this just once when initialising the context, which may be more efficient
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let function_registry = world_ctx.world.script_function_registry();
        let set = function_registry.read().magic_functions.set;

        let new_value = qcvm_script_value_to_bms_script_value::<GlobalAddr, FieldAddr>(
            &value,
            &world_ctx,
            &context.context,
        )?;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world_ctx
            .world
            .with_read_access(ReflectAccessId::for_global(), |world| {
                let key = ScriptValue::String(field.name().into());

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

    fn from_erased_mut<F, O>(erased: EntityRef, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&mut Self) -> O,
    {
        match erased {
            Some(erased) => {
                let id = bevy_ecs::entity::Entity::from_bits(erased.get());

                Ok(callback(&mut Self {
                    id,
                    is_worldspawn: false,
                    _phantom: PhantomData,
                }))
            }
            None => {
                let worldspawn = ThreadWorldContainer
                    .try_get_context()?
                    .attachment
                    .entity()
                    .ok_or(InteropError::MissingWorld)?;

                Ok(callback(&mut Self {
                    id: worldspawn,
                    is_worldspawn: true,
                    _phantom: PhantomData,
                }))
            }
        }
    }

    fn to_erased(&self) -> EntityRef {
        if self.is_worldspawn {
            None
        } else {
            Some(ErasedEntityHandle::new(self.id.to_bits()).expect(
                "Sanity check failed: entity references with all-zero bits \
                 are unsupported as this is used a sentinel for worldspawn",
            ))
        }
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
        // TODO: We could do this just once when initialising the context, which may be more efficient
        let world = ThreadWorldContainer.try_get_context()?.world;

        let function_registry = world.script_function_registry();
        let function_registry = function_registry.read();

        // TODO: This is a wasteful way to do lookup, maybe we should have a `BuiltinAddr` like for globals and fields?
        let def_name = def.name.to_string_lossy().into_owned();

        // TODO: Just in case some progs.dats use renamed builtins, we should have some way of looking up via
        //       index, not just name.
        if let Ok(func) =
            function_registry.get_function(QCWorldspawn::into_namespace(), def_name.clone())
        {
            Ok(Arc::new(BevyBuiltin {
                func: func.clone(),
                context: self.context.clone(),
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
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let function_registry = world_ctx.world.script_function_registry();
        let get = function_registry.read().magic_functions.get;

        let worldspawn = world_ctx
            .attachment
            .entity()
            .ok_or(InteropError::MissingWorld)?;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world_ctx
            .world
            .with_read_access(
                ReflectAccessId::for_global(),
                |world| -> Result<_, AddrError<InteropError>> {
                    let key = ScriptValue::String(addr.name().into());

                    Ok(interop_error_to_addr_error(get(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCWorldspawn>(
                            worldspawn,
                            world.clone(),
                        )?,
                        key,
                    ))?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
            .and_then(|v| {
                interop_error_to_addr_error(bms_script_value_to_qcvm_script_value::<
                    GlobalAddr,
                    FieldAddr,
                >(&v, &world_ctx, &self.context))
            })
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

        // TODO: We could do this just once when initialising the context, which may be more efficient
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let function_registry = world_ctx.world.script_function_registry();
        let set = function_registry.read().magic_functions.set;

        let worldspawn = world_ctx
            .attachment
            .entity()
            .ok_or(InteropError::MissingWorld)?;

        let new_value = qcvm_script_value_to_bms_script_value::<GlobalAddr, FieldAddr>(
            &value,
            &world_ctx,
            &self.context,
        )?;

        // TODO: We unfortunately need global access for now, since we don't know what components a getter will access.
        //       We can add optimised getter/setter configuration later.
        world_ctx
            .world
            .with_write_access(
                ReflectAccessId::for_global(),
                |world| -> Result<_, AddrError<InteropError>> {
                    let key = ScriptValue::String(addr.name().into());

                    Ok(interop_error_to_addr_error(set(
                        FUNCTION_CALL_CONTEXT,
                        ReflectReference::new_component_ref::<QCWorldspawn>(
                            worldspawn,
                            world.clone(),
                        )?,
                        key,
                        new_value,
                    ))?)
                },
            )
            .map_err(|()| InteropError::MissingWorld)?
    }
}

struct BevyScriptArgs<'a, A, GlobalAddr, FieldAddr>
where
    A: ?Sized,
{
    args: &'a A,
    context: QCVmContext,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

impl<A, G, F> fmt::Debug for BevyScriptArgs<'_, A, G, F>
where
    A: ?Sized + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BevyScriptArgs")
            .field("args", &self.args)
            .finish()
    }
}

impl<'a, G, F> QCParams for BevyScriptArgs<'a, [ScriptValue], G, F>
where
    G: Address,
    F: Address,
{
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, ArgError<Self::Error>> {
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_script_value_to_qcvm_script_value::<G, F>(
            arg,
            &world_ctx,
            &self.context,
        )?)
    }

    fn count(&self) -> usize {
        self.args.len()
    }
}

impl<'a, G, F> QCParams for BevyScriptArgs<'a, VecDeque<ScriptValue>, G, F>
where
    G: Address,
    F: Address,
{
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, ArgError<Self::Error>> {
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_script_value_to_qcvm_script_value::<G, F>(
            arg,
            &world_ctx,
            &self.context,
        )?)
    }

    fn count(&self) -> usize {
        self.args.len()
    }
}

const FUNCTION_CALL_CONTEXT: FunctionCallContext = FunctionCallContext::new(
    <QCScriptingPlugin<EmptyAddress, EmptyAddress> as IntoScriptPluginParams>::LANGUAGE,
);

struct BevyBuiltin<GlobalAddr, FieldAddr> {
    func: DynamicScriptFunction,
    context: QCVmContext,
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
        // TODO: We could do this just once when initialising the context, which may be more efficient
        let world_ctx = ThreadWorldContainer.try_get_context()?;

        let sig = self.signature()?;
        let args = context.arguments(&sig);
        let args = args
            .map(|val| {
                qcvm_script_value_to_bms_script_value::<GlobalAddr, FieldAddr>(
                    &val,
                    &world_ctx,
                    &self.context,
                )
            })
            .collect::<Result<ArrayVec<_, { qcvm::MAX_ARGS }>, _>>()?;
        self.func.call(args, FUNCTION_CALL_CONTEXT).and_then(|val| {
            let world = ThreadWorldContainer.try_get_context()?;
            bms_script_value_to_qcvm_script_value::<GlobalAddr, FieldAddr>(
                &val,
                &world,
                &context.context,
            )
        })
    }
}

fn bms_script_value_to_qcvm_script_value<G, F>(
    bms_value: &ScriptValue,
    world_ctx: &ThreadScriptContext<'_>,
    context: &QCVmContext,
) -> Result<qcvm::Value, InteropError>
where
    G: Address,
    F: Address,
{
    let world = &world_ctx.world;
    let worldspawn = world_ctx
        .attachment
        .entity()
        .ok_or(InteropError::MissingWorld)?;
    let qc_entity_id = world.get_component_id(TypeId::of::<QCEntity>())?;
    let qc_worldspawn_id = world.get_component_id(TypeId::of::<QCWorldspawn>())?;

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
                    base_id: ReflectBase::Component(ent, component_id),
                    ..
                },
            ..
        }) if Some(component_id) == qc_entity_id.as_ref() => {
            Ok(qcvm::Value::Entity(
                // The addresses do not matter for this usecase.
                //
                // TODO: Maybe we can make this more ergonomic somehow? Maybe remove the `EntityHandle`
                //       trait entirely?
                BevyEntityHandle::<EmptyAddress, EmptyAddress> {
                    id: *ent,
                    is_worldspawn: *ent == worldspawn,
                    _phantom: PhantomData,
                }
                .to_erased(),
            ))
        }
        ScriptValue::Reference(ReflectReference {
            base:
                ReflectBaseType {
                    base_id: ReflectBase::Component(_, component_id),
                    ..
                },
            ..
        }) if Some(component_id) == qc_worldspawn_id.as_ref() => {
            return Err(InteropError::external_boxed(
                "Tried to convert a `QCWorldspawn` reference to a qcvm entity. If this is what \
                you intended, add a `QCEntity` component to the worldspawn entity and pass a \
                `ReflectReference` to that component instead"
                    .into(),
            ));
        }
        ScriptValue::Reference(val) => val.with_reflect(world.clone(), context.reflect_to_value)?,
        ScriptValue::FunctionMut(_) => todo!(),
        ScriptValue::Function(func) => Ok(qcvm::Value::Function(Arc::new(BevyBuiltin::<G, F> {
            func: func.clone(),
            context: context.clone(),
            _phantom: PhantomData,
        }))),
        ScriptValue::Error(e) => Err(e.clone()),
        _ => Err(InteropError::NotImplemented),
    }
}

fn qcvm_script_value_to_bms_script_value<G, F>(
    qcvm_value: &qcvm::Value,
    world_ctx: &ThreadScriptContext<'_>,
    root_ctx: &QCVmContext,
) -> Result<ScriptValue, InteropError>
where
    G: Address,
    F: Address,
{
    match qcvm_value {
        qcvm::Value::Void => Ok(ScriptValue::Unit),
        qcvm::Value::Entity(Some(bits)) => Ok(ScriptValue::Reference(
            ReflectReference::new_component_ref::<QCEntity>(
                Entity::from_bits(bits.get()),
                world_ctx.world.clone(),
            )?,
        )),
        qcvm::Value::Entity(None) => {
            let worldspawn = world_ctx
                .attachment
                .entity()
                .ok_or(InteropError::MissingWorld)?;

            Ok(ScriptValue::Reference(
                ReflectReference::new_component_ref::<QCEntity>(
                    worldspawn,
                    world_ctx.world.clone(),
                )?,
            ))
        }
        qcvm::Value::Function(function) => {
            if let Some(builtin) = (&**function as &dyn Any).downcast_ref::<BevyBuiltin<G, F>>() {
                Ok(ScriptValue::Function(builtin.func.clone()))
            } else {
                let func = function.clone();
                let root_ctx = root_ctx.clone();
                Ok(ScriptValue::Function(DynamicScriptFunction::from(
                    move |_: FunctionCallContext, args: VecDeque<ScriptValue>| {
                        let inner = || -> Result<ScriptValue, InteropError> {
                            let world_ctx = ThreadWorldContainer.try_get_context()?;
                            let value = root_ctx
                                .vm
                                .run_dynamic(
                                    &mut BevyScriptContext::<G, F> {
                                        special_args: Default::default(),
                                        context: root_ctx.clone(),
                                        _phantom: PhantomData,
                                    },
                                    &*func,
                                    BevyScriptArgs::<VecDeque<ScriptValue>, G, F> {
                                        context: root_ctx.clone(),
                                        args: &args,
                                        _phantom: PhantomData,
                                    },
                                )
                                .map_err(qc_interop_error)?;

                            qcvm_script_value_to_bms_script_value::<G, F>(
                                &value, &world_ctx, &root_ctx,
                            )
                        };

                        inner().unwrap_or_else(ScriptValue::Error)
                    },
                )))
            }
        }
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

fn qc_interop_error(e: qcvm::Error) -> InteropError {
    InteropError::External(ExternalError(e.into_boxed_dyn_error().into()))
}

/// Wrapper around `qcvm::QCVm` that adds extra configuration for converting values
#[derive(Clone, Debug)]
pub struct QCVmContext {
    /// The inner VM
    pub vm: Arc<qcvm::QCVm>,
    /// A conversion function for arbitrary reflect values
    pub reflect_to_value: ReflectToValue,
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

    type C = QCVmContext;
    type R = ();

    fn build_runtime() -> Self::R {}

    fn handler() -> bevy_mod_scripting_core::handler::HandlerFn<Self> {
        fn qc_callback_handler<G, F>(
            mut args: Vec<ScriptValue>,
            context_key: &ScriptAttachment,
            callback: &CallbackLabel,
            context: &mut QCVmContext,
            world_id: WorldId,
        ) -> Result<ScriptValue, InteropError>
        where
            G: Address,
            F: Address,
            QCScriptingPlugin<G, F>: GetPluginThreadConfig<QCScriptingPlugin<G, F>>,
        {
            let config = QCScriptingPlugin::<G, F>::readonly_configuration(world_id);
            for callback in config.pre_handling_callbacks {
                callback(context_key, context)?;
            }

            let world_ctx = ThreadWorldContainer.try_get_context()?;
            let (special_args, args_ref) = match args.get_mut(0) {
                // QuakeC does not support structs, so if any parameter is a struct then we can treat it as
                // special args. We only support passing special args in the first arg slot though, for
                // consistency.
                Some(ScriptValue::Map(map)) => (
                    map.drain()
                        .map(|(k, v)| {
                            let value = bms_script_value_to_qcvm_script_value::<G, F>(
                                &v, &world_ctx, context,
                            )?;
                            let key = G::from_name(&k).ok_or_else(|| {
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
                .vm
                .run(
                    // TODO: Implement special args (should not be part of callargs)
                    &mut BevyScriptContext::<G, F> {
                        special_args,
                        context: context.clone(),
                        _phantom: PhantomData,
                    },
                    callback_cstr,
                    BevyScriptArgs::<[ScriptValue], G, F> {
                        context: context.clone(),
                        args: args_ref,
                        _phantom: PhantomData,
                    },
                )
                .map_err(qc_interop_error)?;

            qcvm_script_value_to_bms_script_value::<G, F>(&val, &world_ctx, context)
        }

        qc_callback_handler::<GlobalAddr, FieldAddr>
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
        // TODO: We should set all globals on the script attachment when loading.
        fn qc_context_loader<G, F>(
            context_key: &ScriptAttachment,
            content: &[u8],
            world_id: WorldId,
        ) -> Result<QCVmContext, InteropError>
        where
            G: Address + 'static,
            F: Address + 'static,
            QCScriptingPlugin<G, F>: GetPluginThreadConfig<QCScriptingPlugin<G, F>>,
        {
            let config = QCScriptingPlugin::<G, F>::readonly_configuration(world_id);
            let mut out = QCVmContext {
                vm: qcvm::QCVm::load(std::io::Cursor::new(content))
                    .map_err(qc_interop_error)?
                    .into(),
                reflect_to_value: |_| Err(InteropError::NotImplemented),
            };

            for init in config.context_initialization_callbacks {
                init(context_key, &mut out)?;
            }

            Ok(out)
        }

        qc_context_loader::<GlobalAddr, FieldAddr>
    }

    fn context_reloader() -> bevy_mod_scripting_core::context::ContextReloadFn<Self> {
        // TODO: We should diff the `progs.dat` global values between the old and new contexts and set globals
        //       on the script attachment.
        fn qc_context_reloader(
            _context_key: &ScriptAttachment,
            content: &[u8],
            previous_context: &mut QCVmContext,
            _world_id: WorldId,
        ) -> Result<(), InteropError> {
            previous_context.vm = qcvm::QCVm::load(std::io::Cursor::new(content))
                .map_err(qc_interop_error)?
                .into();

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
