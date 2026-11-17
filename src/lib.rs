//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate. This is just bindings for `bevy_mod_scripting`.
//!
//! To add builtins to the engine, register callbacks with `NamespaceBuilder::<QCWorldspawn>`. Fields are
//! accessed using the `get` and `set` of `MagicFunctions`, where globals are accessed using
//! `QCWorldspawn` reflect references and fields are accessed using `QCEntity`. In the rare
//! case that fields need to be accessed on worldspawn (which, unfortunately, is allowed by
//! Quake), `QCEntity` is still used.
//!
//! Values are passed to and from the magic functions and builtins as follows:
//!
//! - Floats are [`ScriptValue::Float`].
//! - Vectors are a [`ScriptValue::List`] of three floats.
//! - Entities are references to their [`QCEntity`] component (including worldspawn).
//! - Strings are references to a [`QCString`], which keeps QuakeC's (not necessarily UTF-8) bytes
//!   and distinguishes the null string from other empty strings.
//! - Functions defined by the `progs.dat` are references to a [`QCFunction`].
//! - Field references are references to a [`QCField`].

#![deny(missing_docs)]

use std::{
    any::{Any, TypeId},
    borrow::Cow,
    collections::VecDeque,
    ffi::{CStr, CString},
    fmt,
    marker::PhantomData,
    num::NonZero,
    str::FromStr as _,
    sync::{Arc, RwLock},
};

use bevy_ecs::{
    component::Component, entity::Entity, resource::Resource, world::World, world::WorldId,
};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    CurrentScriptAttachment, DynamicScriptFunction, ExternalError, FunctionCallContext,
    InteropError, ReflectBase, ReflectReference, ScriptValue, ThreadScriptContext,
    ThreadWorldContainer, VariadicTuple, WorldAccessGuard, WorldExtensions, WorldGuard,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin, config::GetPluginThreadConfig, event::CallbackLabel,
};
use bevy_mod_scripting_script::ScriptAttachment;
use bevy_reflect::{PartialReflect, Reflect, std_traits::ReflectDefault};
use qcvm::{
    Address, ArgError, EntityRef, ErasedEntityHandle, FieldDef, FunctionIndex, FunctionRef,
    QCParams, VectorField, anyhow,
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
    /// The options that `progs.dat`s are loaded with (and reloaded with). The defaults are
    /// strict; see [`qcvm::VmOptions`] for the Quake behaviours that can be enabled to run buggy
    /// QuakeC.
    pub vm_options: qcvm::VmOptions,
}

/// The [`qcvm::VmOptions`] that QuakeC contexts are loaded with, inserted by
/// [`QCScriptingPlugin`] from [`QCScriptingPlugin::vm_options`].
#[derive(Resource, Debug, Clone, Default)]
pub struct QCVmOptions(pub qcvm::VmOptions);

/// The [`QCVmOptions`] of the world that a context is being loaded in.
fn vm_options() -> Result<qcvm::VmOptions, InteropError> {
    thread_context()?
        .world
        .with_resource::<QCVmOptions, _, _>(|options| options.0.clone())
}

/// A QuakeC string.
///
/// QuakeC strings are bytes rather than UTF-8 (Quake uses the high bit for "gold" text), and the
/// null string (which unset fields hold) is distinct from other empty strings. The default value is
/// the null string.
#[derive(Reflect, Clone, PartialEq, Eq, Hash)]
#[reflect(opaque, Clone, Debug, PartialEq, Hash, Default)]
pub struct QCString(pub Arc<CStr>);

impl QCString {
    /// The null string.
    pub fn null() -> Self {
        Self(qcvm::null_string())
    }

    /// Whether this is the null string. Note that other strings can also be empty.
    pub fn is_null(&self) -> bool {
        qcvm::is_null_string(&self.0)
    }

    /// The bytes of the string, without the terminator.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.to_bytes()
    }

    /// Creates a (non-null) string from bytes, which are truncated at the first NUL.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        Self(
            CString::new(&bytes[..len])
                .expect("Interior NULs were removed")
                .into(),
        )
    }

    /// The string as UTF-8, replacing invalid sequences.
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.0.to_string_lossy()
    }
}

impl Default for QCString {
    fn default() -> Self {
        Self::null()
    }
}

impl fmt::Debug for QCString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            f.write_str("null")
        } else {
            write!(f, "\"{}\"", self.as_bytes().escape_ascii())
        }
    }
}

impl From<&str> for QCString {
    fn from(value: &str) -> Self {
        Self::from_bytes(value.as_bytes())
    }
}

impl From<Arc<CStr>> for QCString {
    fn from(value: Arc<CStr>) -> Self {
        Self(value)
    }
}

/// A function defined by the `progs.dat` (either in QuakeC or as a builtin), by its index in the
/// function table. Index 0 is the null function.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[reflect(Clone, Debug, PartialEq, Hash, Default)]
pub struct QCFunction(pub i32);

impl QCFunction {
    /// The null function.
    pub const NULL: Self = Self(0);

    /// Whether this is the null function.
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

/// A reference to an entity field (QuakeC's `.float` etc.), by its offset (see
/// [`qcvm::FieldRef`]). This is e.g. the field argument of Quake's `find`.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[reflect(Clone, Debug, PartialEq, Hash, Default)]
pub struct QCField(pub u16);

impl From<QCField> for qcvm::FieldRef {
    fn from(value: QCField) -> Self {
        Self { offset: value.0 }
    }
}

impl From<qcvm::FieldRef> for QCField {
    fn from(value: qcvm::FieldRef) -> Self {
        Self(value.offset)
    }
}

impl From<QCFunction> for FunctionRef {
    fn from(value: QCFunction) -> Self {
        FunctionRef::Index(value.0)
    }
}

#[derive(Debug)]
struct BevyScriptContext<GlobalAddr, FieldAddr> {
    /// Per-execution globals such as `self` and `other`
    special_args: hashbrown::HashMap<u16, qcvm::Value>,
    context: QCVmContext,
    _phantom: PhantomData<(GlobalAddr, FieldAddr)>,
}

impl<G, F> BevyScriptContext<G, F> {
    fn new(context: QCVmContext) -> Self {
        Self {
            special_args: Default::default(),
            context,
            _phantom: PhantomData,
        }
    }
}

#[derive(PartialEq, Eq)]
struct BevyEntityHandle<GlobalAddr, FieldAddr> {
    /// `None` is worldspawn, which is resolved using the context.
    id: Option<Entity>,
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
        self.id.is_none()
    }
}

/// Marker component for individual entities controlled by QC
#[derive(Component, Reflect, Default)]
pub struct QCEntity;

/// Marker component for the worldspawn entity, where globals will be stored
#[derive(Component, Reflect, Default)]
pub struct QCWorldspawn;

impl<G, F> BevyEntityHandle<G, F> {
    fn entity(&self, context: &QCVmContext) -> Entity {
        self.id.unwrap_or(context.worldspawn)
    }
}

fn magic_get<C: Component>(entity: Entity, key: ScriptValue) -> Result<ScriptValue, InteropError> {
    let world = thread_context()?.world;
    let get = world.script_function_registry().read().magic_functions.get;

    get(
        FUNCTION_CALL_CONTEXT,
        ReflectReference::new_component_ref::<C>(entity, world.clone())?,
        key,
    )
}

fn magic_set<C: Component>(
    entity: Entity,
    key: ScriptValue,
    value: ScriptValue,
) -> Result<(), InteropError> {
    let world = thread_context()?.world;
    let set = world.script_function_registry().read().magic_functions.set;

    set(
        FUNCTION_CALL_CONTEXT,
        ReflectReference::new_component_ref::<C>(entity, world.clone())?,
        key,
        value,
    )
}

/// The key used to access a field defined by the `progs.dat` (see [`EntityHandle::get_dynamic`]):
/// its name, or `{name}_x` etc. for the components of vector fields, as `qcc` names them.
fn dynamic_field_key(def: &FieldDef, component: Option<VectorField>) -> ScriptValue {
    let name = def.name.to_string_lossy();
    ScriptValue::String(match component {
        Some(component) => format!("{name}_{component}").into(),
        None => name.into_owned().into(),
    })
}

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
        let key = ScriptValue::String(field.name().into());
        let value =
            interop_error_to_addr_error(magic_get::<QCEntity>(self.entity(&ctx.context), key))?;

        Ok(bms_to_qcvm::<GlobalAddr, FieldAddr>(&value, &ctx.context)?)
    }

    fn set(
        &self,
        ctx: &mut Self::Context,
        field: Self::FieldAddr,
        value: qcvm::Value,
    ) -> Result<(), AddrError<Self::Error>> {
        let key = ScriptValue::String(field.name().into());
        let value = qcvm_to_bms::<GlobalAddr, FieldAddr>(&value, &ctx.context)?;

        interop_error_to_addr_error(magic_set::<QCEntity>(self.entity(&ctx.context), key, value))
    }

    fn get_dynamic(
        &self,
        ctx: &Self::Context,
        def: &FieldDef,
        component: Option<VectorField>,
    ) -> Result<qcvm::Value, AddrError<Self::Error>> {
        let value = interop_error_to_addr_error(magic_get::<QCEntity>(
            self.entity(&ctx.context),
            dynamic_field_key(def, component),
        ))?;

        Ok(bms_to_qcvm::<GlobalAddr, FieldAddr>(&value, &ctx.context)?)
    }

    fn set_dynamic(
        &self,
        ctx: &mut Self::Context,
        def: &FieldDef,
        component: Option<VectorField>,
        value: qcvm::Value,
    ) -> Result<(), AddrError<Self::Error>> {
        let value = qcvm_to_bms::<GlobalAddr, FieldAddr>(&value, &ctx.context)?;

        interop_error_to_addr_error(magic_set::<QCEntity>(
            self.entity(&ctx.context),
            dynamic_field_key(def, component),
            value,
        ))
    }

    fn from_erased_mut<F, O>(erased: EntityRef, callback: F) -> Result<O, Self::Error>
    where
        F: FnOnce(&mut Self) -> O,
    {
        Ok(callback(&mut Self {
            id: erased.map(|erased| Entity::from_bits(erased.get())),
            _phantom: PhantomData,
        }))
    }

    fn to_erased(&self) -> EntityRef {
        self.id.map(entity_to_erased)
    }
}

fn entity_to_erased(entity: Entity) -> ErasedEntityHandle {
    NonZero::new(entity.to_bits()).expect(
        "Sanity check failed: entity references with all-zero bits \
         are unsupported as this is used a sentinel for worldspawn",
    )
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

    fn builtin(&self, def: &qcvm::BuiltinDef) -> Result<Arc<Self::Function>, Self::Error> {
        let cached = self
            .context
            .builtins
            .read()
            .ok()
            .and_then(|builtins| builtins.get(&def.index).cloned())
            .and_then(|builtin| Arc::downcast::<Self::Function>(builtin).ok());
        if let Some(builtin) = cached {
            return Ok(builtin);
        }

        let world = thread_context()?.world;

        let function_registry = world.script_function_registry();
        let function_registry = function_registry.read();

        let def_name = def.name.to_string_lossy();

        // TODO: Just in case some progs.dats use renamed builtins, we should have some way of looking up via
        //       index, not just name.
        let func = function_registry
            .get_function(TypeId::of::<QCWorldspawn>(), def_name.clone())
            .map_err(|_| {
                InteropError::external_boxed(format!("No builtin named {def_name}").into())
            })?;

        let builtin = Arc::new(BevyBuiltin {
            func: func.clone(),
            _phantom: PhantomData,
        });
        if let Ok(mut builtins) = self.context.builtins.write() {
            builtins.insert(def.index, builtin.clone());
        }

        Ok(builtin)
    }

    fn global(&self, addr: Self::GlobalAddr) -> Result<qcvm::Value, AddrError<InteropError>> {
        let (root_addr, field_addr) = addr.vector_field_or_scalar();
        if let Some(val) = self.special_args.get(&root_addr.to_u16()) {
            return Ok(val
                .field(field_addr)
                .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?);
        }

        let key = ScriptValue::String(addr.name().into());
        let value =
            interop_error_to_addr_error(magic_get::<QCWorldspawn>(self.context.worldspawn, key))?;

        interop_error_to_addr_error(bms_to_qcvm::<GlobalAddr, FieldAddr>(&value, &self.context))
    }

    fn set_global(
        &mut self,
        addr: Self::GlobalAddr,
        value: qcvm::Value,
    ) -> Result<(), AddrError<InteropError>> {
        let (root_addr, field_addr) = addr.vector_field_or_scalar();
        if let Some(val) = self.special_args.get_mut(&root_addr.to_u16()) {
            if self.context.special_args_immutable {
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

        let key = ScriptValue::String(addr.name().into());
        let value = qcvm_to_bms::<GlobalAddr, FieldAddr>(&value, &self.context)?;

        interop_error_to_addr_error(magic_set::<QCWorldspawn>(
            self.context.worldspawn,
            key,
            value,
        ))
    }

    /// Implements `OP_STATE` in terms of the `self` and `time` globals and the `nextthink`, `frame`
    /// and `think` fields, as Quake does.
    fn state(&mut self, frame: f32, think_fn: qcvm::Value) -> anyhow::Result<()> {
        let addr = |name: &str| {
            GlobalAddr::from_name(name).ok_or_else(|| anyhow::format_err!("No global named {name}"))
        };
        let field = |name: &str| {
            FieldAddr::from_name(name).ok_or_else(|| anyhow::format_err!("No field named {name}"))
        };

        let get_global = |this: &Self, name: &str| -> anyhow::Result<qcvm::Value> {
            this.global(addr(name)?)
                .map_err(|e| anyhow::format_err!("Failed to get {name}: {e}"))
        };

        let qcvm::Value::Entity(self_) = get_global(self, "self")? else {
            anyhow::bail!("`self` is not an entity");
        };
        let time = match get_global(self, "time")? {
            qcvm::Value::Float(time) => time,
            qcvm::Value::Void => 0.,
            other => anyhow::bail!("`time` is not a float: {other:?}"),
        };

        let handle = BevyEntityHandle::<GlobalAddr, FieldAddr> {
            id: self_.map(|erased| Entity::from_bits(erased.get())),
            _phantom: PhantomData,
        };

        for (name, value) in [
            ("nextthink", qcvm::Value::Float(time + 0.1)),
            ("frame", qcvm::Value::Float(frame)),
            ("think", think_fn),
        ] {
            handle
                .set(self, field(name)?, value)
                .map_err(|e| anyhow::format_err!("Failed to set {name}: {e}"))?;
        }

        Ok(())
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
        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_to_qcvm::<G, F>(arg, &self.context)?)
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
        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_to_qcvm::<G, F>(arg, &self.context)?)
    }

    fn count(&self) -> usize {
        self.args.len()
    }
}

const FUNCTION_CALL_CONTEXT: FunctionCallContext = FunctionCallContext::new(
    <QCScriptingPlugin<qcvm::EmptyAddress, qcvm::EmptyAddress> as IntoScriptPluginParams>::LANGUAGE,
);

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
            .filter(|arg| arg.type_id != TypeId::of::<FunctionCallContext>())
            .take(qcvm::MAX_ARGS)
            .map(|arg| {
                if arg.type_id == TypeId::of::<Vec3>() || arg.type_id == TypeId::of::<[f32; 3]>() {
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
        let args = context
            .arguments(&sig)
            .map(|val| {
                let val = val.map_err(qc_interop_error)?;
                qcvm_to_bms::<GlobalAddr, FieldAddr>(&val, &context.context)
            })
            .collect::<Result<ArrayVec<_, { qcvm::MAX_ARGS }>, _>>()?;

        let result = self.func.call(args, FUNCTION_CALL_CONTEXT)?;
        bms_to_qcvm::<GlobalAddr, FieldAddr>(&result, &context.context)
    }
}

/// The default [`ReflectToValue`], which handles [`QCString`], [`QCFunction`], [`QCField`],
/// [`Vec3`], `[f32; 3]`, `f32`, `f64` and `bool`.
pub fn default_reflect_to_value(value: &dyn PartialReflect) -> Result<qcvm::Value, InteropError> {
    if let Some(s) = value.try_downcast_ref::<QCString>() {
        Ok(qcvm::Value::String(s.0.clone()))
    } else if let Some(f) = value.try_downcast_ref::<QCFunction>() {
        Ok(FunctionIndex(f.0).to_value())
    } else if let Some(f) = value.try_downcast_ref::<QCField>() {
        Ok(qcvm::Value::Field((*f).into()))
    } else if let Some(v) = value.try_downcast_ref::<Vec3>() {
        Ok(qcvm::Value::Vector(v.to_array().into()))
    } else if let Some(v) = value.try_downcast_ref::<[f32; 3]>() {
        Ok(qcvm::Value::Vector((*v).into()))
    } else if let Some(f) = value.try_downcast_ref::<f32>() {
        Ok(qcvm::Value::Float(*f))
    } else if let Some(f) = value.try_downcast_ref::<f64>() {
        Ok(qcvm::Value::Float(*f as f32))
    } else if let Some(b) = value.try_downcast_ref::<bool>() {
        Ok((*b).into())
    } else {
        Err(InteropError::external_boxed(
            format!(
                "Can't convert a value of type {} to QuakeC",
                value
                    .get_represented_type_info()
                    .map(|info| info.type_path())
                    .unwrap_or("<unknown>")
            )
            .into(),
        ))
    }
}

fn bms_to_qcvm<G, F>(
    bms_value: &ScriptValue,
    context: &QCVmContext,
) -> Result<qcvm::Value, InteropError>
where
    G: Address,
    F: Address,
{
    let as_f32 = |value: &ScriptValue| match *value {
        ScriptValue::Float(f) => Some(f as f32),
        ScriptValue::Integer(i) => Some(i as f32),
        _ => None,
    };

    match bms_value {
        ScriptValue::Unit => Ok(qcvm::Value::Void),
        &ScriptValue::Bool(b) => Ok(b.into()),
        &ScriptValue::Integer(i) => Ok(i.into()),
        &ScriptValue::Float(f) => Ok(f.into()),
        ScriptValue::String(cow) => Ok(qcvm::Value::String(
            CString::from_str(cow)
                .map_err(|e| InteropError::External(ExternalError(Arc::new(e))))?
                .into(),
        )),
        ScriptValue::List(script_values) | ScriptValue::Tuple(VariadicTuple(script_values)) => {
            match script_values
                .iter()
                .map(as_f32)
                .collect::<Option<ArrayVec<f32, 3>>>()
            {
                Some(values) if values.len() == 3 => Ok(qcvm::Value::Vector(
                    [values[0], values[1], values[2]].into(),
                )),
                _ => Err(InteropError::external_boxed(
                    "Only lists of 3 numbers (i.e. vectors) can be passed to QuakeC".into(),
                )),
            }
        }
        ScriptValue::Reference(r)
            if let ReflectBase::Component(ent, _) = r.base.base_id()
                && r.base.type_id() == TypeId::of::<QCEntity>() =>
        {
            Ok(qcvm::Value::Entity(if *ent == context.worldspawn {
                None
            } else {
                Some(entity_to_erased(*ent))
            }))
        }
        ScriptValue::Reference(r)
            if matches!(r.base.base_id(), ReflectBase::Component(..))
                && r.base.type_id() == TypeId::of::<QCWorldspawn>() =>
        {
            Err(InteropError::external_boxed(
                "Tried to convert a `QCWorldspawn` reference to a qcvm entity. If this is what \
                you intended, add a `QCEntity` component to the worldspawn entity and pass a \
                `ReflectReference` to that component instead"
                    .into(),
            ))
        }
        ScriptValue::Reference(val) => {
            let world = thread_context()?.world;
            val.with_reflect(world, context.reflect_to_value)?
        }
        ScriptValue::Function(func) => Ok(qcvm::Value::Function(Arc::new(BevyBuiltin::<G, F> {
            func: func.clone(),
            _phantom: PhantomData,
        }))),
        ScriptValue::Error(e) => Err(e.clone()),
        _ => Err(InteropError::NotImplemented),
    }
}

fn qcvm_to_bms<G, F>(
    qcvm_value: &qcvm::Value,
    root_ctx: &QCVmContext,
) -> Result<ScriptValue, InteropError>
where
    G: Address,
    F: Address,
{
    match qcvm_value {
        qcvm::Value::Void => Ok(ScriptValue::Unit),
        qcvm::Value::Entity(entity) => {
            let entity = entity
                .map(|bits| Entity::from_bits(bits.get()))
                .unwrap_or(root_ctx.worldspawn);
            let world = thread_context()?.world;

            Ok(ScriptValue::Reference(
                ReflectReference::new_component_ref::<QCEntity>(entity, world)?,
            ))
        }
        qcvm::Value::Function(function) => {
            let function: &dyn Any = &**function;
            if let Some(index) = function.downcast_ref::<FunctionIndex>() {
                Ok(ScriptValue::Reference(ReflectReference::new_allocated(
                    QCFunction(index.0),
                )))
            } else if let Some(builtin) = function.downcast_ref::<BevyBuiltin<G, F>>() {
                Ok(ScriptValue::Function(builtin.func.clone()))
            } else {
                Err(InteropError::external_boxed(
                    "Only functions defined by the progs or registered as builtins can be passed \
                     out of QuakeC"
                        .into(),
                ))
            }
        }
        qcvm::Value::Float(f) => Ok(ScriptValue::Float(*f as f64)),
        qcvm::Value::Vector(vec3) => Ok(ScriptValue::List(VecDeque::from([
            ScriptValue::Float(vec3.x as f64),
            ScriptValue::Float(vec3.y as f64),
            ScriptValue::Float(vec3.z as f64),
        ]))),
        qcvm::Value::String(cstr) => Ok(ScriptValue::Reference(ReflectReference::new_allocated(
            QCString(cstr.clone()),
        ))),
        qcvm::Value::Field(field) => Ok(ScriptValue::Reference(ReflectReference::new_allocated(
            QCField::from(*field),
        ))),
    }
}

/// Fetches the current thread's world context, converting the error to an `InteropError`
fn thread_context<'a>() -> Result<ThreadScriptContext<'a>, InteropError> {
    Ok(ThreadWorldContainer::try_get_context()?)
}

fn qc_interop_error(e: qcvm::Error) -> InteropError {
    // Keep the whole chain of causes (which includes the QuakeC call stack) in the message.
    InteropError::external_boxed(format!("{e:#}").into())
}

/// Builtins looked up by function index, so that each call doesn't need to search by name.
type BuiltinCache = Arc<RwLock<hashbrown::HashMap<i32, Arc<dyn Any + Send + Sync>>>>;

/// Wrapper around `qcvm::QCVm` that adds extra configuration for converting values
#[derive(Clone)]
pub struct QCVmContext {
    /// The inner VM
    pub vm: Arc<qcvm::QCVm>,
    /// A conversion function for arbitrary reflect values. By default, this is
    /// [`default_reflect_to_value`].
    pub reflect_to_value: ReflectToValue,
    /// The worldspawn entity, which holds the globals and which QuakeC refers to as `world`.
    pub worldspawn: Entity,
    /// If `true`, special arguments (see the callback handler) can't be written by QuakeC.
    pub special_args_immutable: bool,
    builtins: BuiltinCache,
}

impl fmt::Debug for QCVmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QCVmContext")
            .field("vm", &self.vm)
            .field("worldspawn", &self.worldspawn)
            .field("special_args_immutable", &self.special_args_immutable)
            .finish_non_exhaustive()
    }
}

impl QCVmContext {
    /// Creates a context for `vm`, with globals stored on `worldspawn`.
    pub fn new(vm: Arc<qcvm::QCVm>, worldspawn: Entity) -> Self {
        Self {
            vm,
            reflect_to_value: default_reflect_to_value,
            worldspawn,
            special_args_immutable: false,
            builtins: Default::default(),
        }
    }

    /// Calls a QuakeC function with exclusive access to `world`.
    ///
    /// This can be used both from regular systems and from inside builtins (e.g. to run touch
    /// functions from a movement builtin), as it doesn't go through `bevy_mod_scripting`'s context
    /// handling: it creates a fresh world access guard for `world`, which is used by field and
    /// global accessors and builtins until this returns.
    pub fn call_with_world<G, F>(
        &self,
        world: &mut World,
        function: impl Into<FunctionRef>,
        args: impl QCParams,
    ) -> Result<qcvm::Value, InteropError>
    where
        G: Address,
        F: Address,
    {
        let cache = <WorldGuard<'_> as WorldExtensions>::setup_cache(
            world,
            CurrentScriptAttachment::empty(),
        );
        WorldAccessGuard::with_static_guard(world, cache, |_guard| {
            let mut context = BevyScriptContext::<G, F>::new(self.clone());
            self.vm
                .run(&mut context, function, args)
                .map_err(qc_interop_error)
        })
    }

    /// The global definitions in the `progs.dat`.
    pub fn global_defs(&self) -> &[qcvm::GlobalDef] {
        self.vm.global_defs()
    }

    /// The entity field definitions in the `progs.dat`.
    pub fn field_defs(&self) -> &[FieldDef] {
        self.vm.field_defs()
    }

    /// The initial value of a global (see [`qcvm::QCVm::initial_global`]).
    pub fn initial_global(&self, addr: u16) -> anyhow::Result<qcvm::Value> {
        self.vm.initial_global(addr)
    }
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

            let (special_args, args_ref) = match args.get_mut(0) {
                // QuakeC does not support structs, so if any parameter is a struct then we can treat it as
                // special args. We only support passing special args in the first arg slot though, for
                // consistency.
                Some(ScriptValue::Map(map)) => (
                    map.drain()
                        .map(|(k, v)| {
                            let value = bms_to_qcvm::<G, F>(&v, context)?;
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

            // `bevy_mod_scripting` runs callbacks such as `on_script_loaded`, which progs don't
            // usually define.
            if context.vm.functions().get_by_name(&callback_cstr).is_err() {
                return Ok(ScriptValue::Unit);
            }

            let mut script_context = BevyScriptContext::<G, F>::new(context.clone());
            script_context.special_args = special_args;

            let val = context
                .vm
                .run(
                    &mut script_context,
                    callback_cstr,
                    BevyScriptArgs::<[ScriptValue], G, F> {
                        context: context.clone(),
                        args: args_ref,
                        _phantom: PhantomData,
                    },
                )
                .map_err(qc_interop_error)?;

            qcvm_to_bms::<G, F>(&val, context)
        }

        qc_callback_handler::<GlobalAddr, FieldAddr>
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
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
            let vm = qcvm::QCVm::load_with_options(std::io::Cursor::new(content), vm_options()?)
                .map_err(qc_interop_error)?;
            // Globals are stored on the entity the script is attached to.
            let worldspawn = context_key.entity().unwrap_or(Entity::PLACEHOLDER);
            let mut out = QCVmContext::new(vm.into(), worldspawn);

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
            // Keep the options that the context was loaded with.
            let options = previous_context.vm.options().clone();
            previous_context.vm =
                qcvm::QCVm::load_with_options(std::io::Cursor::new(content), options)
                    .map_err(qc_interop_error)?
                    .into();
            previous_context.builtins = Default::default();

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

impl<GlobalAddr, FieldAddr> bevy_app::Plugin for QCScriptingPlugin<GlobalAddr, FieldAddr>
where
    GlobalAddr: Address + 'static,
    FieldAddr: Address + 'static,
    Self: GetPluginThreadConfig<Self>,
{
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(QCVmOptions(self.vm_options.clone()));
        self.scripting_plugin.build(app);
    }

    fn finish(&self, app: &mut bevy_app::App) {
        self.scripting_plugin.finish(app);
    }
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
