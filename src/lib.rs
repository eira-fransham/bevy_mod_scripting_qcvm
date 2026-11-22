//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate. This is just bindings for `bevy_mod_scripting`.
//!
//! Several QuakeC VMs can run in the same world (e.g. a server's `progs.dat` and a client's
//! `csprogs.dat`). Each VM is a script attached to its own worldspawn entity, which must have a
//! [`QcVmConfig`] saying which defs (e.g. `qcvm_defs_quake1::Quake1`) the `progs.dat` is compiled
//! against and which namespace its builtins are registered in. Entities belong to a single VM,
//! through the [`QcOwnedBy`] relationship to its worldspawn, and VMs can't access each other's
//! entities.
//!
//! To add builtins to a VM, register callbacks with `NamespaceBuilder` in its
//! [`QcVmConfig::builtin_namespace`]. Builtins whose number is known to the defs (see
//! [`qcvm::ErasedDefs::builtin_name`]) are looked up by their canonical name, and other builtins
//! are looked up by the name in the `progs.dat`. Fields are accessed using the `get` and `set` of
//! `MagicFunctions`, where globals are accessed using `QcWorldspawn` reflect references and fields
//! are accessed using `QcEntity`. In the rare case that fields need to be accessed on worldspawn
//! (which, unfortunately, is allowed by Quake), `QcEntity` is still used. Builtins and magic
//! functions can find the VM that is running with [`current_vm`].
//!
//! Values are passed to and from the magic functions and builtins as follows:
//!
//! - Floats are [`ScriptValue::Float`].
//! - Vectors are a [`ScriptValue::List`] of three floats.
//! - Entities are references to their [`QcEntity`] component (including worldspawn).
//! - Strings are references to a [`QcString`], which keeps QuakeC's (not necessarily UTF-8) bytes
//!   and distinguishes the null string from other empty strings.
//! - Functions defined by the `progs.dat` are references to a [`QcFunction`].
//! - Field references are references to a [`QcField`].

#![deny(missing_docs)]

use std::{
    any::{Any, TypeId},
    borrow::Cow,
    cell::RefCell,
    collections::VecDeque,
    ffi::{CStr, CString},
    fmt,
    num::NonZero,
    str::FromStr as _,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak},
};

use bevy_ecs::{
    component::Component, entity::Entity, storage::SparseSetIndex as _, world::World,
    world::WorldId,
};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    CurrentScriptAttachment, DynamicScriptFunction, ExternalError, FunctionCallContext,
    FunctionInfo, InteropError, ReflectBase, ReflectReference, ScriptValue, ThreadScriptContext,
    ThreadWorldContainer, V, VariadicTuple, WorldAccessGuard, WorldExtensions, WorldGuard,
};
use bevy_mod_scripting_core::{
    IntoScriptPluginParams, ScriptingPlugin,
    config::{GetPluginThreadConfig, ScriptingPluginConfiguration},
    event::CallbackLabel,
};
use bevy_mod_scripting_script::ScriptAttachment;
use bevy_reflect::{PartialReflect, Reflect, std_traits::ReflectDefault};
use qcvm::{
    ArgError, EntityRef, EntityTypeDef, ErasedDefs, ErasedEntityHandle, FieldDef, FunctionIndex,
    FunctionRef, QcParams, Type, VectorField, anyhow,
    arrayvec::ArrayVec,
    userdata::{AddrError, ErasedContext, ErasedFunction, FnCall, QcType},
};

pub use qcvm;

/// A function converting an arbitrary `PartialReflect` to a QuakeC value
pub type ReflectToValue = fn(&dyn PartialReflect) -> Result<qcvm::Value, InteropError>;

/// The language of QuakeC scripts.
const QC_LANGUAGE: Language = Language::External {
    name: Cow::Borrowed("QC"),
    one_indexed: false,
};

/// The main "entry point" plugin for adding QC Scripting.
///
/// The configuration of each VM (its defs, builtins and [`qcvm::VmOptions`]) is the
/// [`QcVmConfig`] of the entity its `progs.dat` is attached to.
pub struct QcScriptingPlugin {
    /// The inner config for the plugin
    pub scripting_plugin: ScriptingPlugin<QcScriptingPlugin>,
}

impl Default for QcScriptingPlugin {
    /// Loads `.dat` files as QuakeC, with a separate context (i.e. VM) for each entity that a
    /// `progs.dat` is attached to.
    fn default() -> Self {
        Self {
            scripting_plugin: ScriptingPlugin {
                runtime_initializers: vec![],
                context_policy: bevy_mod_scripting_core::script::ContextPolicy::per_entity(),
                language: QC_LANGUAGE,
                supported_extensions: vec!["dat"],
                context_initializers: vec![],
                context_pre_handling_initializers: vec![],
                emit_responses: false,
                processing_pipeline_plugin: Default::default(),
            },
        }
    }
}

/// The configuration of a QuakeC VM, which must be on the worldspawn entity that a `progs.dat` is
/// attached to before it is loaded.
#[derive(Component, Clone)]
pub struct QcVmConfig {
    /// The defs that the `progs.dat` is compiled against, e.g. `qcvm_defs_quake1::Quake1`.
    /// `progs.dat`s with a different CRC fail to load.
    pub defs: &'static dyn ErasedDefs,
    /// The namespace (as for `NamespaceBuilder`, i.e. the [`TypeId`] of a type) that builtins are
    /// looked up in.
    pub builtin_namespace: TypeId,
    /// The options that the `progs.dat` is loaded with (and reloaded with). The defaults are
    /// strict; see [`qcvm::VmOptions`].
    pub options: qcvm::VmOptions,
    /// Globals that the `progs.dat` may define for itself (i.e. that aren't in [`Self::defs`]) but
    /// that are stored by the host, like the globals in the defs. These are accessed by name, as
    /// they are at whatever address the `progs.dat` puts them (see [`QcVmContext::host_globals`]).
    /// This is for engine extension globals, e.g. CSQC's `cltime`, which mods compiled against
    /// older defs declare themselves.
    pub host_globals: &'static [&'static str],
}

impl fmt::Debug for QcVmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QcVmConfig")
            .field("defs", &self.defs)
            .field("builtin_namespace", &self.builtin_namespace)
            .field("options", &self.options)
            .field("host_globals", &self.host_globals)
            .finish()
    }
}

impl QcVmConfig {
    /// A configuration for `progs.dat`s compiled against `defs`, with builtins registered in the
    /// namespace `N` and the default (strict) options.
    pub fn new<N: 'static>(defs: &'static dyn ErasedDefs) -> Self {
        Self {
            defs,
            builtin_namespace: TypeId::of::<N>(),
            options: Default::default(),
            host_globals: &[],
        }
    }

    /// Sets the options that the `progs.dat` is loaded with.
    pub fn with_options(mut self, options: qcvm::VmOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets the globals that the host stores even though they aren't in the defs (see
    /// [`QcVmConfig::host_globals`]).
    pub fn with_host_globals(mut self, host_globals: &'static [&'static str]) -> Self {
        self.host_globals = host_globals;
        self
    }
}

/// The [`QcVmConfig`] of `worldspawn`, in the world that a context is being loaded in.
fn vm_config(worldspawn: Entity) -> Result<QcVmConfig, InteropError> {
    thread_context()?
        .world
        .with_world_access(|world| world.get::<QcVmConfig>(worldspawn).cloned())?
        .ok_or_else(|| {
            InteropError::external_boxed(
                format!(
                    "QuakeC scripts must be attached to an entity with a `QcVmConfig`, but                      {worldspawn} has none"
                )
                .into(),
            )
        })
}

/// A QuakeC string.
///
/// QuakeC strings are bytes rather than UTF-8 (Quake uses the high bit for "gold" text), and the
/// null string (which unset fields hold) is distinct from other empty strings. The default value is
/// the null string.
#[derive(Reflect, Clone, PartialEq, Eq, Hash)]
#[reflect(opaque, Clone, Debug, PartialEq, Hash, Default)]
pub struct QcString(pub Arc<CStr>);

impl QcString {
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

impl Default for QcString {
    fn default() -> Self {
        Self::null()
    }
}

impl fmt::Debug for QcString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            f.write_str("null")
        } else {
            write!(f, "\"{}\"", self.as_bytes().escape_ascii())
        }
    }
}

impl From<&str> for QcString {
    fn from(value: &str) -> Self {
        Self::from_bytes(value.as_bytes())
    }
}

impl From<Arc<CStr>> for QcString {
    fn from(value: Arc<CStr>) -> Self {
        Self(value)
    }
}

/// A function defined by the `progs.dat` (either in QuakeC or as a builtin), by its index in the
/// function table. Index 0 is the null function.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[reflect(Clone, Debug, PartialEq, Hash, Default)]
pub struct QcFunction(pub i32);

impl QcFunction {
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
pub struct QcField(pub u16);

impl From<QcField> for qcvm::FieldRef {
    fn from(value: QcField) -> Self {
        Self { offset: value.0 }
    }
}

impl From<qcvm::FieldRef> for QcField {
    fn from(value: qcvm::FieldRef) -> Self {
        Self(value.offset)
    }
}

impl From<QcFunction> for FunctionRef {
    fn from(value: QcFunction) -> Self {
        FunctionRef::Index(value.0)
    }
}

#[derive(Debug)]
struct BevyScriptContext {
    /// Per-execution globals such as `self` and `other`
    special_args: hashbrown::HashMap<u16, qcvm::Value>,
    context: QcVmContext,
}

impl BevyScriptContext {
    fn new(context: QcVmContext) -> Self {
        Self {
            special_args: Default::default(),
            context,
        }
    }

    /// The entity that an erased entity reference refers to (`None` is worldspawn).
    fn entity(&self, erased: EntityRef) -> Entity {
        erased
            .map(|bits| Entity::from_bits(bits.get()))
            .unwrap_or(self.context.worldspawn)
    }
}

/// Marker component for individual entities controlled by QC
#[derive(Component, Reflect, Default)]
pub struct QcEntity;

/// Marker component for the worldspawn entity of a QuakeC VM, where globals will be stored.
///
/// Worldspawn is the root of its VM: every other entity of the VM is [`QcOwnedBy`] it.
#[derive(Component, Reflect, Default)]
pub struct QcWorldspawn;

/// The QuakeC VM that an entity belongs to, by its worldspawn.
///
/// Every [`QcEntity`] other than worldspawn itself must have this, and QuakeC can only access
/// entities that belong to its own VM. Despawning a worldspawn despawns every entity that it owns.
#[derive(Component, Reflect, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = QcVmEntities)]
pub struct QcOwnedBy(pub Entity);

/// The entities owned by a QuakeC VM's worldspawn (see [`QcOwnedBy`]).
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = QcOwnedBy, linked_spawn)]
pub struct QcVmEntities(Vec<Entity>);

impl QcVmEntities {
    /// The entities owned by the VM, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
}

/// The worldspawn of the QuakeC VM that `entity` belongs to: `entity` itself for worldspawns, and
/// the target of [`QcOwnedBy`] for other entities. Entities that don't belong to a VM return `None`.
pub fn vm_root(world: &World, entity: Entity) -> Option<Entity> {
    let entity_ref = world.get_entity(entity).ok()?;
    if entity_ref.contains::<QcWorldspawn>() {
        Some(entity)
    } else {
        entity_ref.get::<QcOwnedBy>().map(|owner| owner.0)
    }
}

thread_local! {
    /// The worldspawns of the VMs that are running on this thread, innermost last.
    static VM_STACK: RefCell<Vec<Entity>> = const { RefCell::new(Vec::new()) };
}

/// Pops the current VM when dropped.
struct VmStackGuard;

impl VmStackGuard {
    fn push(worldspawn: Entity) -> Self {
        VM_STACK.with_borrow_mut(|stack| stack.push(worldspawn));
        Self
    }
}

impl Drop for VmStackGuard {
    fn drop(&mut self) {
        VM_STACK.with_borrow_mut(|stack| stack.pop());
    }
}

/// The worldspawn of the QuakeC VM that is currently running on this thread, e.g. the VM that
/// called the builtin that is running. This is the innermost VM when calls are nested.
pub fn current_vm() -> Option<Entity> {
    VM_STACK.with_borrow(|stack| stack.last().copied())
}

/// Runs `f` as if the VM with the worldspawn `worldspawn` were running, so that [`current_vm`]
/// is `worldspawn` inside it. This is for hosts that run builtin code outside of QuakeC (e.g. in
/// tests).
pub fn with_current_vm<R>(worldspawn: Entity, f: impl FnOnce() -> R) -> R {
    let _guard = VmStackGuard::push(worldspawn);
    f()
}

/// The arguments of a builtin call, for [`builtin_argc`] and [`builtin_vector_arg`].
#[derive(Debug, Clone, Default)]
struct BuiltinCall {
    argc: usize,
    vectors: ArrayVec<Option<Vec3>, { qcvm::MAX_ARGS }>,
}

thread_local! {
    /// The builtins that are running on this thread, innermost last.
    // TODO: There's got to be a cleaner way to do this.
    static BUILTIN_CALLS: RefCell<Vec<BuiltinCall>> = const { RefCell::new(Vec::new()) };
}

/// Pops the current builtin call when dropped.
struct BuiltinCallGuard;

impl BuiltinCallGuard {
    fn push(call: BuiltinCall) -> Self {
        BUILTIN_CALLS.with_borrow_mut(|stack| stack.push(call));
        Self
    }
}

impl Drop for BuiltinCallGuard {
    fn drop(&mut self) {
        BUILTIN_CALLS.with_borrow_mut(|stack| stack.pop());
    }
}

/// The number of arguments that the builtin running on this thread was called with (Quake's
/// `pr_argc`), for variadic builtins. Arguments past this are `void`. This is 0 outside of a
/// builtin.
pub fn builtin_argc() -> usize {
    BUILTIN_CALLS.with_borrow(|stack| stack.last().map_or(0, |call| call.argc))
}

/// Argument `index` of the builtin running on this thread, read as a vector. Untyped arguments
/// (e.g. of `sprintf`) are passed to the builtin's function as a single scalar, so this is how to
/// read them when they are vectors. `None` if there is no such argument or it isn't made of
/// floats.
pub fn builtin_vector_arg(index: usize) -> Option<Vec3> {
    BUILTIN_CALLS.with_borrow(|stack| stack.last()?.vectors.get(index).copied().flatten())
}

/// Checks that `entity` belongs to the VM of `context`.
fn check_owner(entity: Entity, context: &QcVmContext) -> Result<(), InteropError> {
    if entity == context.worldspawn {
        return Ok(());
    }

    let world = thread_context()?.world;
    let root = world.with_world_access(|world| vm_root(world, entity))?;
    if root == Some(context.worldspawn) {
        Ok(())
    } else {
        Err(InteropError::external_boxed(
            format!(
                "Entity {entity} belongs to a different QuakeC VM ({root:?}) than the one that is                  running ({})",
                context.worldspawn
            )
            .into(),
        ))
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

/// The name that a builtin is registered under.
///
/// Builtins whose number is known to `defs` are dispatched by number like Quake does, using the
/// canonical name from `defs`, as `progs.dat`s can rename or alias builtins (e.g. LibreQuake's
/// `bprint_e = #23`). Unknown numbers (e.g. extensions) are registered by the name in the
/// `progs.dat`.
fn builtin_name<'a, T>(defs: &dyn ErasedDefs, def: &'a qcvm::FunctionDef<T>) -> Cow<'a, str> {
    match def.builtin_number().and_then(|n| defs.builtin_name(n)) {
        Some(name) => name.into(),
        None => def.name.to_string_lossy(),
    }
}

fn anyhow_addr_error(e: AddrError<InteropError>) -> AddrError<anyhow::Error> {
    match e {
        AddrError::OutOfRange => AddrError::OutOfRange,
        AddrError::Other { error } => AddrError::Other {
            error: anyhow::format_err!("{error}"),
        },
    }
}

/// The type to look an address up with for a value being stored, as `void` is the zero value of
/// whatever it is stored into.
fn lookup_type(value: &qcvm::Value) -> Type {
    match value.type_() {
        Type::Void => Type::AnyScalar,
        ty => ty,
    }
}

impl BevyScriptContext {
    /// The key of the entity field at `field`, and the type of its values.
    fn field_key(
        &self,
        field: u16,
        ty: Type,
        entity_def: &EntityTypeDef,
    ) -> Result<(ScriptValue, Type), AddrError<InteropError>> {
        if let Some(info) = self.context.defs.field(field, ty) {
            return Ok((ScriptValue::String(info.name.into()), info.type_));
        }

        match qcvm::userdata::dynamic_field(entity_def, field, ty) {
            Some((def, component)) => Ok((
                dynamic_field_key(def, component),
                qcvm::userdata::dynamic_field_type(def, component),
            )),
            None => Err(AddrError::OutOfRange),
        }
    }

    /// The key and type of the host global at `addr` that isn't in the defs (see
    /// [`QcVmConfig::host_globals`]).
    fn host_global(&self, addr: u16, ty: Type) -> Option<(ScriptValue, Type)> {
        let globals = read(&self.context.host_globals);
        let global = globals.get(addr, ty)?;
        Some((ScriptValue::String(global.name.clone().into()), global.type_))
    }

    fn get_global(&self, addr: u16, ty: Type) -> Result<qcvm::Value, AddrError<InteropError>> {
        let Some(info) = self.context.defs.global(addr, ty) else {
            let (key, _) = self.host_global(addr, ty).ok_or(AddrError::OutOfRange)?;
            let value = interop_error_to_addr_error(magic_get::<QcWorldspawn>(
                self.context.worldspawn,
                key,
            ))?;
            return interop_error_to_addr_error(bms_to_qcvm(&value, &self.context));
        };
        if let Some(val) = self.special_args.get(&info.vector_root) {
            return Ok(val
                .field(info.vector_field)
                .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?);
        }

        let key = ScriptValue::String(info.name.into());
        let value =
            interop_error_to_addr_error(magic_get::<QcWorldspawn>(self.context.worldspawn, key))?;

        interop_error_to_addr_error(bms_to_qcvm(&value, &self.context))
    }

    fn set_global(&mut self, addr: u16, value: qcvm::Value) -> Result<(), AddrError<InteropError>> {
        let Some(info) = self.context.defs.global(addr, lookup_type(&value)) else {
            let (key, ty) = self
                .host_global(addr, lookup_type(&value))
                .ok_or(AddrError::OutOfRange)?;
            let value = qcvm_to_bms(&value.or_zero(ty), &self.context)?;
            return interop_error_to_addr_error(magic_set::<QcWorldspawn>(
                self.context.worldspawn,
                key,
                value,
            ));
        };
        let value = value.or_zero(info.type_);

        if let Some(val) = self.special_args.get_mut(&info.vector_root) {
            if self.context.special_args_immutable {
                let arg_name = info.name;
                return Err(AddrError::Other {
                    error: qc_interop_error(anyhow::format_err!(
                        "Special argument {arg_name} is immutable"
                    )),
                });
            } else {
                val.set(info.vector_field, value)
                    .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?;
                return Ok(());
            }
        }

        let key = ScriptValue::String(info.name.into());
        let value = qcvm_to_bms(&value, &self.context)?;

        interop_error_to_addr_error(magic_set::<QcWorldspawn>(
            self.context.worldspawn,
            key,
            value,
        ))
    }

    fn set_field_by_name(
        &mut self,
        entity: Entity,
        name: &str,
        value: qcvm::Value,
    ) -> Result<(), AddrError<InteropError>> {
        let value = qcvm_to_bms(&value, &self.context)?;
        interop_error_to_addr_error(magic_set::<QcEntity>(
            entity,
            ScriptValue::String(name.to_owned().into()),
            value,
        ))
    }
}

impl ErasedContext for BevyScriptContext {
    fn dyn_builtin(&self, def: &qcvm::BuiltinDef) -> anyhow::Result<Arc<dyn ErasedFunction>> {
        let cached = self
            .context
            .builtins
            .read()
            .ok()
            .and_then(|builtins| builtins.get(&def.index).cloned());
        if let Some(builtin) = cached {
            return Ok(builtin);
        }

        let world = thread_context()
            .map_err(|e| anyhow::format_err!("{e}"))?
            .world;

        let function_registry = world.script_function_registry();
        let function_registry = function_registry.read();

        let def_name = builtin_name(&*self.context.defs, def);

        let func = function_registry
            .get_function(self.context.builtin_namespace, def_name.clone())
            .map_err(|_| {
                let number = def
                    .builtin_number()
                    .map(|n| format!(" (#{n})"))
                    .unwrap_or_default();
                anyhow::format_err!("No builtin named {def_name:?}{number}")
            })?;

        let builtin: Arc<dyn ErasedFunction> = Arc::new(BevyBuiltin { func: func.clone() });
        if let Ok(mut builtins) = self.context.builtins.write() {
            builtins.insert(def.index, builtin.clone());
        }

        Ok(builtin)
    }

    /// Implements `OP_STATE` in terms of the `self` and `time` globals and the `nextthink`, `frame`
    /// and `think` fields, as Quake does.
    fn dyn_state(&mut self, frame: f32, think_fn: qcvm::Value) -> anyhow::Result<()> {
        let get_global = |this: &Self, name: &str| -> anyhow::Result<qcvm::Value> {
            let info = this
                .context
                .defs
                .global_by_name(name)
                .ok_or_else(|| anyhow::format_err!("No global named {name}"))?;
            this.get_global(info.offset, info.type_)
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

        let entity = self.entity(self_);
        for (name, value) in [
            ("nextthink", qcvm::Value::Float(time + 0.1)),
            ("frame", qcvm::Value::Float(frame)),
            ("think", think_fn),
        ] {
            if self.context.defs.field_by_name(name).is_none() {
                anyhow::bail!("No field named {name}");
            }
            self.set_field_by_name(entity, name, value)
                .map_err(|e| anyhow::format_err!("Failed to set {name}: {e}"))?;
        }

        Ok(())
    }

    fn dyn_entity_get(
        &self,
        erased_ent: EntityRef,
        field: u16,
        ty: Type,
        entity_def: &EntityTypeDef,
    ) -> Result<qcvm::Value, AddrError<anyhow::Error>> {
        let (key, _) = self
            .field_key(field, ty, entity_def)
            .map_err(anyhow_addr_error)?;
        let value =
            interop_error_to_addr_error(magic_get::<QcEntity>(self.entity(erased_ent), key))
                .map_err(anyhow_addr_error)?;

        bms_to_qcvm(&value, &self.context).map_err(|e| AddrError::Other {
            error: anyhow::format_err!("{e}"),
        })
    }

    fn dyn_entity_set(
        &mut self,
        erased_ent: EntityRef,
        field: u16,
        value: qcvm::Value,
        entity_def: &EntityTypeDef,
    ) -> Result<(), AddrError<anyhow::Error>> {
        let (key, ty) = self
            .field_key(field, lookup_type(&value), entity_def)
            .map_err(anyhow_addr_error)?;
        let value =
            qcvm_to_bms(&value.or_zero(ty), &self.context).map_err(|e| AddrError::Other {
                error: anyhow::format_err!("{e}"),
            })?;

        interop_error_to_addr_error(magic_set::<QcEntity>(self.entity(erased_ent), key, value))
            .map_err(anyhow_addr_error)
    }

    fn dyn_global(&self, def: u16, ty: Type) -> Result<qcvm::Value, AddrError<anyhow::Error>> {
        self.get_global(def, ty).map_err(anyhow_addr_error)
    }

    fn dyn_set_global(
        &mut self,
        def: u16,
        value: qcvm::Value,
    ) -> Result<(), AddrError<anyhow::Error>> {
        self.set_global(def, value).map_err(anyhow_addr_error)
    }

    fn dyn_global_type(&self, def: u16) -> Option<Type> {
        self.context
            .defs
            .global(def, Type::AnyScalar)
            .map(|info| info.type_)
            .or_else(|| {
                read(&self.context.host_globals)
                    .scalars
                    .get(&def)
                    .map(|global| global.type_)
            })
    }
}

struct BevyScriptArgs<'a, A>
where
    A: ?Sized,
{
    args: &'a A,
    context: QcVmContext,
}

impl<A> fmt::Debug for BevyScriptArgs<'_, A>
where
    A: ?Sized + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BevyScriptArgs")
            .field("args", &self.args)
            .finish()
    }
}

impl<'a> QcParams for BevyScriptArgs<'a, [ScriptValue]> {
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, ArgError<Self::Error>> {
        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_to_qcvm(arg, &self.context)?)
    }

    fn count(&self) -> usize {
        self.args.len()
    }
}

impl<'a> QcParams for BevyScriptArgs<'a, VecDeque<ScriptValue>> {
    type Error = InteropError;

    fn nth(&self, index: usize) -> Result<qcvm::Value, ArgError<Self::Error>> {
        let arg = self
            .args
            .get(index)
            .ok_or_else(|| ArgError::ArgOutOfRange {
                len: self.count(),
                index,
            })?;

        Ok(bms_to_qcvm(arg, &self.context)?)
    }

    fn count(&self) -> usize {
        self.args.len()
    }
}

const FUNCTION_CALL_CONTEXT: FunctionCallContext = FunctionCallContext::new(QC_LANGUAGE);

struct BevyBuiltin {
    func: DynamicScriptFunction,
}

impl PartialEq for BevyBuiltin {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl fmt::Debug for BevyBuiltin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BevyBuiltin").field(&self.func).finish()
    }
}

impl QcType for BevyBuiltin {
    fn type_(&self) -> qcvm::Type {
        qcvm::Type::Function
    }

    fn is_null(&self) -> bool {
        false
    }
}

/// The QuakeC type of a builtin parameter of the Rust type with ID `type_id`.
///
/// Precise types let the VM resolve untyped arguments to the parameter's type, as Quake does: `void`
/// (the zero word, e.g. a global initialized to 0) becomes the zero of that type, and an untyped
/// word (an immediate whose def was stripped by an optimizing compiler) is reinterpreted as that
/// type. Unknown types are [`qcvm::Type::AnyScalar`], which pass `void` through as `()` and
/// untyped words as floats.
fn param_type(type_id: TypeId) -> qcvm::Type {
    /// Whether `type_id` is `T`, or `T` wrapped in `V` and/or `Option`.
    fn is<T: 'static>(type_id: TypeId) -> bool {
        [
            TypeId::of::<T>(),
            TypeId::of::<V<T>>(),
            TypeId::of::<Option<T>>(),
            TypeId::of::<Option<V<T>>>(),
        ]
        .contains(&type_id)
    }

    if is::<Vec3>(type_id) || is::<[f32; 3]>(type_id) {
        qcvm::Type::Vector
    } else if is::<f32>(type_id) || is::<f64>(type_id) {
        qcvm::Type::Float
    } else if is::<QcString>(type_id) {
        qcvm::Type::String
    } else if is::<QcFunction>(type_id) || is::<DynamicScriptFunction>(type_id) {
        qcvm::Type::Function
    } else if is::<QcField>(type_id) {
        qcvm::Type::Field
    } else if is::<ReflectReference>(type_id) {
        qcvm::Type::Entity
    } else {
        qcvm::Type::AnyScalar
    }
}

impl BevyBuiltin {
    fn signature(&self) -> ArrayVec<qcvm::Type, { qcvm::MAX_ARGS }> {
        self.func
            .info
            .arg_info
            .iter()
            .filter(|arg| arg.type_id != TypeId::of::<FunctionCallContext>())
            .take(qcvm::MAX_ARGS)
            .map(|arg| param_type(arg.type_id))
            .collect()
    }

    fn call(&self, context: FnCall<'_>) -> Result<qcvm::Value, InteropError> {
        let erased: &dyn Any = &*context;
        let Some(script_context) = erased.downcast_ref::<BevyScriptContext>() else {
            return Err(InteropError::external_boxed(
                "Builtins registered with `bevy_mod_scripting` can only be called by \
                 `bevy_mod_scripting_qcvm`"
                    .into(),
            ));
        };
        let qc_context = script_context.context.clone();

        let sig = self.signature();
        let argc = context.num_args();
        let args = context
            .arguments(&sig)
            .zip(&sig)
            .enumerate()
            .map(|(i, (val, ty))| {
                let val = val.map_err(qc_interop_error)?;
                // Arguments that weren't passed are the zero of their type, so that builtins
                // can take optional arguments of any type.
                let val = match val {
                    qcvm::Value::Void if i >= argc => qcvm::Value::zero(*ty),
                    val => val,
                };
                qcvm_to_bms(&val, &qc_context)
            })
            .collect::<Result<ArrayVec<_, { qcvm::MAX_ARGS }>, _>>()?;

        let _call = BuiltinCallGuard::push(BuiltinCall {
            argc,
            vectors: (0..argc.min(qcvm::MAX_ARGS))
                .map(|i| {
                    context
                        .vector_arg(i)
                        .map(|v| Vec3::from_array(v.to_array()))
                })
                .collect(),
        });
        let result = self.func.call(args, FUNCTION_CALL_CONTEXT)?;
        bms_to_qcvm(&result, &qc_context)
    }
}

impl ErasedFunction for BevyBuiltin {
    fn dyn_signature(&self) -> anyhow::Result<ArrayVec<qcvm::Type, { qcvm::MAX_ARGS }>> {
        Ok(self.signature())
    }

    fn dyn_call(&self, context: FnCall<'_>) -> anyhow::Result<qcvm::Value> {
        self.call(context).map_err(|e| anyhow::format_err!("{e}"))
    }
}

/// The default [`ReflectToValue`], which handles [`QcString`], [`QcFunction`], [`QcField`],
/// [`Vec3`], `[f32; 3]`, `f32`, `f64` and `bool`.
pub fn default_reflect_to_value(value: &dyn PartialReflect) -> Result<qcvm::Value, InteropError> {
    if let Some(s) = value.try_downcast_ref::<QcString>() {
        Ok(qcvm::Value::String(s.0.clone()))
    } else if let Some(f) = value.try_downcast_ref::<QcFunction>() {
        Ok(FunctionIndex(f.0).to_value())
    } else if let Some(f) = value.try_downcast_ref::<QcField>() {
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

fn bms_to_qcvm(
    bms_value: &ScriptValue,
    context: &QcVmContext,
) -> Result<qcvm::Value, InteropError> {
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
                && r.base.type_id() == TypeId::of::<QcEntity>() =>
        {
            check_owner(*ent, context)?;
            Ok(qcvm::Value::Entity(if *ent == context.worldspawn {
                None
            } else {
                Some(entity_to_erased(*ent))
            }))
        }
        ScriptValue::Reference(r)
            if matches!(r.base.base_id(), ReflectBase::Component(..))
                && r.base.type_id() == TypeId::of::<QcWorldspawn>() =>
        {
            Err(InteropError::external_boxed(
                "Tried to convert a `QcWorldspawn` reference to a qcvm entity. If this is what \
                you intended, add a `QcEntity` component to the worldspawn entity and pass a \
                `ReflectReference` to that component instead"
                    .into(),
            ))
        }
        ScriptValue::Reference(val) => {
            let world = thread_context()?.world;
            val.with_reflect(world, context.reflect_to_value)?
        }
        ScriptValue::Function(func) => Ok(qcvm::Value::Function(
            match context.function_index(func) {
                // One of this VM's own function values (see `QcVmContext::function_value`).
                Some(index) => qcvm::Function::Index(FunctionIndex(index)),
                None => qcvm::Function::Host(Arc::new(BevyBuiltin { func: func.clone() })),
            },
        )),
        ScriptValue::Error(e) => Err(e.clone()),
        _ => Err(InteropError::NotImplemented),
    }
}

fn qcvm_to_bms(
    qcvm_value: &qcvm::Value,
    root_ctx: &QcVmContext,
) -> Result<ScriptValue, InteropError> {
    match qcvm_value {
        qcvm::Value::Void => Ok(ScriptValue::Unit),
        qcvm::Value::Entity(entity) => {
            let entity = entity
                .map(|bits| Entity::from_bits(bits.get()))
                .unwrap_or(root_ctx.worldspawn);
            let world = thread_context()?.world;

            Ok(ScriptValue::Reference(
                ReflectReference::new_component_ref::<QcEntity>(entity, world)?,
            ))
        }
        qcvm::Value::Function(qcvm::Function::Index(index)) => root_ctx.function_value(index.0),
        qcvm::Value::Function(qcvm::Function::Host(function)) => {
            let function: &dyn Any = &**function;
            if let Some(builtin) = function.downcast_ref::<BevyBuiltin>() {
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
            QcString(cstr.clone()),
        ))),
        qcvm::Value::Field(field) => Ok(ScriptValue::Reference(ReflectReference::new_allocated(
            QcField::from(*field),
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
type BuiltinCache = Arc<RwLock<hashbrown::HashMap<i32, Arc<dyn ErasedFunction>>>>;

/// Locks `lock` for reading, ignoring poisoning (the data is a cache or a swappable handle, so a
/// panic while it was held can't leave it inconsistent).
fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

/// Locks `lock` for writing, ignoring poisoning (see [`read`]).
fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// The function values of a VM (see [`QcVmContext::function_value`]), one per function index so
/// that QuakeC's `==` on functions still works after a function has been passed out of the VM and
/// back in.
#[derive(Default)]
struct FunctionCache {
    by_index: RwLock<hashbrown::HashMap<i32, DynamicScriptFunction>>,
    /// Function index by [`DynamicScriptFunction::identity`]. The identities stay valid because
    /// `by_index` keeps the functions alive.
    by_identity: RwLock<hashbrown::HashMap<usize, i32>>,
}

impl FunctionCache {
    fn clear(&self) {
        write(&self.by_index).clear();
        write(&self.by_identity).clear();
    }
}

/// A global that the `progs.dat` defines and the host stores (see [`QcVmConfig::host_globals`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostGlobal {
    /// The name that the host stores it under. Vectors are stored both as a whole (under their
    /// own name) and as their components (`{name}_x` etc., as `qcc` names them).
    pub name: String,
    /// Its address in the `progs.dat`.
    pub offset: u16,
    /// Its type.
    pub type_: Type,
}

/// The [`HostGlobal`]s of a VM, by address.
#[derive(Debug, Default)]
struct HostGlobals {
    /// Whole vectors.
    vectors: hashbrown::HashMap<u16, HostGlobal>,
    /// Scalars, including the components of vectors.
    scalars: hashbrown::HashMap<u16, HostGlobal>,
}

impl HostGlobals {
    fn new(vm: &qcvm::QcVm, defs: &dyn ErasedDefs, names: &[&str]) -> Self {
        let mut out = Self::default();
        if names.is_empty() {
            return out;
        }
        for def in vm.global_defs() {
            let Ok(name) = def.name.to_str() else {
                continue;
            };
            // Globals in the defs are stored by the host anyway, by address.
            if !names.contains(&name) || defs.global_by_name(name).is_some() {
                continue;
            }
            let Ok(type_) = Type::try_from(def.type_) else {
                continue;
            };
            if type_ == Type::Vector {
                out.vectors.insert(
                    def.offset,
                    HostGlobal {
                        name: name.to_owned(),
                        offset: def.offset,
                        type_,
                    },
                );
                for (i, component) in VectorField::FIELDS.iter().enumerate() {
                    let offset = def.offset + i as u16;
                    out.scalars.insert(
                        offset,
                        HostGlobal {
                            name: format!("{name}_{component}"),
                            offset,
                            type_: Type::Float,
                        },
                    );
                }
            } else {
                out.scalars.insert(
                    def.offset,
                    HostGlobal {
                        name: name.to_owned(),
                        offset: def.offset,
                        type_,
                    },
                );
            }
        }
        out
    }

    fn get(&self, addr: u16, ty: Type) -> Option<&HostGlobal> {
        match ty {
            Type::Vector => self.vectors.get(&addr),
            _ => self.scalars.get(&addr),
        }
    }
}

/// The VM of a context. It is shared by every clone of the context (including the ones captured
/// by function values), so that hot reloading replaces the VM for all of them.
///
/// Running QuakeC holds a read lock for as long as it runs. Calls from inside a call (e.g. a
/// builtin calling a function value) take the lock again with
/// [`parking_lot::RwLock::read_recursive`], which doesn't wait for writers, so they can't
/// deadlock. Replacing the VM (see [`QcVmContext::replace_vm`]) takes the write lock, so it waits
/// until no QuakeC from this VM is running.
type VmCell = Arc<parking_lot::RwLock<qcvm::QcVm>>;

/// Wrapper around `qcvm::QcVm` that adds extra configuration for converting values
#[derive(Clone)]
pub struct QcVmContext {
    vm: VmCell,
    /// A conversion function for arbitrary reflect values. By default, this is
    /// [`default_reflect_to_value`].
    pub reflect_to_value: ReflectToValue,
    /// The worldspawn entity, which holds the globals and which QuakeC refers to as `world`.
    pub worldspawn: Entity,
    /// If `true`, special arguments (see the callback handler) can't be written by QuakeC.
    pub special_args_immutable: bool,
    /// The defs that the `progs.dat` is compiled against (see [`QcVmConfig::defs`]).
    pub defs: &'static dyn ErasedDefs,
    /// The namespace that builtins are looked up in (see [`QcVmConfig::builtin_namespace`]).
    pub builtin_namespace: TypeId,
    /// The script attachment that this context was loaded for, if it was loaded by
    /// `bevy_mod_scripting`. Calling a function value of this VM makes this the current
    /// attachment (see [`WorldExtensions::current_attachment`]) while it runs, as
    /// `bevy_mod_scripting` does for callbacks.
    pub attachment: Option<ScriptAttachment>,
    builtins: BuiltinCache,
    functions: Arc<FunctionCache>,
    host_global_names: &'static [&'static str],
    host_globals: Arc<RwLock<HostGlobals>>,
}

/// A [`QcVmContext`] that doesn't keep its function values alive, for the function values
/// themselves to hold (which would otherwise be a reference cycle through [`FunctionCache`]).
struct WeakQcVmContext {
    vm: VmCell,
    reflect_to_value: ReflectToValue,
    worldspawn: Entity,
    special_args_immutable: bool,
    defs: &'static dyn ErasedDefs,
    builtin_namespace: TypeId,
    attachment: Option<ScriptAttachment>,
    builtins: BuiltinCache,
    functions: Weak<FunctionCache>,
    host_global_names: &'static [&'static str],
    host_globals: Arc<RwLock<HostGlobals>>,
}

impl WeakQcVmContext {
    fn upgrade(&self) -> Option<QcVmContext> {
        Some(QcVmContext {
            vm: self.vm.clone(),
            reflect_to_value: self.reflect_to_value,
            worldspawn: self.worldspawn,
            special_args_immutable: self.special_args_immutable,
            defs: self.defs,
            builtin_namespace: self.builtin_namespace,
            attachment: self.attachment.clone(),
            builtins: self.builtins.clone(),
            functions: self.functions.upgrade()?,
            host_global_names: self.host_global_names,
            host_globals: self.host_globals.clone(),
        })
    }
}

/// Makes an attachment the current one of a world guard, putting the previous one back when
/// dropped (also on panic).
struct AttachmentGuard<'w> {
    world: WorldGuard<'w>,
    previous: Option<CurrentScriptAttachment>,
}

impl<'w> AttachmentGuard<'w> {
    fn set(world: WorldGuard<'w>, attachment: ScriptAttachment) -> Self {
        let previous =
            world.replace_cached_registry(CurrentScriptAttachment(Some(attachment)));
        Self {
            world,
            previous: Some(previous),
        }
    }
}

impl Drop for AttachmentGuard<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.world.replace_cached_registry(previous);
        }
    }
}

impl fmt::Debug for QcVmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QcVmContext")
            .field("vm", &*self.vm())
            .field("worldspawn", &self.worldspawn)
            .field("special_args_immutable", &self.special_args_immutable)
            .field("defs", &self.defs)
            .field("attachment", &self.attachment)
            .finish_non_exhaustive()
    }
}

impl QcVmContext {
    /// Creates a context for `vm`, with globals stored on `worldspawn` and the defs and builtins of
    /// `config`.
    pub fn new(vm: qcvm::QcVm, worldspawn: Entity, config: &QcVmConfig) -> Self {
        let host_globals = HostGlobals::new(&vm, config.defs, config.host_globals);
        Self {
            vm: Arc::new(parking_lot::RwLock::new(vm)),
            reflect_to_value: default_reflect_to_value,
            worldspawn,
            special_args_immutable: false,
            defs: config.defs,
            builtin_namespace: config.builtin_namespace,
            attachment: None,
            builtins: Default::default(),
            functions: Default::default(),
            host_global_names: config.host_globals,
            host_globals: Arc::new(RwLock::new(host_globals)),
        }
    }

    /// The VM. This locks it for reading, which keeps hot reloading (see
    /// [`QcVmContext::replace_vm`]) from replacing it until the guard is dropped, so don't hold
    /// on to it for longer than needed. It can be called while the VM is already locked on this
    /// thread (e.g. from a builtin).
    pub fn vm(&self) -> parking_lot::RwLockReadGuard<'_, qcvm::QcVm> {
        self.vm.read_recursive()
    }

    /// Replaces the VM (e.g. when the `progs.dat` is reloaded) for every clone of this context.
    /// Function values of the old VM are forgotten, so function values that the host still holds
    /// call the function with the same index in the new VM.
    ///
    /// This waits until no QuakeC from this VM is running (on any thread), so that a reload never
    /// happens in the middle of a call. It deadlocks if it is called while this thread is running
    /// QuakeC from this VM (e.g. from a builtin) or holds the guard from [`QcVmContext::vm`]:
    /// reloads requested from inside a call have to be deferred until the call returns (e.g. by
    /// sending a message that is handled by a system).
    pub fn replace_vm(&self, vm: qcvm::QcVm) {
        let mut current = self.vm.write();
        *write(&self.host_globals) = HostGlobals::new(&vm, self.defs, self.host_global_names);
        *current = vm;
        write(&self.builtins).clear();
        self.functions.clear();
    }

    fn downgrade(&self) -> WeakQcVmContext {
        WeakQcVmContext {
            vm: self.vm.clone(),
            reflect_to_value: self.reflect_to_value,
            worldspawn: self.worldspawn,
            special_args_immutable: self.special_args_immutable,
            defs: self.defs,
            builtin_namespace: self.builtin_namespace,
            attachment: self.attachment.clone(),
            builtins: self.builtins.clone(),
            functions: Arc::downgrade(&self.functions),
            host_global_names: self.host_global_names,
            host_globals: self.host_globals.clone(),
        }
    }

    /// The globals that the `progs.dat` defines and the host stores (see
    /// [`QcVmConfig::host_globals`]), in no particular order. Vectors are listed both as a whole
    /// and as their components.
    pub fn host_globals(&self) -> Vec<HostGlobal> {
        let globals = read(&self.host_globals);
        globals
            .vectors
            .values()
            .chain(globals.scalars.values())
            .cloned()
            .collect()
    }

    /// The function with index `index` as a value that can be called from anywhere
    /// (`ScriptValue::Unit` for the null function).
    ///
    /// QuakeC functions are [`DynamicScriptFunction`]s named after the function, which run the
    /// function in this VM when called. Calling one doesn't go through `bevy_mod_scripting`'s
    /// callback handling (and so doesn't lock the script context), so it works while the VM is
    /// already running, e.g. from inside a builtin. Builtins that are registered in the VM's
    /// builtin namespace are the registered function itself.
    ///
    /// Each index always gives the same function (until the VM is replaced), and passing it back
    /// into the VM gives the function index again, so QuakeC can compare function values.
    pub fn function_value(&self, index: i32) -> Result<ScriptValue, InteropError> {
        if index == 0 {
            return Ok(ScriptValue::Unit);
        }
        if let Some(func) = read(&self.functions.by_index).get(&index) {
            return Ok(ScriptValue::Function(func.clone()));
        }

        let vm = self.vm();
        let def = vm
            .functions()
            .get_by_index(index)
            .map_err(|e| InteropError::string(format!("No QuakeC function #{index}: {e}")))?;

        let registered = if def.is_builtin() {
            let name = builtin_name(&*self.defs, def);
            thread_context()
                .ok()
                .and_then(|ctx| {
                    ctx.world
                        .script_function_registry()
                        .read()
                        .get_function(self.builtin_namespace, name.into_owned())
                        .ok()
                        .cloned()
                })
        } else {
            None
        };

        let func = match registered {
            Some(func) => func,
            None => {
                let weak = self.downgrade();
                let info = FunctionInfo::default().with_name(def.name.to_string_lossy().into_owned());
                DynamicScriptFunction::from(
                    move |_caller: FunctionCallContext, args: VecDeque<ScriptValue>| match weak
                        .upgrade()
                    {
                        Some(context) => context
                            .call_index(index, &args)
                            .unwrap_or_else(ScriptValue::Error),
                        None => ScriptValue::Error(InteropError::string(
                            "The QuakeC VM of this function has been unloaded".to_owned(),
                        )),
                    },
                )
                .with_info(info)
            }
        };

        // Another thread may have got here first, in which case use its function so that there's
        // only ever one value per index.
        let mut by_index = write(&self.functions.by_index);
        let func = by_index.entry(index).or_insert(func).clone();
        write(&self.functions.by_identity).insert(func.identity(), index);
        Ok(ScriptValue::Function(func))
    }

    /// The function index of `func`, if it is a function value of this VM (see
    /// [`QcVmContext::function_value`]).
    pub fn function_index(&self, func: &DynamicScriptFunction) -> Option<i32> {
        read(&self.functions.by_identity).get(&func.identity()).copied()
    }

    /// Runs the function with index `index`, with the thread's world. The return value is read as
    /// a scalar, as the progs don't say what type functions return.
    fn call_index(&self, index: i32, args: &VecDeque<ScriptValue>) -> Result<ScriptValue, InteropError> {
        let world = thread_context()?.world;
        let _attachment = self
            .attachment
            .clone()
            .map(|attachment| AttachmentGuard::set(world, attachment));
        let _vm = VmStackGuard::push(self.worldspawn);
        let vm = self.vm();
        let mut script_context = BevyScriptContext::new(self.clone());
        let ret = vm
            .run(
                &mut script_context,
                FunctionRef::Index(index),
                BevyScriptArgs::<VecDeque<ScriptValue>> {
                    context: self.clone(),
                    args,
                },
            )
            .map_err(qc_interop_error)?;
        self.return_value(&ret)
    }

    /// A function's return value, read as a scalar (a vector return value is its `x`). Functions
    /// that don't return anything leave whatever was last returned, so this is only meaningful for
    /// functions that do.
    fn return_value(&self, ret: &qcvm::ReturnValue<'_>) -> Result<ScriptValue, InteropError> {
        match ret.scalar() {
            Ok(value) => qcvm_to_bms(&value, self),
            // Garbage left over from an earlier return, for functions that return nothing.
            Err(_) => Ok(ScriptValue::Unit),
        }
    }

    /// Calls a QuakeC function with exclusive access to `world`.
    ///
    /// This doesn't go through `bevy_mod_scripting`: it creates a fresh world access guard for
    /// `world`, which is used by field and global accessors and builtins until this returns. Hosts
    /// should normally call scripts through `bevy_mod_scripting` instead (named callbacks with
    /// `AppScriptCallbackRunners`, and function values with [`DynamicScriptFunction::call`]),
    /// which works for every scripting language; this is mostly useful in tests.
    ///
    /// The progs don't say what type functions return, so the return value is read as `R` (see
    /// [`qcvm::ReturnValue`]). Use `()` to ignore it, which never fails.
    ///
    /// While the function runs, [`current_vm`] is this context's worldspawn.
    pub fn call_with_world<R>(
        &self,
        world: &mut World,
        function: impl Into<FunctionRef>,
        args: impl QcParams,
    ) -> Result<R, InteropError>
    where
        R: qcvm::FromReturnValue,
    {
        let cache = <WorldGuard<'_> as WorldExtensions>::setup_cache(
            world,
            CurrentScriptAttachment(self.attachment.clone()),
        );
        WorldAccessGuard::with_static_guard(world, cache, |_guard| {
            let _vm = VmStackGuard::push(self.worldspawn);
            let mut context = BevyScriptContext::new(self.clone());
            self.vm()
                .run(&mut context, function, args)
                .and_then(|ret| ret.get())
                .map_err(qc_interop_error)
        })
    }

    /// The global definitions in the `progs.dat`.
    pub fn global_defs(&self) -> Vec<qcvm::GlobalDef> {
        self.vm().global_defs().to_vec()
    }

    /// The entity field definitions in the `progs.dat`.
    pub fn field_defs(&self) -> Vec<FieldDef> {
        self.vm().field_defs().to_vec()
    }

    /// The initial value of a global (see [`qcvm::QcVm::initial_global`]).
    pub fn initial_global(&self, addr: u16) -> anyhow::Result<qcvm::Value> {
        self.vm().initial_global(addr)
    }

    /// The initial value of a vector global (see [`qcvm::QcVm::initial_global_vector`]).
    pub fn initial_global_vector(&self, addr: u16) -> anyhow::Result<Vec3> {
        self.vm()
            .initial_global_vector(addr)
            .map(|v| Vec3::from_array(v.to_array()))
    }

    /// Whether the function with index `index` is a builtin that is registered in this VM's
    /// builtin namespace (e.g. for FTEQW's `checkbuiltin`).
    pub fn has_builtin(&self, world: &World, index: i32) -> bool {
        let vm = self.vm();
        let Ok(def) = vm.functions().get_by_index(index) else {
            return false;
        };
        if !def.is_builtin() {
            return false;
        }
        let Some(registry) =
            world.get_resource::<bevy_mod_scripting_bindings::AppScriptFunctionRegistry>()
        else {
            return false;
        };
        let name = builtin_name(&*self.defs, def).into_owned();
        registry
            .read()
            .get_function(self.builtin_namespace, name)
            .is_ok()
    }

    /// The QuakeC value for a reference to `entity` (`world` for the worldspawn).
    pub fn entity_value(&self, entity: Entity) -> qcvm::Value {
        qcvm::Value::Entity(if entity == self.worldspawn {
            None
        } else {
            Some(entity_to_erased(entity))
        })
    }

    /// The entity that a QuakeC entity reference refers to (the worldspawn for `world`).
    pub fn value_entity(&self, entity: qcvm::EntityRef) -> Entity {
        entity
            .map(|bits| Entity::from_bits(bits.get()))
            .unwrap_or(self.worldspawn)
    }

    /// The definition of the global called `name` in the `progs.dat`, if it's stored in the VM
    /// rather than by the host (i.e. it isn't in [`QcVmContext::defs`]). These are globals that
    /// the `progs.dat` defines for itself, including engine extension globals that the host's
    /// defs don't cover.
    pub fn vm_global_def(&self, name: &str) -> Option<qcvm::GlobalDef> {
        self.vm()
            .global_defs()
            .iter()
            .find(|def| {
                def.name.to_bytes() == name.as_bytes()
                    && self
                        .defs
                        .global(def.offset, qcvm::Type::AnyScalar)
                        .is_none()
            })
            .cloned()
    }

    /// The value of a global stored in the VM by name (see [`QcVmContext::vm_global_def`]).
    pub fn vm_global(&self, name: &str) -> Option<anyhow::Result<qcvm::Value>> {
        let def = self.vm_global_def(name)?;
        let vm = self.vm();
        Some(if matches!(def.type_, qcvm::VmType::Vector) {
            vm.global_vector(def.offset).map(qcvm::Value::Vector)
        } else {
            vm.global(def.offset)
        })
    }

    /// Sets a global stored in the VM by name (see [`QcVmContext::vm_global_def`]). Returns
    /// `Ok(false)` if the `progs.dat` doesn't define it.
    pub fn set_vm_global(&self, name: &str, value: qcvm::Value) -> anyhow::Result<bool> {
        let Some(def) = self.vm_global_def(name) else {
            return Ok(false);
        };
        self.vm().set_global(def.offset, value)?;
        Ok(true)
    }
}

impl IntoScriptPluginParams for QcScriptingPlugin {
    const LANGUAGE: Language = QC_LANGUAGE;

    type C = QcVmContext;
    type R = ();

    fn build_runtime() -> Self::R {}

    fn handler() -> bevy_mod_scripting_core::handler::HandlerFn<Self> {
        fn qc_callback_handler(
            mut args: Vec<ScriptValue>,
            context_key: &ScriptAttachment,
            callback: &CallbackLabel,
            context: &mut QcVmContext,
            world_id: WorldId,
        ) -> Result<ScriptValue, InteropError> {
            let config = QcScriptingPlugin::readonly_configuration(world_id);
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
                            let value = bms_to_qcvm(&v, context)?;
                            let key = context.defs.global_by_name(&k).ok_or_else(|| {
                                let reason = format!("No global with name {k}");
                                InteropError::InvalidIndex {
                                    index: Box::new(k.into()),
                                    reason: Box::new(reason),
                                }
                            })?;

                            Ok((key.offset, value))
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
            let vm = context.vm();
            if vm.functions().get_by_name(&callback_cstr).is_err() {
                return Ok(ScriptValue::Unit);
            }

            let mut script_context = BevyScriptContext::new(context.clone());
            script_context.special_args = special_args;
            let _vm = VmStackGuard::push(context.worldspawn);

            // The progs don't say what type callbacks return, so the return value is read as a
            // scalar (see `QcVmContext::return_value`).
            let ret = vm
                .run(
                    &mut script_context,
                    callback_cstr,
                    BevyScriptArgs::<[ScriptValue]> {
                        context: context.clone(),
                        args: args_ref,
                    },
                )
                .map_err(qc_interop_error)?;

            context.return_value(&ret)
        }

        qc_callback_handler
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
        fn qc_context_loader(
            context_key: &ScriptAttachment,
            content: &[u8],
            world_id: WorldId,
        ) -> Result<QcVmContext, InteropError> {
            let config = QcScriptingPlugin::readonly_configuration(world_id);
            // Globals are stored on the entity the script is attached to, which also has the
            // configuration of the VM.
            let worldspawn = context_key.entity().ok_or_else(|| {
                InteropError::external_boxed(
                    "QuakeC scripts must be attached to a worldspawn entity".into(),
                )
            })?;
            let vm_config = vm_config(worldspawn)?;
            let vm = qcvm::QcVm::load_with_defs(
                std::io::Cursor::new(content),
                &*vm_config.defs,
                vm_config.options.clone(),
            )
            .map_err(qc_interop_error)?;
            let mut out = QcVmContext::new(vm, worldspawn, &vm_config);
            out.attachment = Some(context_key.clone());

            for init in config.context_initialization_callbacks {
                init(context_key, &mut out)?;
            }

            Ok(out)
        }

        qc_context_loader
    }

    fn context_reloader() -> bevy_mod_scripting_core::context::ContextReloadFn<Self> {
        // TODO: We should diff the `progs.dat` global values between the old and new contexts and set globals
        //       on the script attachment.
        fn qc_context_reloader(
            context_key: &ScriptAttachment,
            content: &[u8],
            previous_context: &mut QcVmContext,
            world_id: WorldId,
        ) -> Result<(), InteropError> {
            let config = QcScriptingPlugin::readonly_configuration(world_id);
            // Keep the defs and options that the context was loaded with.
            // The read guard has to be dropped before `replace_vm`, which would otherwise wait
            // for it forever.
            let options = previous_context.vm().options().clone();
            let vm = qcvm::QcVm::load_with_defs(
                std::io::Cursor::new(content),
                &*previous_context.defs,
                options,
            )
            .map_err(qc_interop_error)?;
            // Every clone of the context (including the ones held by function values) shares the
            // VM, so they all see the new one.
            previous_context.replace_vm(vm);

            // Hosts set up per-VM state from the `progs.dat` (e.g. which functions it defines)
            // in these, which has to be redone for the new `progs.dat`.
            for init in config.context_initialization_callbacks {
                init(context_key, previous_context)?;
            }

            Ok(())
        }

        qc_context_reloader
    }
}

/// The plugin configuration of [`QcScriptingPlugin`], by world.
type PluginConfigs = Vec<Option<ScriptingPluginConfiguration<QcScriptingPlugin>>>;

static PLUGIN_CONFIGS: RwLock<PluginConfigs> = RwLock::new(Vec::new());

impl GetPluginThreadConfig<QcScriptingPlugin> for QcScriptingPlugin {
    fn readonly_configuration(world: WorldId) -> ScriptingPluginConfiguration<Self> {
        PLUGIN_CONFIGS
            .read()
            .unwrap()
            .get(world.sparse_set_index())
            .and_then(|config| *config)
            .unwrap_or_else(|| {
                panic!(
                    "Configuration for plugin {} not set for world {world:?}. Did you add the \
                     plugin to the app?",
                    std::any::type_name::<Self>(),
                )
            })
    }

    fn set_world_local_config(world: WorldId, config: ScriptingPluginConfiguration<Self>) {
        let mut configs = PLUGIN_CONFIGS.write().unwrap();
        let index = world.sparse_set_index();
        if index >= configs.len() {
            configs.resize_with(index + 1, || None);
        }
        configs[index] = Some(config);
    }
}

impl bevy_app::Plugin for QcScriptingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.register_type::<QcEntity>()
            .register_type::<QcWorldspawn>()
            .register_type::<QcOwnedBy>();
        self.scripting_plugin.build(app);
    }

    fn finish(&self, app: &mut bevy_app::App) {
        self.scripting_plugin.finish(app);
    }
}

impl AsMut<ScriptingPlugin<Self>> for QcScriptingPlugin {
    fn as_mut(&mut self) -> &mut ScriptingPlugin<Self> {
        &mut self.scripting_plugin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defs with no globals, fields or builtins.
    struct EmptyDefs;

    impl qcvm::QcDefs for EmptyDefs {
        type GlobalAddr = qcvm::EmptyAddress;
        type FieldAddr = qcvm::EmptyAddress;
        type BuiltinAddr = qcvm::EmptyAddress;

        const CRC: i32 = 1234;
    }

    /// A `progs.dat` with no code, compiled against [`EmptyDefs`].
    fn empty_progs() -> Vec<u8> {
        const HEADER_SIZE: i32 = 4 + 4 + 6 * 8 + 4;
        let mut bytes = Vec::new();
        bytes.extend(6i32.to_le_bytes());
        bytes.extend(<EmptyDefs as qcvm::QcDefs>::CRC.to_le_bytes());
        // Statements, global defs, field defs, functions, strings (just the null string), globals.
        for (offset, count) in [
            (HEADER_SIZE + 1, 0i32),
            (HEADER_SIZE + 1, 0i32),
            (HEADER_SIZE + 1, 0i32),
            (HEADER_SIZE + 1, 0i32),
            (HEADER_SIZE, 1),
            (HEADER_SIZE + 1, 0i32),
        ] {
            bytes.extend(offset.to_le_bytes());
            bytes.extend(count.to_le_bytes());
        }
        // Entity field count.
        bytes.extend(0i32.to_le_bytes());
        bytes.push(0);
        bytes
    }

    fn context(worldspawn: Entity) -> QcVmContext {
        let config = QcVmConfig::new::<QcWorldspawn>(&EmptyDefs);
        let vm = qcvm::QcVm::load_with_defs(
            std::io::Cursor::new(empty_progs()),
            &*config.defs,
            Default::default(),
        )
        .unwrap();
        QcVmContext::new(vm, worldspawn, &config)
    }

    /// Runs `f` with `world` as the thread's world, as builtins are.
    fn with_world<O>(world: &mut World, f: impl FnOnce() -> O) -> O {
        let cache = <WorldGuard<'_> as WorldExtensions>::setup_cache(
            world,
            CurrentScriptAttachment::empty(),
        );
        WorldAccessGuard::with_static_guard(world, cache, |_guard| f())
    }

    fn entity_value(entity: Entity) -> ScriptValue {
        let world = thread_context().unwrap().world;
        ScriptValue::Reference(
            ReflectReference::new_component_ref::<QcEntity>(entity, world).unwrap(),
        )
    }

    #[test]
    fn vms_only_accept_their_own_entities() {
        let mut world = World::new();
        let a = world.spawn((QcWorldspawn, QcEntity)).id();
        let b = world.spawn((QcWorldspawn, QcEntity)).id();
        let a_ent = world.spawn((QcEntity, QcOwnedBy(a))).id();
        let b_ent = world.spawn((QcEntity, QcOwnedBy(b))).id();
        let unowned = world.spawn(QcEntity).id();

        // Both VMs have the same defs, so only ownership can tell their entities apart.
        let (ctx_a, ctx_b) = (context(a), context(b));

        with_world(&mut world, || {
            assert_eq!(
                bms_to_qcvm(&entity_value(a), &ctx_a).unwrap(),
                qcvm::Value::Entity(None)
            );
            assert_eq!(
                bms_to_qcvm(&entity_value(a_ent), &ctx_a).unwrap(),
                qcvm::Value::Entity(Some(entity_to_erased(a_ent)))
            );
            assert!(bms_to_qcvm(&entity_value(b_ent), &ctx_a).is_err());
            assert!(bms_to_qcvm(&entity_value(b), &ctx_a).is_err());
            assert!(bms_to_qcvm(&entity_value(unowned), &ctx_a).is_err());

            assert!(bms_to_qcvm(&entity_value(b_ent), &ctx_b).is_ok());
            assert!(bms_to_qcvm(&entity_value(a_ent), &ctx_b).is_err());
        });
    }

    #[test]
    fn vm_root_follows_ownership() {
        let mut world = World::new();
        let root = world.spawn((QcWorldspawn, QcEntity)).id();
        let owned = world.spawn((QcEntity, QcOwnedBy(root))).id();
        let unowned = world.spawn(QcEntity).id();

        assert_eq!(vm_root(&world, root), Some(root));
        assert_eq!(vm_root(&world, owned), Some(root));
        assert_eq!(vm_root(&world, unowned), None);
        assert_eq!(
            world
                .get::<QcVmEntities>(root)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [owned]
        );
    }

    #[test]
    fn despawning_worldspawn_only_despawns_its_entities() {
        let mut world = World::new();
        let a = world.spawn((QcWorldspawn, QcEntity)).id();
        let b = world.spawn((QcWorldspawn, QcEntity)).id();
        let a_ent = world.spawn((QcEntity, QcOwnedBy(a))).id();
        let b_ent = world.spawn((QcEntity, QcOwnedBy(b))).id();

        world.despawn(a);

        assert!(world.get_entity(a_ent).is_err());
        assert!(world.get_entity(b).is_ok());
        assert!(world.get_entity(b_ent).is_ok());
    }

    #[test]
    fn current_vm_is_innermost() {
        let (a, b) = (
            Entity::from_raw_u32(1).unwrap(),
            Entity::from_raw_u32(2).unwrap(),
        );
        assert_eq!(current_vm(), None);
        {
            let _a = VmStackGuard::push(a);
            assert_eq!(current_vm(), Some(a));
            {
                let _b = VmStackGuard::push(b);
                assert_eq!(current_vm(), Some(b));
            }
            assert_eq!(current_vm(), Some(a));
        }
        assert_eq!(current_vm(), None);
    }

    #[test]
    fn with_current_vm_sets_the_current_vm() {
        let vm = Entity::from_raw_u32(1).unwrap();
        assert_eq!(with_current_vm(vm, current_vm), Some(vm));
        assert_eq!(current_vm(), None);
    }

    #[test]
    fn builtin_argc_is_innermost() {
        assert_eq!(builtin_argc(), 0);
        {
            let mut vectors = ArrayVec::new();
            vectors.push(Some(Vec3::X));
            let _outer = BuiltinCallGuard::push(BuiltinCall { argc: 3, vectors });
            assert_eq!(builtin_argc(), 3);
            assert_eq!(builtin_vector_arg(0), Some(Vec3::X));
            {
                let _inner = BuiltinCallGuard::push(BuiltinCall {
                    argc: 1,
                    ..Default::default()
                });
                assert_eq!(builtin_argc(), 1);
                assert_eq!(builtin_vector_arg(0), None);
            }
            assert_eq!(builtin_argc(), 3);
        }
        assert_eq!(builtin_argc(), 0);
    }

    #[test]
    fn loading_checks_the_defs_crc() {
        struct OtherDefs;
        impl qcvm::QcDefs for OtherDefs {
            type GlobalAddr = qcvm::EmptyAddress;
            type FieldAddr = qcvm::EmptyAddress;
            type BuiltinAddr = qcvm::EmptyAddress;
            const CRC: i32 = 4321;
        }

        assert!(
            qcvm::QcVm::load_with_defs(
                std::io::Cursor::new(empty_progs()),
                &OtherDefs,
                Default::default()
            )
            .is_err()
        );
    }

    // Function values and callbacks through `bevy_mod_scripting` ==================================

    /// A hand-assembled `progs.dat` for [`EmptyDefs`] (so every global is stored in the VM):
    ///
    /// ```c
    /// float fortytwo = 42;
    /// float result;
    /// float() answer = { return fortytwo; };
    /// float() inner = { return fortytwo; };
    /// float(float() f) call_back = #0; // a builtin registered by name
    /// float() outer = { result = call_back(inner); return result; };
    /// ```
    fn function_progs() -> Vec<u8> {
        const OP_DONE: i16 = 0;
        const OP_STORE_F: i16 = 31;
        const OP_STORE_FNC: i16 = 36;
        const OP_RETURN: i16 = 43;
        const OP_CALL1: i16 = 52;
        const TYPE_FLOAT: u16 = 2;
        const TYPE_FUNCTION: u16 = 6;
        // 0..28 are the return value and parameters.
        const FORTYTWO: i16 = 28;
        const INNER_PTR: i16 = 29;
        const CALL_BACK_PTR: i16 = 30;
        const RESULT: i16 = 31;
        const NUM_GLOBALS: usize = 64;

        let statements: [[i16; 4]; 10] = [
            [OP_DONE, 0, 0, 0],
            // answer
            [OP_RETURN, FORTYTWO, 0, 0],
            [OP_DONE, 0, 0, 0],
            // inner
            [OP_RETURN, FORTYTWO, 0, 0],
            [OP_DONE, 0, 0, 0],
            // outer
            [OP_STORE_FNC, INNER_PTR, 4, 0],
            [OP_CALL1, CALL_BACK_PTR, 0, 0],
            [OP_STORE_F, 1, RESULT, 0],
            [OP_RETURN, RESULT, 0, 0],
            [OP_DONE, 0, 0, 0],
        ];

        let mut strings = vec![0u8];
        let mut string = |s: &str| {
            let id = strings.len() as i32;
            strings.extend_from_slice(s.as_bytes());
            strings.push(0);
            id
        };
        // (first statement, name, number of parameters, parameter sizes)
        let functions = [
            (0, string(""), 0, [0u8; 8]),
            (1, string("answer"), 0, [0; 8]),
            (3, string("inner"), 0, [0; 8]),
            // Builtin #1, which isn't known to `EmptyDefs` so it's looked up by name.
            (-1, string("call_back"), 1, [1, 0, 0, 0, 0, 0, 0, 0]),
            (5, string("outer"), 0, [0; 8]),
        ];
        let global_defs = [
            (TYPE_FLOAT, FORTYTWO, string("fortytwo")),
            (TYPE_FUNCTION, INNER_PTR, string("inner_ptr")),
            (TYPE_FUNCTION, CALL_BACK_PTR, string("call_back_ptr")),
            (TYPE_FLOAT, RESULT, string("result")),
        ];
        let file = string("test.qc");

        let mut globals = vec![[0u8; 4]; NUM_GLOBALS];
        globals[FORTYTWO as usize] = 42f32.to_le_bytes();
        globals[INNER_PTR as usize] = 2i32.to_le_bytes();
        globals[CALL_BACK_PTR as usize] = 3i32.to_le_bytes();

        let mut lumps: Vec<(Vec<u8>, usize)> = Vec::new();
        lumps.push((
            statements
                .iter()
                .flatten()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
            statements.len(),
        ));
        lumps.push((
            global_defs
                .iter()
                .flat_map(|(ty, offset, name)| {
                    [ty.to_le_bytes(), (*offset as u16).to_le_bytes()]
                        .concat()
                        .into_iter()
                        .chain(name.to_le_bytes())
                })
                .collect(),
            global_defs.len(),
        ));
        lumps.push((vec![], 0));
        lumps.push((
            functions
                .iter()
                .flat_map(|(first, name, argc, sizes)| {
                    // first statement, parm start, locals, profile, name, file, argc, sizes
                    [*first, 40, 0, 0, *name, file, *argc]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .chain(*sizes)
                        .collect::<Vec<_>>()
                })
                .collect(),
            functions.len(),
        ));
        lumps.push((strings.clone(), strings.len()));
        lumps.push((globals.iter().flatten().copied().collect(), globals.len()));

        const HEADER_SIZE: usize = 4 + 4 + 6 * 8 + 4;
        let mut header = Vec::new();
        header.extend(6i32.to_le_bytes());
        header.extend(<EmptyDefs as qcvm::QcDefs>::CRC.to_le_bytes());
        let mut body = Vec::new();
        for (bytes, count) in &lumps {
            header.extend(((HEADER_SIZE + body.len()) as i32).to_le_bytes());
            header.extend((*count as i32).to_le_bytes());
            body.extend_from_slice(bytes);
        }
        // Entity field count.
        header.extend(0i32.to_le_bytes());
        assert_eq!(header.len(), HEADER_SIZE);
        header.extend(body);
        header
    }

    /// The namespace the builtins of [`function_app`] are registered in.
    struct TestBuiltins;

    /// An app with a VM running [`function_progs`], as `bevy_mod_scripting` would have loaded it,
    /// and the attachment it was loaded for.
    fn function_app() -> (bevy_app::App, ScriptAttachment) {
        let mut app = bevy_app::App::new();
        app.add_plugins((
            bevy_asset::AssetPlugin::default(),
            bevy_mod_scripting_core::BMSScriptingInfrastructurePlugin::default(),
            QcScriptingPlugin::default(),
        ));
        app.finish();
        app.cleanup();

        let world = app.world_mut();
        bevy_mod_scripting_bindings::NamespaceBuilder::<TestBuiltins>::new_unregistered(world)
            .register(
                "call_back",
                |ctx: FunctionCallContext, f: DynamicScriptFunction| -> Result<f32, InteropError> {
                    // Calls back into the VM that is running this builtin.
                    match f.call([], ctx)? {
                        ScriptValue::Float(f) => Ok(f as f32),
                        other => Err(InteropError::string(format!("Not a float: {other:?}"))),
                    }
                },
            );

        let worldspawn = world.spawn((QcWorldspawn, QcEntity)).id();
        let attachment = ScriptAttachment::EntityScript(worldspawn, bevy_asset::Handle::default());
        let config = QcVmConfig::new::<TestBuiltins>(&EmptyDefs);
        let vm = qcvm::QcVm::load_with_defs(
            std::io::Cursor::new(function_progs()),
            &*config.defs,
            Default::default(),
        )
        .unwrap();
        let mut context = QcVmContext::new(vm, worldspawn, &config);
        context.attachment = Some(attachment.clone());
        world
            .resource::<bevy_mod_scripting_core::script::ScriptContexts<QcScriptingPlugin>>()
            .write()
            .insert(
                attachment.clone(),
                Arc::new(parking_lot::Mutex::new(context)).into(),
            )
            .unwrap();
        (app, attachment)
    }

    fn context_of(world: &World, attachment: &ScriptAttachment) -> QcVmContext {
        world
            .resource::<bevy_mod_scripting_core::script::ScriptContexts<QcScriptingPlugin>>()
            .read()
            .get_context(attachment)
            .and_then(|context| context.as_loaded().cloned())
            .unwrap()
            .lock()
            .clone()
    }

    /// Runs the named callback through `bevy_mod_scripting`'s language-agnostic runners, as
    /// `run_script_callback` does.
    fn run_callback(
        world: &mut World,
        attachment: &ScriptAttachment,
        label: &str,
    ) -> Result<ScriptValue, InteropError> {
        let cache = <WorldGuard<'_> as WorldExtensions>::setup_cache(
            world,
            CurrentScriptAttachment::empty(),
        );
        let guard = WorldAccessGuard::new_exclusive(world, cache);
        let runners = guard.script_callback_runners().clone();
        let results = runners.run(
            guard,
            &bevy_mod_scripting_script::Recipients::from(attachment.clone()),
            label,
            vec![],
        )?;
        assert_eq!(results.len(), 1);
        results.into_iter().next().unwrap().1
    }

    #[test]
    fn callbacks_return_their_return_value() {
        let (mut app, attachment) = function_app();
        let value = run_callback(app.world_mut(), &attachment, "answer").unwrap();
        assert!(matches!(value, ScriptValue::Float(42.)), "{value:?}");
        // Functions that the progs don't define do nothing.
        let value = run_callback(app.world_mut(), &attachment, "on_script_loaded").unwrap();
        assert!(matches!(value, ScriptValue::Unit), "{value:?}");
    }

    #[test]
    fn function_values_can_be_called_while_their_vm_is_running() {
        let (mut app, attachment) = function_app();
        // `outer` passes `inner` to a builtin, which calls it while `outer` (run as a callback,
        // which locks the script context) is still running.
        let value = run_callback(app.world_mut(), &attachment, "outer").unwrap();
        assert!(matches!(value, ScriptValue::Float(42.)), "{value:?}");
    }

    #[test]
    fn function_values_keep_their_identity() {
        let (mut app, attachment) = function_app();
        let world = app.world_mut();
        let context = context_of(world, &attachment);
        with_world(world, || {
            let (ScriptValue::Function(a), ScriptValue::Function(b)) =
                (context.function_value(2).unwrap(), context.function_value(2).unwrap())
            else {
                panic!("Not functions");
            };
            assert_eq!(a, b);
            assert_eq!(a.name(), "inner");
            assert_eq!(context.function_index(&a), Some(2));
            assert_eq!(
                bms_to_qcvm(&ScriptValue::Function(a), &context).unwrap(),
                FunctionIndex(2).to_value()
            );
            assert!(matches!(context.function_value(0).unwrap(), ScriptValue::Unit));
            // Registered builtins are the registered function itself.
            let ScriptValue::Function(builtin) = context.function_value(3).unwrap() else {
                panic!("Not a function");
            };
            assert_eq!(builtin.name(), "call_back");

            // Called from outside of QuakeC, with the VM's attachment as the current one.
            let ScriptValue::Function(answer) = context.function_value(1).unwrap() else {
                panic!("Not a function");
            };
            let value = answer.call([], FUNCTION_CALL_CONTEXT).unwrap();
            assert!(matches!(value, ScriptValue::Float(42.)), "{value:?}");
        });
    }

    #[test]
    fn host_globals_are_found_by_name() {
        let vm = qcvm::QcVm::load_with_defs(
            std::io::Cursor::new(function_progs()),
            &EmptyDefs,
            Default::default(),
        )
        .unwrap();
        let config = QcVmConfig::new::<TestBuiltins>(&EmptyDefs)
            .with_host_globals(&["fortytwo", "not_defined"]);
        let context = QcVmContext::new(vm, Entity::PLACEHOLDER, &config);
        assert_eq!(
            context.host_globals(),
            [HostGlobal {
                name: "fortytwo".into(),
                offset: 28,
                type_: Type::Float
            }]
        );
        assert_eq!(context.initial_global(28).unwrap(), qcvm::Value::Float(42.));
    }

    #[test]
    fn replacing_the_vm_waits_for_running_quakec() {
        let load = || {
            qcvm::QcVm::load_with_defs(
                std::io::Cursor::new(function_progs()),
                &EmptyDefs,
                Default::default(),
            )
            .unwrap()
        };
        let config = QcVmConfig::new::<TestBuiltins>(&EmptyDefs);
        let context = QcVmContext::new(load(), Entity::PLACEHOLDER, &config);
        let old: *const qcvm::QcVm = &*context.vm();

        // Running QuakeC holds the VM like this (and calls from inside it lock it again).
        let running = context.vm();
        let _nested = context.vm();
        let reloaded = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                context.replace_vm(load());
                reloaded.store(true, std::sync::atomic::Ordering::SeqCst);
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!reloaded.load(std::sync::atomic::Ordering::SeqCst));
            drop(_nested);
            drop(running);
        });
        assert!(reloaded.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!std::ptr::eq(old, &*context.vm()));
    }

    #[test]
    fn replacing_the_vm_affects_every_clone() {
        let (mut app, attachment) = function_app();
        let world = app.world_mut();
        let context = context_of(world, &attachment);
        let clone = context.clone();
        let old = with_world(world, || context.function_value(1).unwrap());
        let vm = qcvm::QcVm::load_with_defs(
            std::io::Cursor::new(function_progs()),
            &EmptyDefs,
            Default::default(),
        )
        .unwrap();
        context.replace_vm(vm);
        assert!(std::ptr::eq(&*clone.vm(), &*context.vm()));
        // The old function values are forgotten, so new ones are made.
        let ScriptValue::Function(old) = old else {
            panic!("Not a function");
        };
        assert_eq!(clone.function_index(&old), None);
        let new = with_world(world, || clone.function_value(1).unwrap());
        assert!(matches!(new, ScriptValue::Function(new) if new != old));
    }
}
