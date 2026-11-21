//! # `bevy_mod_scripting_qcvm`
//!
//! The main engine is in the `qcvm` crate. This is just bindings for `bevy_mod_scripting`.
//!
//! Several QuakeC VMs can run in the same world (e.g. a server's `progs.dat` and a client's
//! `csprogs.dat`). Each VM is a script attached to its own worldspawn entity, which must have a
//! [`QCVmConfig`] saying which defs (e.g. `qcvm_defs_quake1::Quake1`) the `progs.dat` is compiled
//! against and which namespace its builtins are registered in. Entities belong to a single VM,
//! through the [`QCOwnedBy`] relationship to its worldspawn, and VMs can't access each other's
//! entities.
//!
//! To add builtins to a VM, register callbacks with `NamespaceBuilder` in its
//! [`QCVmConfig::builtin_namespace`]. Builtins whose number is known to the defs (see
//! [`qcvm::ErasedDefs::builtin_name`]) are looked up by their canonical name, and other builtins
//! are looked up by the name in the `progs.dat`. Fields are accessed using the `get` and `set` of
//! `MagicFunctions`, where globals are accessed using `QCWorldspawn` reflect references and fields
//! are accessed using `QCEntity`. In the rare case that fields need to be accessed on worldspawn
//! (which, unfortunately, is allowed by Quake), `QCEntity` is still used. Builtins and magic
//! functions can find the VM that is running with [`current_vm`].
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
    cell::RefCell,
    collections::VecDeque,
    ffi::{CStr, CString},
    fmt,
    num::NonZero,
    str::FromStr as _,
    sync::{Arc, RwLock},
};

use bevy_ecs::{
    component::Component, entity::Entity, storage::SparseSetIndex as _, world::World,
    world::WorldId,
};
use bevy_math::Vec3;
use bevy_mod_scripting_asset::Language;
use bevy_mod_scripting_bindings::{
    CurrentScriptAttachment, DynamicScriptFunction, ExternalError, FunctionCallContext,
    InteropError, ReflectBase, ReflectReference, ScriptValue, ThreadScriptContext,
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
    FunctionRef, QCParams, Type, VectorField, anyhow,
    arrayvec::ArrayVec,
    userdata::{AddrError, ErasedContext, ErasedFunction, FnCall, QCType},
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
/// [`QCVmConfig`] of the entity its `progs.dat` is attached to.
pub struct QCScriptingPlugin {
    /// The inner config for the plugin
    pub scripting_plugin: ScriptingPlugin<QCScriptingPlugin>,
}

impl Default for QCScriptingPlugin {
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
pub struct QCVmConfig {
    /// The defs that the `progs.dat` is compiled against, e.g. `qcvm_defs_quake1::Quake1`.
    /// `progs.dat`s with a different CRC fail to load.
    pub defs: &'static dyn ErasedDefs,
    /// The namespace (as for `NamespaceBuilder`, i.e. the [`TypeId`] of a type) that builtins are
    /// looked up in.
    pub builtin_namespace: TypeId,
    /// The options that the `progs.dat` is loaded with (and reloaded with). The defaults are
    /// strict; see [`qcvm::VmOptions`].
    pub options: qcvm::VmOptions,
}

impl fmt::Debug for QCVmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QCVmConfig")
            .field("defs", &self.defs)
            .field("builtin_namespace", &self.builtin_namespace)
            .field("options", &self.options)
            .finish()
    }
}

impl QCVmConfig {
    /// A configuration for `progs.dat`s compiled against `defs`, with builtins registered in the
    /// namespace `N` and the default (strict) options.
    pub fn new<N: 'static>(defs: &'static dyn ErasedDefs) -> Self {
        Self {
            defs,
            builtin_namespace: TypeId::of::<N>(),
            options: Default::default(),
        }
    }

    /// Sets the options that the `progs.dat` is loaded with.
    pub fn with_options(mut self, options: qcvm::VmOptions) -> Self {
        self.options = options;
        self
    }
}

/// The [`QCVmConfig`] of `worldspawn`, in the world that a context is being loaded in.
fn vm_config(worldspawn: Entity) -> Result<QCVmConfig, InteropError> {
    thread_context()?
        .world
        .with_world_access(|world| world.get::<QCVmConfig>(worldspawn).cloned())?
        .ok_or_else(|| {
            InteropError::external_boxed(
                format!(
                    "QuakeC scripts must be attached to an entity with a `QCVmConfig`, but                      {worldspawn} has none"
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
struct BevyScriptContext {
    /// Per-execution globals such as `self` and `other`
    special_args: hashbrown::HashMap<u16, qcvm::Value>,
    context: QCVmContext,
}

impl BevyScriptContext {
    fn new(context: QCVmContext) -> Self {
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
pub struct QCEntity;

/// Marker component for the worldspawn entity of a QuakeC VM, where globals will be stored.
///
/// Worldspawn is the root of its VM: every other entity of the VM is [`QCOwnedBy`] it.
#[derive(Component, Reflect, Default)]
pub struct QCWorldspawn;

/// The QuakeC VM that an entity belongs to, by its worldspawn.
///
/// Every [`QCEntity`] other than worldspawn itself must have this, and QuakeC can only access
/// entities that belong to its own VM. Despawning a worldspawn despawns every entity that it owns.
#[derive(Component, Reflect, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = QCVmEntities)]
pub struct QCOwnedBy(pub Entity);

/// The entities owned by a QuakeC VM's worldspawn (see [`QCOwnedBy`]).
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = QCOwnedBy, linked_spawn)]
pub struct QCVmEntities(Vec<Entity>);

impl QCVmEntities {
    /// The entities owned by the VM, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
}

/// The worldspawn of the QuakeC VM that `entity` belongs to: `entity` itself for worldspawns, and
/// the target of [`QCOwnedBy`] for other entities. Entities that don't belong to a VM return `None`.
pub fn vm_root(world: &World, entity: Entity) -> Option<Entity> {
    let entity_ref = world.get_entity(entity).ok()?;
    if entity_ref.contains::<QCWorldspawn>() {
        Some(entity)
    } else {
        entity_ref.get::<QCOwnedBy>().map(|owner| owner.0)
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
fn check_owner(entity: Entity, context: &QCVmContext) -> Result<(), InteropError> {
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

    fn get_global(&self, addr: u16, ty: Type) -> Result<qcvm::Value, AddrError<InteropError>> {
        let info = self
            .context
            .defs
            .global(addr, ty)
            .ok_or(AddrError::OutOfRange)?;
        if let Some(val) = self.special_args.get(&info.vector_root) {
            return Ok(val
                .field(info.vector_field)
                .map_err(|e| qc_interop_error(anyhow::format_err!("{e}")))?);
        }

        let key = ScriptValue::String(info.name.into());
        let value =
            interop_error_to_addr_error(magic_get::<QCWorldspawn>(self.context.worldspawn, key))?;

        interop_error_to_addr_error(bms_to_qcvm(&value, &self.context))
    }

    fn set_global(&mut self, addr: u16, value: qcvm::Value) -> Result<(), AddrError<InteropError>> {
        let info = self
            .context
            .defs
            .global(addr, lookup_type(&value))
            .ok_or(AddrError::OutOfRange)?;
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

        interop_error_to_addr_error(magic_set::<QCWorldspawn>(
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
        interop_error_to_addr_error(magic_set::<QCEntity>(
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
            interop_error_to_addr_error(magic_get::<QCEntity>(self.entity(erased_ent), key))
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

        interop_error_to_addr_error(magic_set::<QCEntity>(self.entity(erased_ent), key, value))
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
    }
}

struct BevyScriptArgs<'a, A>
where
    A: ?Sized,
{
    args: &'a A,
    context: QCVmContext,
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

impl<'a> QCParams for BevyScriptArgs<'a, [ScriptValue]> {
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

impl<'a> QCParams for BevyScriptArgs<'a, VecDeque<ScriptValue>> {
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

impl QCType for BevyBuiltin {
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
    } else if is::<QCString>(type_id) {
        qcvm::Type::String
    } else if is::<QCFunction>(type_id) {
        qcvm::Type::Function
    } else if is::<QCField>(type_id) {
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

fn bms_to_qcvm(
    bms_value: &ScriptValue,
    context: &QCVmContext,
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
                && r.base.type_id() == TypeId::of::<QCEntity>() =>
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
        ScriptValue::Function(func) => Ok(qcvm::Value::Function(qcvm::Function::Host(Arc::new(
            BevyBuiltin { func: func.clone() },
        )))),
        ScriptValue::Error(e) => Err(e.clone()),
        _ => Err(InteropError::NotImplemented),
    }
}

fn qcvm_to_bms(
    qcvm_value: &qcvm::Value,
    root_ctx: &QCVmContext,
) -> Result<ScriptValue, InteropError> {
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
        qcvm::Value::Function(qcvm::Function::Index(index)) => Ok(ScriptValue::Reference(
            ReflectReference::new_allocated(QCFunction(index.0)),
        )),
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
type BuiltinCache = Arc<RwLock<hashbrown::HashMap<i32, Arc<dyn ErasedFunction>>>>;

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
    /// The defs that the `progs.dat` is compiled against (see [`QCVmConfig::defs`]).
    pub defs: &'static dyn ErasedDefs,
    /// The namespace that builtins are looked up in (see [`QCVmConfig::builtin_namespace`]).
    pub builtin_namespace: TypeId,
    builtins: BuiltinCache,
}

impl fmt::Debug for QCVmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QCVmContext")
            .field("vm", &self.vm)
            .field("worldspawn", &self.worldspawn)
            .field("special_args_immutable", &self.special_args_immutable)
            .field("defs", &self.defs)
            .finish_non_exhaustive()
    }
}

impl QCVmContext {
    /// Creates a context for `vm`, with globals stored on `worldspawn` and the defs and builtins of
    /// `config`.
    pub fn new(vm: Arc<qcvm::QCVm>, worldspawn: Entity, config: &QCVmConfig) -> Self {
        Self {
            vm,
            reflect_to_value: default_reflect_to_value,
            worldspawn,
            special_args_immutable: false,
            defs: config.defs,
            builtin_namespace: config.builtin_namespace,
            builtins: Default::default(),
        }
    }

    /// Calls a QuakeC function with exclusive access to `world`.
    ///
    /// This can be used both from regular systems and from inside builtins (e.g. to run touch
    /// functions from a movement builtin), as it doesn't go through `bevy_mod_scripting`'s context
    /// handling: it creates a fresh world access guard for `world`, which is used by field and
    /// global accessors and builtins until this returns.
    ///
    /// The progs don't say what type functions return, so the return value is read as `R` (see
    /// [`qcvm::ReturnValue`]). Use `()` to ignore it, which never fails.
    ///
    /// While the function runs, [`current_vm`] is this context's worldspawn.
    pub fn call_with_world<R>(
        &self,
        world: &mut World,
        function: impl Into<FunctionRef>,
        args: impl QCParams,
    ) -> Result<R, InteropError>
    where
        R: qcvm::FromReturnValue,
    {
        let cache = <WorldGuard<'_> as WorldExtensions>::setup_cache(
            world,
            CurrentScriptAttachment::empty(),
        );
        WorldAccessGuard::with_static_guard(world, cache, |_guard| {
            let _vm = VmStackGuard::push(self.worldspawn);
            let mut context = BevyScriptContext::new(self.clone());
            self.vm
                .run(&mut context, function, args)
                .and_then(|ret| ret.get())
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

    /// Whether the function with index `index` is a builtin that is registered in this VM's
    /// builtin namespace (e.g. for FTEQW's `checkbuiltin`).
    pub fn has_builtin(&self, world: &World, index: i32) -> bool {
        let Ok(def) = self.vm.functions().get_by_index(index) else {
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

    /// The QuakeC value for a reference to `entity` (`world` for the worldspawn), e.g. to pass to
    /// [`QCVmContext::set_vm_global`].
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
    /// rather than by the host (i.e. it isn't in [`QCVmContext::defs`]). These are globals that
    /// the `progs.dat` defines for itself, including engine extension globals that the host's
    /// defs don't cover.
    pub fn vm_global_def(&self, name: &str) -> Option<&qcvm::GlobalDef> {
        self.global_defs().iter().find(|def| {
            def.name.to_bytes() == name.as_bytes()
                && self
                    .defs
                    .global(def.offset, qcvm::Type::AnyScalar)
                    .is_none()
        })
    }

    /// The value of a global stored in the VM by name (see [`QCVmContext::vm_global_def`]).
    pub fn vm_global(&self, name: &str) -> Option<anyhow::Result<qcvm::Value>> {
        let def = self.vm_global_def(name)?;
        Some(if matches!(def.type_, qcvm::VmType::Vector) {
            self.vm.global_vector(def.offset).map(qcvm::Value::Vector)
        } else {
            self.vm.global(def.offset)
        })
    }

    /// Sets a global stored in the VM by name (see [`QCVmContext::vm_global_def`]). Returns
    /// `Ok(false)` if the `progs.dat` doesn't define it.
    pub fn set_vm_global(&self, name: &str, value: qcvm::Value) -> anyhow::Result<bool> {
        let Some(def) = self.vm_global_def(name) else {
            return Ok(false);
        };
        let offset = def.offset;
        self.vm.set_global(offset, value)?;
        Ok(true)
    }
}

impl IntoScriptPluginParams for QCScriptingPlugin {
    const LANGUAGE: Language = QC_LANGUAGE;

    type C = QCVmContext;
    type R = ();

    fn build_runtime() -> Self::R {}

    fn handler() -> bevy_mod_scripting_core::handler::HandlerFn<Self> {
        fn qc_callback_handler(
            mut args: Vec<ScriptValue>,
            context_key: &ScriptAttachment,
            callback: &CallbackLabel,
            context: &mut QCVmContext,
            world_id: WorldId,
        ) -> Result<ScriptValue, InteropError> {
            let config = QCScriptingPlugin::readonly_configuration(world_id);
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
            if context.vm.functions().get_by_name(&callback_cstr).is_err() {
                return Ok(ScriptValue::Unit);
            }

            let mut script_context = BevyScriptContext::new(context.clone());
            script_context.special_args = special_args;
            let _vm = VmStackGuard::push(context.worldspawn);

            // The progs don't say what type callbacks return, so their return values are ignored.
            context
                .vm
                .run(
                    &mut script_context,
                    callback_cstr,
                    BevyScriptArgs::<[ScriptValue]> {
                        context: context.clone(),
                        args: args_ref,
                    },
                )
                .map_err(qc_interop_error)?;

            Ok(ScriptValue::Unit)
        }

        qc_callback_handler
    }

    fn context_loader() -> bevy_mod_scripting_core::context::ContextLoadFn<Self> {
        fn qc_context_loader(
            context_key: &ScriptAttachment,
            content: &[u8],
            world_id: WorldId,
        ) -> Result<QCVmContext, InteropError> {
            let config = QCScriptingPlugin::readonly_configuration(world_id);
            // Globals are stored on the entity the script is attached to, which also has the
            // configuration of the VM.
            let worldspawn = context_key.entity().ok_or_else(|| {
                InteropError::external_boxed(
                    "QuakeC scripts must be attached to a worldspawn entity".into(),
                )
            })?;
            let vm_config = vm_config(worldspawn)?;
            let vm = qcvm::QCVm::load_with_defs(
                std::io::Cursor::new(content),
                &*vm_config.defs,
                vm_config.options.clone(),
            )
            .map_err(qc_interop_error)?;
            let mut out = QCVmContext::new(vm.into(), worldspawn, &vm_config);

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
            _context_key: &ScriptAttachment,
            content: &[u8],
            previous_context: &mut QCVmContext,
            _world_id: WorldId,
        ) -> Result<(), InteropError> {
            // Keep the defs and options that the context was loaded with.
            let options = previous_context.vm.options().clone();
            previous_context.vm = qcvm::QCVm::load_with_defs(
                std::io::Cursor::new(content),
                &*previous_context.defs,
                options,
            )
            .map_err(qc_interop_error)?
            .into();
            previous_context.builtins = Default::default();

            Ok(())
        }

        qc_context_reloader
    }
}

/// The plugin configuration of [`QCScriptingPlugin`], by world.
type PluginConfigs = Vec<Option<ScriptingPluginConfiguration<QCScriptingPlugin>>>;

static PLUGIN_CONFIGS: RwLock<PluginConfigs> = RwLock::new(Vec::new());

impl GetPluginThreadConfig<QCScriptingPlugin> for QCScriptingPlugin {
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

impl bevy_app::Plugin for QCScriptingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.register_type::<QCEntity>()
            .register_type::<QCWorldspawn>()
            .register_type::<QCOwnedBy>();
        self.scripting_plugin.build(app);
    }

    fn finish(&self, app: &mut bevy_app::App) {
        self.scripting_plugin.finish(app);
    }
}

impl AsMut<ScriptingPlugin<Self>> for QCScriptingPlugin {
    fn as_mut(&mut self) -> &mut ScriptingPlugin<Self> {
        &mut self.scripting_plugin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defs with no globals, fields or builtins.
    struct EmptyDefs;

    impl qcvm::QCDefs for EmptyDefs {
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
        bytes.extend(<EmptyDefs as qcvm::QCDefs>::CRC.to_le_bytes());
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

    fn context(worldspawn: Entity) -> QCVmContext {
        let config = QCVmConfig::new::<QCWorldspawn>(&EmptyDefs);
        let vm = qcvm::QCVm::load_with_defs(
            std::io::Cursor::new(empty_progs()),
            &*config.defs,
            Default::default(),
        )
        .unwrap();
        QCVmContext::new(vm.into(), worldspawn, &config)
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
            ReflectReference::new_component_ref::<QCEntity>(entity, world).unwrap(),
        )
    }

    #[test]
    fn vms_only_accept_their_own_entities() {
        let mut world = World::new();
        let a = world.spawn((QCWorldspawn, QCEntity)).id();
        let b = world.spawn((QCWorldspawn, QCEntity)).id();
        let a_ent = world.spawn((QCEntity, QCOwnedBy(a))).id();
        let b_ent = world.spawn((QCEntity, QCOwnedBy(b))).id();
        let unowned = world.spawn(QCEntity).id();

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
        let root = world.spawn((QCWorldspawn, QCEntity)).id();
        let owned = world.spawn((QCEntity, QCOwnedBy(root))).id();
        let unowned = world.spawn(QCEntity).id();

        assert_eq!(vm_root(&world, root), Some(root));
        assert_eq!(vm_root(&world, owned), Some(root));
        assert_eq!(vm_root(&world, unowned), None);
        assert_eq!(
            world
                .get::<QCVmEntities>(root)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [owned]
        );
    }

    #[test]
    fn despawning_worldspawn_only_despawns_its_entities() {
        let mut world = World::new();
        let a = world.spawn((QCWorldspawn, QCEntity)).id();
        let b = world.spawn((QCWorldspawn, QCEntity)).id();
        let a_ent = world.spawn((QCEntity, QCOwnedBy(a))).id();
        let b_ent = world.spawn((QCEntity, QCOwnedBy(b))).id();

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
        impl qcvm::QCDefs for OtherDefs {
            type GlobalAddr = qcvm::EmptyAddress;
            type FieldAddr = qcvm::EmptyAddress;
            type BuiltinAddr = qcvm::EmptyAddress;
            const CRC: i32 = 4321;
        }

        assert!(
            qcvm::QCVm::load_with_defs(
                std::io::Cursor::new(empty_progs()),
                &OtherDefs,
                Default::default()
            )
            .is_err()
        );
    }
}
