# `bevy_mod_scripting_qcvm`

This is bindings to my [`qcvm`](https://crates.io/crates/qcvm) crate for
[`bevy_mod_scripting`](https://crates.io/crates/bevy_mod_scripting). This allows
writing game code for Bevy games in QuakeC. It is built as a component of the
[Seismon](https://github.com/eira-fransham/seismon) engine, and intende to
support its needs. The QuakeC language is pretty terrible, so using it for your
own games outside of the context of Seismon is a bad idea.

## Usage

See [the `bevy_mod_scripting` docs](https://makspll.github.io/bevy_mod_scripting/)
for basic usage of that library.

Add `QCScriptingPlugin::default()`. Each QuakeC VM is a `progs.dat` attached
(with `AttachScript`) to its own worldspawn entity, which must have a
`QCVmConfig`. Several VMs can run in the same world, e.g. a server's `progs.dat`
and a client's `csprogs.dat`. The config says:

- `defs` - the `QCDefs` that the `progs.dat` is compiled against, type-erased as
  an `&dyn qcvm::ErasedDefs`.
  `progs.dat`s with a different CRC fail to load.
- `builtin_namespace` - the `NamespaceBuilder` namespace that builtin functions
  are looked up in. If the builtin number (`#n`) is known to the defs, the
  canonical name from the defs is used, so renamed or aliased builtins still
  resolve. Otherwise (e.g. extension builtins), the name defined in the loaded
  `progs.dat` is used. Builtins shared between VMs can be registered in each
  VM's namespace.
- `options` - the `qcvm::VmOptions` that the `progs.dat` is loaded with.

Globals and fields are accessed with the `get` and `set` magic functions, by
name:

- `QCWorldspawn` - globals are looked up on references to this component of the
  VM's worldspawn.
- `QCEntity` - fields are looked up on references to this component of an
  entity. The intention is that different fields will access different
  components, for example the `origin` field can access the [`Transform`](https://docs.rs/bevy/latest/bevy/transform/components/struct.Transform.html)
  component.

Entities belong to a single VM: every entity other than worldspawn has a
`QCOwnedBy` relationship to its VM's worldspawn, and QuakeC can't access
entities of other VMs. Despawning a worldspawn despawns all of its VM's
entities. Builtins and magic functions can find the VM that called them with
`current_vm()`, and the VM that an entity belongs to with `vm_root()`.

`OP_STATE` is implemented as Quake does, by setting the `nextthink`, `frame` and
`think` fields of `self`.

## Sources

Writing a new language for `bevy_mod_scripting` is currently poorly documented,
and so [the Rhai bindings](https://github.com/makspll/bevy_mod_scripting/blob/main/crates/languages/bevy_mod_scripting_rhai/src/lib.rs)
were used as reference. QuakeC is a far, far simpler language than Rhai, and
it is a specific non-goal to support `bevy_mod_scripting`'s APIs for interacting
with the ECS, so a lot of issues with implementing a new BMS language are
avoided.
