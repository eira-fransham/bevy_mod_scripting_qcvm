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

The namespaces used are:

- `QCWorldspawn` - builtin functions are registered in this namespace, and are looked up
  using the name defined in the loaded `progs.dat`. Globals are accessed through the `get` and
  `set` magic functions on a reference to the `QCWorldspawn` component, by name. The intention
  is that the world is an entity with the script attached, rather than a resource.
- `QCEntity` - fields are accessed through the `get` and `set` magic functions on a reference
  to the `QCEntity` component, by name. Fields defined by the `progs.dat` rather than the field
  address type (e.g. `pos1` in Quake) use their name from the `progs.dat`, and the components of
  vector fields are `{name}_x` etc.

Values cross into and out of QuakeC as follows:

- Floats are `ScriptValue::Float` and vectors are a `ScriptValue::List` of three floats.
- Entities are references to their `QCEntity` component (worldspawn included).
- Strings are references to a `QCString`, which keeps the raw bytes (Quake strings aren't
  necessarily UTF-8) and distinguishes the null string from other empty strings.
- Functions defined by the `progs.dat` are references to a `QCFunction`, i.e. their index in the
  function table.
- Field references (e.g. the field argument of Quake's `find`) are references to a `QCField`,
  i.e. their offset.

`OP_STATE` is implemented in terms of the `self` and `time` globals and the `nextthink`,
`frame` and `think` fields.

To call QuakeC from Rust, including from inside a builtin, use `QCVmContext::call_with_world`.
The `progs.dat` doesn't say what type functions return, so the caller chooses what to read the
return value as (`()` to ignore it). For the same reason, callbacks run through
`bevy_mod_scripting` always return `ScriptValue::Unit`.

`progs.dat`s are loaded with `QCScriptingPlugin::vm_options` (a `qcvm::VmOptions`, also available
as the `QCVmOptions` resource), and reloaded with the options they were loaded with. The defaults
are strict, so that bugs in QuakeC are errors. Some of the original Quake QuakeC is buggy and relies
on quirks of Quake's VM, so to run it unmodified, opt into those quirks, e.g.
`MissingArguments::Stale` for functions called with fewer arguments than they take.

## Sources

Writing a new language for `bevy_mod_scripting` is currently poorly documented,
and so [the Rhai bindings](https://github.com/makspll/bevy_mod_scripting/blob/main/crates/languages/bevy_mod_scripting_rhai/src/lib.rs)
were used as reference. QuakeC is a far, far simpler language than Rhai, and
it is a specific non-goal to support `bevy_mod_scripting`'s APIs for interacting
with the ECS, so a lot of issues with implementing a new BMS language are
avoided.
