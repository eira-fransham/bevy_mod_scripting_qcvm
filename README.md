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

- `QCBuiltin` - builtin functions will be looked up here, using the name defined
  in the loaded `progs.dat`.
- `QCWorldspawn` - globals will be looked up here using the `get` and `set` magic
  functions, by name. The intention is that the world is an entity with the script
  attached, rather than a resource, although this may change in the future.
- `QCEntity` - globals will be looked up here using the `get` and `set` magic
  functions, by name. The intention is that different fields will access different
  components, for example the `origin` field can access the [`Transform`](https://docs.rs/bevy/latest/bevy/transform/components/struct.Transform.html)
  component.

`qcvm` uses a separate mechanism for `OP_STATE`, intended to be implemented by the
host. At this time, `bevy_mod_scripting_qcvm` does not provide any way to configure
this implementation, so any use of that opcode will simply cause an error.

## Sources

Writing a new language for `bevy_mod_scripting` is currently poorly documented,
and so [the Rhai bindings](https://github.com/makspll/bevy_mod_scripting/blob/main/crates/languages/bevy_mod_scripting_rhai/src/lib.rs)
were used as reference. QuakeC is a far, far simpler language than Rhai, and
it is a specific non-goal to support `bevy_mod_scripting`'s APIs for interacting
with the ECS, so a lot of issues with implementing a new BMS language are
avoided.
