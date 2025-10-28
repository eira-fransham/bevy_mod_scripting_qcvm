# `bevy_mod_scripting_qcvm`

This is a (still unfinished) rewrite of the QuakeC runtime from
[Seismon](https://github.com/eira-fransham/seismon) to be a reusable module that can be included
like any `bevy_mod_scripting` language. The main part of the design that deviates is the use of
 managed values instead of every value being an index. It's not possible to entirely avoid
indices, since that's ultimately what the format uses internally, but QuakeC is surprisingly
strongly-typed and so passing (for example) function pointers or opaque entity references from
the host is pretty simple.

The main goal is to externalise entity storage and fields. The fields of the entities should be
runtime-definable, and ultimately I want to have fields and builtins be defined in a scripting
language such as Lua. This may sound like a strange goal to have, but there are two motivations:

- Transparent multi-variant support: FTEQW-only mods, Darkplaces-only mods and mods for some strange
  variant I haven't even heard of should be supportable without changing the core engine.
- Runtime redefinition: As Seismon is designed for hackability and moddability, it should be trivial
  to change things about the engine on-the-fly, and for mods to override functionality that would
  usually require engine changes.
