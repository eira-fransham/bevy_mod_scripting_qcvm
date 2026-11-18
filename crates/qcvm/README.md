# `qcvm`

This is a clean-room [QuakeC](https://en.wikipedia.org/wiki/QuakeC) VM implementation,
designed to be integrated into the [Seismon](https://github.com/eira-fransham/seismon)
engine. Unlike most other implementations, it is designed with embedding in mind, and
is not tied to only being used for Quake-like games.

## FAQ

#### Is this the fastest QuakeC VM in the world

It is almost certainly not going to get best-in-class performance for now, as it is
not designed with that in mind. Seismon is built for extensibility and modability
first and efficiency second, and this VM adopts the same mindset.

#### Why did you make this

Because I want a VM that is resilient enough that hobbyist game developers and modders
can mess around with a repl, override functions, just generally treat the code like
it's a rockstar's hotel room and still have the game engine generally respond in a
reasonable way.

#### Is this a meme

No, I actually have a goal in mind and am taking this project seriously. You could
use this to write a web server in QuakeC, though, and I do think that's very funny.

#### Should I write my game in QuakeC then?

Not unless you want to be sectioned.

## Design

The VM is designed with no specific global and builtin definitions in mind. The
globals, fields and builtins that a host knows about come from a `defs.qc`, and are
described by an implementation of the `QCDefs` trait, which bundles the global and
field address types (implementing `Address`), the builtin type (implementing
`BuiltinAddress`) and the `progdefs.h` CRC that `progs.dat`s compiled against those
defs carry. `progs.dat`s are loaded with `QCVm::load::<Defs, _>`, which rejects
`progs.dat`s with a different CRC. Quake 1-compatible defs are provided by the
sister crate `qcvm-defs-quake1`, and other `defs.qc`s (e.g. of engines with
extensions) can be provided by other `qcvm-defs-*` crates, or the user can define
a completely new set of definitions that is entirely unrelated to Quake 1's. The
`BuiltinAddress` trait maps builtin numbers (`#n` in QuakeC) to canonical names, and
`qcvm_defs_quake1::BuiltinAddr` implements it for the vanilla Quake builtins. However, when resolving a builtin the host is
still provided with the raw `BuiltinDef` (name, index and builtin number) read from the
`progs.dat` (the filename of compiled QuakeC bytecode), and can use
`BuiltinDef::builtin_addr` to look up the typed builtin.

The host is intended to manage storage for globals and fields, although any
host-unknown globals will be managed internally by the VM. This is to allow mods to
add their own internal shared state, which is a common pattern in QuakeC.

Values in `qcvm` are full managed values, allowing arbitrary numbers of temp strings
and for external functions to be used rather than just function indices. This bloats
the storage size, however, and so in the future a more-efficient encoding mechanism
will be used.

The vanilla QuakeC VM, and most of its derivatives, makes no distinction between
function locals and global values. However, compiled QuakeC bytecode _does_ make a
distinction. To that end, we try to isolate locals of functions from one another, to
prevent a function from overwriting data of another. This makes the behaviour more
reliable, makes the VM easier to implement, and opens the door for further
optimization in the future.

There is one opcode in QuakeC, `OP_STATE`, which makes assumptions about the layout
of globals and entity fields. In order to avoid a strict reliance on Quake 1's entity
layout, this is implemented by the host. It's unclear when or if the QuakeC compiler
actually emits this opcode, so a default implementation is provided that errors out.

## Future work

This VM could stand to be a lot more optimized. While the current implementation is
reasonable, it uses a lot more memory than necessary and uses dynamic dispatch for
interacting with the host. Both of these issues would be relatively straightforward
to resolve.

On a related note, it should be achievable to write a JIT for the bytecode by using
[Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift). Other
projects (notably, [FTEQW](https://github.com/fte-team/fteqw/blob/master/engine/qclib/pr_x86.c))
have written hand-rolled JITs for QuakeC, but they are non-optimizing and locked to
a single architecture. A Cranelift-based JIT would avoid these issues, without
restricting the API surface.
