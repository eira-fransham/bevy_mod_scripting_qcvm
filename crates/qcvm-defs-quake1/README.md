# `qcvm-defs-quake1`

The globals, fields and builtins of the Quake GPL release (`progdefs.q1` and id1's `defs.qc`),
for use with [`qcvm`](../qcvm). The `Quake1` type implements `qcvm::QCDefs`, so hosts that run
unmodified Quake `progs.dat`s can be generic over it, e.g.
`QCVm::load::<qcvm_defs_quake1::Quake1, _>(bytes)`.

Other `defs.qc`s (e.g. those of NetQuake or FTEQW extensions) should be defined in their own
`qcvm-defs-*` crates following the same pattern: one enum each implementing `qcvm::Address`
for globals and fields and `qcvm::BuiltinAddress` for builtins, plus a marker type implementing
`qcvm::QCDefs` with the matching `progdefs.h` CRC.
