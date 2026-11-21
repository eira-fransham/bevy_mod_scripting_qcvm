//! [`qcvm::ErasedDefs`] for [`Quake1`] agrees with its typed addresses.

use qcvm::{Address, BuiltinAddress, ErasedAddrInfo, ErasedDefs, strum::VariantArray};
use qcvm_defs_quake1::{BuiltinAddr, FieldAddr, GlobalAddr, Quake1};

#[test]
fn erased_globals_match_typed() {
    let defs: &dyn ErasedDefs = &Quake1;
    for addr in GlobalAddr::VARIANTS {
        let info = ErasedAddrInfo::new(addr);
        assert_eq!(
            defs.global(addr.to_u16(), addr.type_()),
            Some(info),
            "{}",
            addr.name()
        );
        assert_eq!(defs.global_by_name(addr.name()), Some(info));
    }
}

#[test]
fn erased_fields_match_typed() {
    let defs: &dyn ErasedDefs = &Quake1;
    for addr in FieldAddr::VARIANTS {
        let info = ErasedAddrInfo::new(addr);
        assert_eq!(
            defs.field(addr.to_u16(), addr.type_()),
            Some(info),
            "{}",
            addr.name()
        );
        assert_eq!(defs.field_by_name(addr.name()), Some(info));
    }
}

#[test]
fn erased_builtins_match_typed() {
    let defs: &dyn ErasedDefs = &Quake1;
    assert_eq!(defs.crc(), 5927);
    for addr in BuiltinAddr::VARIANTS {
        assert_eq!(defs.builtin_name(addr.number()), Some(addr.name()));
    }
    assert_eq!(defs.builtin_name(5), None);
}
