use std::collections::HashSet;

use qcvm::{BuiltinAddress, ErasedDefs, strum::VariantArray};
use qcvm_defs_csqc::{BuiltinAddr, Csqc, builtins::NAMED_BUILTINS};

#[test]
fn numbers_and_names_round_trip() {
    for addr in BuiltinAddr::VARIANTS {
        assert_eq!(BuiltinAddr::from_number(addr.number()), Some(*addr));
        assert_eq!(BuiltinAddr::from_name(addr.name()), Some(*addr));
        assert_eq!(Csqc.builtin_name(addr.number()), Some(addr.name()));
    }
}

#[test]
fn shared_builtins_have_quake1_names() {
    for addr in qcvm_defs_quake1::BuiltinAddr::VARIANTS {
        if let Some(csqc) = BuiltinAddr::from_number(addr.number()) {
            assert_eq!(csqc.name(), addr.name(), "#{}", addr.number());
        }
    }
}

#[test]
fn names_are_unique() {
    let mut names = HashSet::new();
    for name in BuiltinAddr::VARIANTS
        .iter()
        .map(|addr| addr.name())
        .chain(NAMED_BUILTINS.iter().copied())
    {
        assert!(names.insert(name), "{name} is listed twice");
    }
    assert_eq!(names.len(), 289);
}
