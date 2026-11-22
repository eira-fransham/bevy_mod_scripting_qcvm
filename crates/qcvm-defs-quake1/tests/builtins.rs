use qcvm::{BuiltinAddress as _, QcVm};
use qcvm_defs_quake1::{BuiltinAddr, Quake1};

#[test]
fn quake1_builtin_names_are_by_number() {
    let builtin_name = |n| BuiltinAddr::from_number(n).map(|b| b.name());

    assert_eq!(builtin_name(1), Some("makevectors"));
    assert_eq!(builtin_name(5), None);
    assert_eq!(builtin_name(13), Some("vectoyaw"));
    assert_eq!(builtin_name(14), Some("spawn"));
    assert_eq!(builtin_name(45), Some("cvar"));
    assert_eq!(builtin_name(48), Some("particle"));
    assert_eq!(builtin_name(0), None);
    assert_eq!(builtin_name(83), None);
}

/// Loads the real `progs.dat` from the path in `QcVM_PROGS`. If it isn't set, the test is skipped.
fn real_progs_vm() -> Option<QcVm> {
    let Some(path) = std::env::var_os("QcVM_PROGS") else {
        eprintln!("Skipping test: `QcVM_PROGS` isn't set");
        return None;
    };
    let bytes = std::fs::read(path).unwrap();
    Some(QcVm::load::<Quake1, _>(std::io::Cursor::new(bytes)).unwrap())
}

/// Every builtin in a real `progs.dat` (see [`real_progs_vm`]) must be known to [`BuiltinAddr`],
/// as optimizing compilers strip builtin names so builtins can only be resolved by number.
#[test]
fn real_progs_builtins_are_known() {
    let Some(vm) = real_progs_vm() else {
        return;
    };

    let mut unknown = vm
        .functions()
        .iter()
        .filter_map(|def| def.builtin_number())
        .filter(|&n| BuiltinAddr::from_number(n).is_none())
        .collect::<Vec<_>>();
    unknown.sort_unstable();
    unknown.dedup();
    assert!(unknown.is_empty(), "unknown builtin numbers: {unknown:?}");
}
