//! Tests for the VM, using hand-assembled programs.

use std::{
    ffi::{CStr, CString},
    fmt,
    num::NonZero,
    sync::Arc,
};

use arrayvec::ArrayVec;
use strum::VariantArray;

use crate::{
    Address, EmptyAddress, EntityRef, EntityTypeDef, FieldDef, FieldRef, FunctionIndex, GlobalDef,
    MAX_ARGS, MissingArguments, QCDefs, QCVm, ReturnValue, Type, Value, VectorField, VmOptions,
    load::{LoadFn, Progs, def_type, parse_arg_sizes},
    ops::Opcode,
    progs::{
        StringRef, StringTable, VmScalar, VmScalarType, VmType,
        functions::{ArgSize, FunctionBody, FunctionRegistry, Statement},
        globals::GlobalRegistry,
    },
    userdata::{AddrError, Context, EntityHandle, FnCall, Function, QCType},
};

use Opcode as O;

// Program builder =================================================================================

/// Locals are allocated from here, so that globals and locals can be added in any order.
const LOCALS_START: u16 = 1000;

/// A bare-bones `progs.dat` builder, where function bodies are written as raw statements.
struct Program {
    globals: Vec<GlobalDef>,
    values: Vec<[u8; 4]>,
    next_global: u16,
    next_local: u16,
    fields: Vec<FieldDef>,
    next_field: u16,
    strings: Vec<u8>,
    functions: Vec<LoadFn>,
    statements: Vec<Statement>,
    next_builtin: i32,
}

fn stmt(opcode: Opcode, arg1: u16, arg2: u16, arg3: u16) -> Statement {
    Statement {
        opcode,
        arg1: arg1 as i16,
        arg2: arg2 as i16,
        arg3: arg3 as i16,
    }
}

/// A jump statement, which takes a relative offset rather than an address.
fn jump(opcode: Opcode, cond: u16, offset: i16) -> Statement {
    match opcode {
        O::Goto => Statement {
            opcode,
            arg1: offset,
            arg2: 0,
            arg3: 0,
        },
        _ => Statement {
            opcode,
            arg1: cond as i16,
            arg2: offset,
            arg3: 0,
        },
    }
}

fn cstring(s: &str) -> Arc<CStr> {
    CString::new(s).unwrap().into()
}

impl Program {
    fn new() -> Self {
        Self {
            globals: vec![],
            // 0..28 are the return value and parameters.
            values: vec![[0; 4]; 28],
            next_global: 28,
            next_local: LOCALS_START,
            fields: vec![],
            next_field: 0,
            // String 0 is the null string.
            strings: vec![0],
            // Function 0 is the null function, which is just `DONE`.
            functions: vec![LoadFn {
                offset: 0,
                name: cstring(""),
                source: cstring(""),
                locals: 0..0,
                args: ArrayVec::new(),
            }],
            statements: vec![stmt(O::Done, 0, 0, 0)],
            next_builtin: -1,
        }
    }

    fn set_value(&mut self, addr: u16, value: [u8; 4]) {
        let addr = addr as usize;
        if self.values.len() <= addr {
            self.values.resize(addr + 1, [0; 4]);
        }
        self.values[addr] = value;
    }

    fn global(&mut self, name: &str, ty: impl Into<VmType>, values: &[[u8; 4]]) -> u16 {
        let ty = ty.into();
        assert_eq!(values.len(), ty.num_elements());
        let addr = self.next_global;
        assert!(
            addr + (values.len() as u16) < LOCALS_START,
            "Too many globals"
        );
        self.next_global += values.len() as u16;
        self.globals.push(GlobalDef {
            save: false,
            type_: ty,
            offset: addr,
            name: cstring(name),
        });
        for (i, value) in values.iter().enumerate() {
            self.set_value(addr + i as u16, *value);
        }
        addr
    }

    /// A global without a def, as optimizing compilers emit for immediates.
    fn untyped(&mut self, value: [u8; 4]) -> u16 {
        let addr = self.next_global;
        assert!(addr + 1 < LOCALS_START, "Too many globals");
        self.next_global += 1;
        self.set_value(addr, value);
        addr
    }

    fn float(&mut self, name: &str, f: f32) -> u16 {
        self.global(name, VmScalarType::Float, &[f.to_le_bytes()])
    }

    fn vector(&mut self, name: &str, v: [f32; 3]) -> u16 {
        self.global(name, VmType::Vector, &v.map(f32::to_le_bytes))
    }

    fn string_id(&mut self, s: &str) -> i32 {
        let id = self.strings.len() as i32;
        self.strings.extend_from_slice(s.as_bytes());
        self.strings.push(0);
        id
    }

    fn string(&mut self, name: &str, s: &str) -> u16 {
        let id = self.string_id(s);
        self.string_with_id(name, id)
    }

    fn string_with_id(&mut self, name: &str, id: i32) -> u16 {
        self.global(name, VmScalarType::String, &[id.to_le_bytes()])
    }

    fn entity(&mut self, name: &str) -> u16 {
        self.global(name, VmScalarType::Entity, &[[0; 4]])
    }

    /// Defines an entity field, and a global holding a reference to it (as `qcc` does).
    fn field(&mut self, name: &str, ty: impl Into<VmType>) -> u16 {
        let ty = ty.into();
        let offset = self.next_field;
        self.next_field += ty.num_elements() as u16;
        self.fields.push(FieldDef {
            type_: ty,
            offset,
            name: cstring(name),
        });
        // `qcc` also defines the components of vector fields.
        if ty == VmType::Vector {
            for (i, component) in ["x", "y", "z"].iter().enumerate() {
                self.fields.push(FieldDef {
                    type_: VmScalarType::Float.into(),
                    offset: offset + i as u16,
                    name: cstring(&format!("{name}_{component}")),
                });
            }
        }
        self.field_ref(name, offset)
    }

    fn field_ref(&mut self, name: &str, offset: u16) -> u16 {
        self.global(
            &format!("{name}$field"),
            VmScalarType::FieldRef,
            &[i32::from(offset).to_le_bytes()],
        )
    }

    /// A global holding a pointer to function `index`.
    fn function_ptr(&mut self, index: i32) -> u16 {
        self.global(
            &format!("fn{index}"),
            VmScalarType::Function,
            &[index.to_le_bytes()],
        )
    }

    fn builtin(&mut self, name: &str, args: &[ArgSize]) -> i32 {
        let index = self.functions.len() as i32;
        self.functions.push(LoadFn {
            offset: self.next_builtin,
            name: cstring(name),
            source: cstring("<builtin>"),
            locals: 0..0,
            args: args.iter().copied().collect(),
        });
        self.next_builtin -= 1;
        index
    }

    /// A builtin with an explicit builtin number (`#number` in QuakeC), which doesn't affect the
    /// numbering of later [`Self::builtin`]s.
    fn numbered_builtin(&mut self, name: &str, number: i32) -> i32 {
        let index = self.functions.len() as i32;
        self.functions.push(LoadFn {
            offset: -number,
            name: cstring(name),
            source: cstring("<builtin>"),
            locals: 0..0,
            args: ArrayVec::new(),
        });
        index
    }

    /// Adds a function with `locals` local scalars (including its arguments, which come first).
    /// `body` is passed the address of the first local.
    fn function(
        &mut self,
        name: &str,
        args: &[ArgSize],
        locals: u16,
        body: impl FnOnce(u16) -> Vec<Statement>,
    ) -> i32 {
        let index = self.functions.len() as i32;
        let local_start = self.next_local;
        self.next_local += locals;
        let offset = self.statements.len() as i32;
        self.statements.extend(body(local_start));
        self.functions.push(LoadFn {
            offset,
            name: cstring(name),
            source: cstring("<test>"),
            locals: local_start as usize..(local_start + locals) as usize,
            args: args.iter().copied().collect(),
        });
        index
    }

    fn build(self) -> QCVm {
        self.build_with(VmOptions::default())
    }

    fn build_with(mut self, options: VmOptions) -> QCVm {
        // Make sure that the locals (and the arguments of functions without locals, which come
        // after them) are inside the globals lump.
        let end = self.next_local + 3 * MAX_ARGS as u16;
        self.set_value(end, [0; 4]);

        let raw = self.values.iter().flatten().copied().collect::<Vec<u8>>();
        QCVm::from_progs(
            Progs {
                globals: GlobalRegistry::new(self.globals, &raw).unwrap(),
                entity_def: EntityTypeDef::new(self.fields),
                string_table: StringTable::new(self.strings).unwrap(),
                functions: FunctionRegistry::new(self.statements.into(), &self.functions).unwrap(),
            },
            options,
        )
    }
}

/// Reads a float global from the VM.
fn read_float(vm: &QCVm, addr: u16) -> f32 {
    match vm.global(addr).unwrap() {
        Value::Float(f) => f,
        Value::Void => 0.,
        other => panic!("Expected a float, found {other:?}"),
    }
}

fn read_vector(vm: &QCVm, addr: u16) -> [f32; 3] {
    [0, 1, 2].map(|i| read_float(vm, addr + i))
}

fn set_entity(vm: &QCVm, addr: u16, id: u64) {
    vm.progs
        .globals
        .set_value(addr.into(), VmScalar::Entity(NonZero::new(id)))
        .unwrap();
}

// Test host =======================================================================================

#[derive(Debug)]
struct TestError(String);

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TestError {}

/// The defs of [`TestContext`]. The CRC is Quake's, as [`real_progs_vm`] loads id1's `progs.dat`.
enum TestDefs {}

impl QCDefs for TestDefs {
    type GlobalAddr = EmptyAddress;
    type FieldAddr = TestFieldAddr;
    type BuiltinAddr = EmptyAddress;

    const CRC: i32 = 5927;
}

/// Fields known to the host at compile time. Other fields go through
/// [`EntityHandle::get_dynamic`].
#[derive(VariantArray, Debug, Copy, Clone, PartialEq, Eq)]
enum TestFieldAddr {
    Float,
    Vector,
    VectorX,
    VectorY,
    VectorZ,
}

impl Address for TestFieldAddr {
    fn name(&self) -> &'static str {
        match self {
            Self::Float => "float_field",
            Self::Vector => "vector_field",
            Self::VectorX => "vector_field_x",
            Self::VectorY => "vector_field_y",
            Self::VectorZ => "vector_field_z",
        }
    }

    fn vector_field_or_scalar(&self) -> (Self, VectorField) {
        match self {
            Self::VectorX => (Self::Vector, VectorField::XOrScalar),
            Self::VectorY => (Self::Vector, VectorField::Y),
            Self::VectorZ => (Self::Vector, VectorField::Z),
            other => (*other, VectorField::XOrScalar),
        }
    }

    fn type_(&self) -> Type {
        match self {
            Self::Vector => Type::Vector,
            _ => Type::Float,
        }
    }

    fn to_u16(&self) -> u16 {
        match self {
            Self::Float => 0,
            Self::Vector | Self::VectorX => 1,
            Self::VectorY => 2,
            Self::VectorZ => 3,
        }
    }
}

type FieldMap = hashbrown::HashMap<(u64, String), Value>;

#[derive(Default)]
struct TestContext {
    /// Host fields, by entity and name.
    fields: FieldMap,
    /// Fields defined by the progs, by entity and name.
    dynamic: FieldMap,
    /// Arguments to each `OP_STATE`.
    states: Vec<(f32, Value)>,
    /// Values passed to the `record` builtin.
    recorded: Vec<Value>,
}

#[derive(Debug, PartialEq)]
struct TestEnt {
    id: u64,
}

impl QCType for TestEnt {
    fn type_(&self) -> Type {
        Type::Entity
    }

    fn is_null(&self) -> bool {
        self.id == 0
    }
}

fn get_field(map: &FieldMap, id: u64, name: &str, ty: Type, field: Option<VectorField>) -> Value {
    let value = map
        .get(&(id, name.to_owned()))
        .cloned()
        .unwrap_or(Value::zero(ty));
    match field {
        Some(field) => value.field(field).unwrap(),
        None => value,
    }
}

fn set_field(
    map: &mut FieldMap,
    id: u64,
    name: &str,
    ty: Type,
    field: Option<VectorField>,
    value: Value,
) {
    let existing = map.entry((id, name.to_owned())).or_insert(Value::zero(ty));
    match field {
        Some(field) => existing.set(field, value).unwrap(),
        None => *existing = value,
    }
}

impl EntityHandle for TestEnt {
    type Context = TestContext;
    type Error = TestError;

    fn get(
        &self,
        context: &TestContext,
        field: TestFieldAddr,
    ) -> Result<Value, AddrError<TestError>> {
        let (root, component) = field.vector_field_or_scalar();
        let component = (root != field).then_some(component);
        Ok(get_field(
            &context.fields,
            self.id,
            root.name(),
            root.type_(),
            component,
        ))
    }

    fn set(
        &self,
        context: &mut TestContext,
        field: TestFieldAddr,
        value: Value,
    ) -> Result<(), AddrError<TestError>> {
        let (root, component) = field.vector_field_or_scalar();
        let component = (root != field).then_some(component);
        set_field(
            &mut context.fields,
            self.id,
            root.name(),
            root.type_(),
            component,
            value,
        );
        Ok(())
    }

    fn get_dynamic(
        &self,
        context: &TestContext,
        def: &FieldDef,
        component: Option<VectorField>,
    ) -> Result<Value, AddrError<TestError>> {
        let ty = Type::try_from(def.type_).unwrap();
        Ok(get_field(
            &context.dynamic,
            self.id,
            def.name.to_str().unwrap(),
            ty,
            component,
        ))
    }

    fn set_dynamic(
        &self,
        context: &mut TestContext,
        def: &FieldDef,
        component: Option<VectorField>,
        value: Value,
    ) -> Result<(), AddrError<TestError>> {
        let ty = Type::try_from(def.type_).unwrap();
        set_field(
            &mut context.dynamic,
            self.id,
            def.name.to_str().unwrap(),
            ty,
            component,
            value,
        );
        Ok(())
    }

    fn from_erased_mut<F, O>(erased: EntityRef, callback: F) -> Result<O, TestError>
    where
        F: FnOnce(&mut Self) -> O,
    {
        Ok(callback(&mut Self {
            id: erased.map(NonZero::get).unwrap_or_default(),
        }))
    }

    fn to_erased(&self) -> EntityRef {
        NonZero::new(self.id)
    }
}

type TestFnBody = dyn Fn(&mut FnCall<'_, TestContext>, Vec<Value>) -> Value + Send + Sync;

struct TestFn {
    signature: Vec<Type>,
    body: Box<TestFnBody>,
}

impl PartialEq for TestFn {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl fmt::Debug for TestFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestFn")
            .field("signature", &self.signature)
            .finish_non_exhaustive()
    }
}

impl QCType for TestFn {
    fn type_(&self) -> Type {
        Type::Function
    }

    fn is_null(&self) -> bool {
        false
    }
}

impl Function for TestFn {
    type Context = TestContext;
    type Error = TestError;

    fn signature(&self) -> Result<ArrayVec<Type, MAX_ARGS>, TestError> {
        Ok(self.signature.iter().copied().collect())
    }

    fn call(&self, mut context: FnCall<'_, TestContext>) -> Result<Value, TestError> {
        let args = context
            .arguments(&self.signature)
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| TestError(format!("{e:#}")))?;
        Ok((self.body)(&mut context, args))
    }
}

fn builtin(
    signature: &[Type],
    body: impl Fn(&mut FnCall<'_, TestContext>, Vec<Value>) -> Value + Send + Sync + 'static,
) -> Arc<TestFn> {
    Arc::new(TestFn {
        signature: signature.to_vec(),
        body: Box::new(body),
    })
}

impl Context for TestContext {
    type Entity = TestEnt;
    type Function = TestFn;
    type Error = TestError;
    type Defs = TestDefs;

    fn builtin(&self, def: &crate::BuiltinDef) -> Result<Arc<TestFn>, TestError> {
        Ok(match def.name.to_bytes() {
            b"add" => builtin(&[Type::Float, Type::Float], |_, args| {
                let [a, b]: [f32; 2] = args
                    .try_into()
                    .map(|args: [Value; 2]| args.map(|v| v.try_into().unwrap()))
                    .unwrap();
                (a + b).into()
            }),
            b"record" => builtin(&[Type::AnyScalar], |ctx, mut args| {
                ctx.recorded.push(args.remove(0));
                Value::Void
            }),
            // Records its argument, which must be a float.
            b"record_float" => builtin(&[Type::Float], |ctx, mut args| {
                ctx.recorded.push(args.remove(0));
                Value::Void
            }),
            // Records its argument, which must be a string.
            b"record_string" => builtin(&[Type::String], |ctx, mut args| {
                ctx.recorded.push(args.remove(0));
                Value::Void
            }),
            // Records its argument, and returns it.
            b"identity" => builtin(&[Type::AnyScalar], |ctx, args| {
                ctx.recorded.push(args[0].clone());
                args[0].clone()
            }),
            // Calls its argument from inside a builtin, as e.g. `walkmove` runs touch functions.
            b"callback" => builtin(&[Type::Function, Type::Float], |ctx, args| {
                let index = FunctionIndex::try_from(args[0].clone())
                    .expect("QuakeC functions should be passed as `Function::Index`");
                ctx.call(index, (args[1].clone(),))
                    .unwrap()
                    .scalar()
                    .unwrap()
            }),
            b"hello" => builtin(&[], |_, _| Value::try_from("hello").unwrap()),
            // Records the builtin number it was declared with.
            name if name.starts_with(b"which") => {
                let number = def.builtin_number().expect("builtins have numbers") as f32;
                builtin(&[], move |ctx, _| {
                    ctx.recorded.push(number.into());
                    Value::Void
                })
            }
            // Accepts anything, for running real `progs.dat`s.
            _ => builtin(&[], |_, _| Value::Void),
        })
    }

    fn global(&self, _def: EmptyAddress) -> Result<Value, AddrError<TestError>> {
        unreachable!()
    }

    fn set_global(
        &mut self,
        _def: EmptyAddress,
        _value: Value,
    ) -> Result<(), AddrError<TestError>> {
        unreachable!()
    }

    fn state(&mut self, frame: f32, think_fn: Value) -> anyhow::Result<()> {
        self.states.push((frame, think_fn));
        Ok(())
    }
}

fn run<'vm>(
    vm: &'vm QCVm,
    ctx: &mut TestContext,
    function: i32,
) -> anyhow::Result<ReturnValue<'vm>> {
    vm.run(ctx, function, ())
}

/// Builds a program with a single function `main` (no arguments or locals) and runs it.
fn run_main(
    program: Program,
    body: Vec<Statement>,
    ctx: &mut TestContext,
) -> (QCVm, anyhow::Result<()>) {
    let mut program = program;
    let main = program.function("main", &[], 0, |_| body);
    let vm = program.build();
    let result = run(&vm, ctx, main).map(|_| ());
    (vm, result)
}

fn function_index(value: &Value) -> i32 {
    FunctionIndex::try_from(value.clone()).unwrap().0
}

// Tests ===========================================================================================

#[test]
fn string_table_ids_into_middle_of_strings() {
    let table = StringTable::new(b"\0hello\0world\0").unwrap();

    assert!(crate::is_null_string(&table.get(0).unwrap()));
    assert_eq!(&*table.get(1).unwrap(), c"hello");
    // `qcc` can share the suffixes of strings.
    assert_eq!(&*table.get(3).unwrap(), c"llo");
    assert_eq!(&*table.get(9).unwrap(), c"rld");
    // The terminator of a string is an empty, but not null, string.
    let empty = table.get(6).unwrap();
    assert!(empty.is_empty());
    assert!(!crate::is_null_string(&empty));
    assert!(table.get(100).is_err());
    assert!(table.get(-1).is_err());

    // Unterminated lumps are terminated.
    let table = StringTable::new(b"\0abc").unwrap();
    assert_eq!(&*table.get(1).unwrap(), c"abc");
    assert_eq!(&*table.get(StringRef::Id(2)).unwrap(), c"bc");
}

#[test]
fn arithmetic() {
    let mut p = Program::new();
    let two = p.float("two", 2.);
    let three = p.float("three", 3.);
    let v1 = p.vector("v1", [1., 2., 3.]);
    let v2 = p.vector("v2", [4., 5., 6.]);
    let mul_f = p.float("mul_f", 0.);
    let div = p.float("div", 0.);
    let add_f = p.float("add_f", 0.);
    let sub_f = p.float("sub_f", 0.);
    let dot = p.float("dot", 0.);
    let mul_fv = p.vector("mul_fv", [0.; 3]);
    let mul_vf = p.vector("mul_vf", [0.; 3]);
    let add_v = p.vector("add_v", [0.; 3]);
    let sub_v = p.vector("sub_v", [0.; 3]);
    let copy_v = p.vector("copy_v", [0.; 3]);

    let (vm, result) = run_main(
        p,
        vec![
            stmt(O::MulF, two, three, mul_f),
            stmt(O::Div, three, two, div),
            stmt(O::AddF, two, three, add_f),
            stmt(O::SubF, two, three, sub_f),
            stmt(O::MulV, v1, v2, dot),
            stmt(O::MulFV, two, v1, mul_fv),
            stmt(O::MulVF, v1, three, mul_vf),
            stmt(O::AddV, v1, v2, add_v),
            stmt(O::SubV, v1, v2, sub_v),
            stmt(O::StoreV, v2, copy_v, 0),
            stmt(O::Done, 0, 0, 0),
        ],
        &mut TestContext::default(),
    );
    result.unwrap();

    assert_eq!(read_float(&vm, mul_f), 6.);
    assert_eq!(read_float(&vm, div), 1.5);
    assert_eq!(read_float(&vm, add_f), 5.);
    assert_eq!(read_float(&vm, sub_f), -1.);
    assert_eq!(read_float(&vm, dot), 32.);
    assert_eq!(read_vector(&vm, mul_fv), [2., 4., 6.]);
    assert_eq!(read_vector(&vm, mul_vf), [3., 6., 9.]);
    assert_eq!(read_vector(&vm, add_v), [5., 7., 9.]);
    assert_eq!(read_vector(&vm, sub_v), [-3., -3., -3.]);
    assert_eq!(read_vector(&vm, copy_v), [4., 5., 6.]);
}

#[test]
fn comparisons() {
    let mut p = Program::new();
    let one = p.float("one", 1.);
    let two = p.float("two", 2.);
    let v1 = p.vector("v1", [1., 2., 3.]);
    let v2 = p.vector("v2", [1., 2., 4.]);
    // Two different string IDs with the same contents.
    let s1 = p.string("s1", "foo");
    let s2 = p.string("s2", "foo");
    let s3 = p.string("s3", "bar");
    let e1 = p.entity("e1");
    let e2 = p.entity("e2");
    let outs = (0..20)
        .map(|i| p.float(&format!("out{i}"), -1.))
        .collect::<Vec<_>>();

    let (vm, result) = run_main(
        p,
        vec![
            stmt(O::EqF, one, one, outs[0]),
            stmt(O::NeF, one, two, outs[1]),
            stmt(O::Le, one, one, outs[2]),
            stmt(O::Ge, one, two, outs[3]),
            stmt(O::Lt, one, two, outs[4]),
            stmt(O::Gt, one, two, outs[5]),
            stmt(O::EqV, v1, v1, outs[6]),
            stmt(O::NeV, v1, v2, outs[7]),
            stmt(O::EqS, s1, s2, outs[8]),
            stmt(O::NeS, s1, s2, outs[9]),
            stmt(O::EqS, s1, s3, outs[10]),
            stmt(O::NeS, s1, s3, outs[11]),
            stmt(O::EqEnt, e1, e2, outs[12]),
            stmt(O::NeEnt, e1, e2, outs[13]),
            stmt(O::Done, 0, 0, 0),
        ],
        &mut TestContext::default(),
    );

    let _ = (e1, e2);
    result.unwrap();

    let expected = [1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0.];
    for (i, expected) in expected.into_iter().enumerate() {
        assert_eq!(read_float(&vm, outs[i]), expected, "out{i}");
    }
}

#[test]
fn entity_comparisons() {
    let mut p = Program::new();
    let e1 = p.entity("e1");
    let e2 = p.entity("e2");
    let world = p.entity("world");
    let eq = p.float("eq", -1.);
    let ne = p.float("ne", -1.);
    let not_world = p.float("not_world", -1.);
    let not_e1 = p.float("not_e1", -1.);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::EqEnt, e1, e2, eq),
            stmt(O::NeEnt, e1, world, ne),
            stmt(O::NotEnt, world, 0, not_world),
            stmt(O::NotEnt, e1, 0, not_e1),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, e1, 5);
    set_entity(&vm, e2, 5);
    run(&vm, &mut TestContext::default(), main).unwrap();

    assert_eq!(read_float(&vm, eq), 1.);
    assert_eq!(read_float(&vm, ne), 1.);
    assert_eq!(read_float(&vm, not_world), 1.);
    assert_eq!(read_float(&vm, not_e1), 0.);
}

/// `IF`/`IFNOT` test the raw bits of their argument.
#[test]
fn truthiness() {
    let mut p = Program::new();
    let zero = p.float("zero", 0.);
    let neg_zero = p.float("neg_zero", -0.);
    let one = p.float("one", 1.);
    let null_string = p.string_with_id("null_string", 0);
    let empty_string = p.string("empty_string", "");
    let string = p.string("string", "foo");
    let null_fn = p.function_ptr(0);
    let some_fn = p.function_ptr(1);

    let conditions = [
        (zero, false),
        (neg_zero, true),
        (one, true),
        (null_string, false),
        (empty_string, true),
        (string, true),
        (null_fn, false),
        (some_fn, true),
    ];
    let if_outs = conditions
        .iter()
        .enumerate()
        .map(|(i, _)| p.float(&format!("if{i}"), -1.))
        .collect::<Vec<_>>();
    let ifnot_outs = conditions
        .iter()
        .enumerate()
        .map(|(i, _)| p.float(&format!("ifnot{i}"), -1.))
        .collect::<Vec<_>>();

    let mut body = vec![];
    for (i, (cond, _)) in conditions.iter().enumerate() {
        // out = 1; if (cond) skip; out = 0;
        body.extend([
            stmt(O::StoreF, one, if_outs[i], 0),
            jump(O::If, *cond, 2),
            stmt(O::StoreF, zero, if_outs[i], 0),
        ]);
        // out = 0; if (!cond) skip; out = 1;
        body.extend([
            stmt(O::StoreF, zero, ifnot_outs[i], 0),
            jump(O::IfNot, *cond, 2),
            stmt(O::StoreF, one, ifnot_outs[i], 0),
        ]);
    }
    body.push(stmt(O::Done, 0, 0, 0));

    let (vm, result) = run_main(p, body, &mut TestContext::default());
    result.unwrap();

    for (i, (_, expected)) in conditions.iter().enumerate() {
        let expected = if *expected { 1. } else { 0. };
        assert_eq!(read_float(&vm, if_outs[i]), expected, "IF condition {i}");
        assert_eq!(
            read_float(&vm, ifnot_outs[i]),
            expected,
            "IFNOT condition {i}"
        );
    }
}

/// `NOT_*`, `AND` and `OR` compare by value, so `-0.0` is false.
#[test]
fn logic() {
    let mut p = Program::new();
    let zero = p.float("zero", 0.);
    let neg_zero = p.float("neg_zero", -0.);
    let one = p.float("one", 1.);
    let null_string = p.string_with_id("null_string", 0);
    let empty_string = p.string("empty_string", "");
    let string = p.string("string", "foo");
    let zero_v = p.vector("zero_v", [0.; 3]);
    let some_v = p.vector("some_v", [0., 0., 1.]);
    let null_fn = p.function_ptr(0);
    let some_fn = p.function_ptr(1);
    let outs = (0..14)
        .map(|i| p.float(&format!("out{i}"), -1.))
        .collect::<Vec<_>>();

    let (vm, result) = run_main(
        p,
        vec![
            stmt(O::NotF, zero, 0, outs[0]),
            stmt(O::NotF, neg_zero, 0, outs[1]),
            stmt(O::NotF, one, 0, outs[2]),
            stmt(O::NotS, null_string, 0, outs[3]),
            // Quake treats empty strings as false here.
            stmt(O::NotS, empty_string, 0, outs[4]),
            stmt(O::NotS, string, 0, outs[5]),
            stmt(O::NotV, zero_v, 0, outs[6]),
            stmt(O::NotV, some_v, 0, outs[7]),
            stmt(O::NotFnc, null_fn, 0, outs[8]),
            stmt(O::NotFnc, some_fn, 0, outs[9]),
            stmt(O::And, one, neg_zero, outs[10]),
            stmt(O::And, one, one, outs[11]),
            stmt(O::Or, zero, neg_zero, outs[12]),
            stmt(O::Or, zero, one, outs[13]),
            stmt(O::Done, 0, 0, 0),
        ],
        &mut TestContext::default(),
    );
    result.unwrap();

    let expected = [1., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1.];
    for (i, expected) in expected.into_iter().enumerate() {
        assert_eq!(read_float(&vm, outs[i]), expected, "out{i}");
    }
}

/// Bitwise operations go through `i32`, so negative numbers work.
#[test]
fn bitwise() {
    let mut p = Program::new();
    let neg_one = p.float("neg_one", -1.);
    let five = p.float("five", 5.);
    let four = p.float("four", 4.);
    let three = p.float("three", 3.);
    let and = p.float("and", 0.);
    let or = p.float("or", 0.);
    let and2 = p.float("and2", 0.);
    let or2 = p.float("or2", 0.);

    let (vm, result) = run_main(
        p,
        vec![
            stmt(O::BitAnd, neg_one, five, and),
            stmt(O::BitOr, neg_one, five, or),
            stmt(O::BitAnd, four, three, and2),
            stmt(O::BitOr, four, three, or2),
            stmt(O::Done, 0, 0, 0),
        ],
        &mut TestContext::default(),
    );
    result.unwrap();

    assert_eq!(read_float(&vm, and), 5.);
    assert_eq!(read_float(&vm, or), -1.);
    assert_eq!(read_float(&vm, and2), 0.);
    assert_eq!(read_float(&vm, or2), 7.);
}

#[test]
fn state() {
    let mut p = Program::new();
    let frame = p.float("frame", 7.);
    let think = p.function("think", &[], 0, |_| vec![stmt(O::Done, 0, 0, 0)]);
    let think_ptr = p.function_ptr(think);

    let mut ctx = TestContext::default();
    let (_, result) = run_main(
        p,
        vec![stmt(O::State, frame, think_ptr, 0), stmt(O::Done, 0, 0, 0)],
        &mut ctx,
    );
    result.unwrap();

    assert_eq!(ctx.states.len(), 1);
    assert_eq!(ctx.states[0].0, 7.);
    assert_eq!(function_index(&ctx.states[0].1), think);
}

#[test]
fn function_pointers_are_indices() {
    let mut p = Program::new();
    // Builtins and padding make indices and statement offsets differ.
    let add = p.builtin("add", &[ArgSize::Scalar, ArgSize::Scalar]);
    let _padding = p.function("padding", &[], 0, |_| {
        vec![
            stmt(O::Done, 0, 0, 0),
            stmt(O::Done, 0, 0, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let two = p.float("two", 2.);
    let ten = p.float("ten", 10.);
    let add_ptr = p.function_ptr(add);
    let double = p.function("double", &[ArgSize::Scalar], 2, |locals| {
        let (arg, tmp) = (locals, locals + 1);
        vec![stmt(O::MulF, arg, two, tmp), stmt(O::Return, tmp, 0, 0)]
    });
    let double_ptr = p.function_ptr(double);
    let out = p.float("out", 0.);
    let fn_local = p.global("fn_local", VmScalarType::Function, &[[0; 4]]);
    let eq = p.float("eq", -1.);
    let ne = p.float("ne", -1.);

    let main = p.function("main", &[], 0, |_| {
        vec![
            // out = double(add(2, 10))
            stmt(O::StoreF, two, 4, 0),
            stmt(O::StoreF, ten, 7, 0),
            stmt(O::Call2, add_ptr, 0, 0),
            stmt(O::StoreF, 1, 4, 0),
            stmt(O::Call1, double_ptr, 0, 0),
            stmt(O::StoreF, 1, out, 0),
            // Function pointers copied around still compare equal.
            stmt(O::StoreFnc, double_ptr, fn_local, 0),
            stmt(O::EqFnc, fn_local, double_ptr, eq),
            stmt(O::NeFnc, fn_local, add_ptr, ne),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    assert_ne!(
        vm.functions().get_by_index(double).unwrap().offset,
        double,
        "The test should use a function whose index isn't its offset"
    );

    run(&vm, &mut TestContext::default(), main).unwrap();
    assert_eq!(read_float(&vm, out), 24.);
    assert_eq!(read_float(&vm, eq), 1.);
    assert_eq!(read_float(&vm, ne), 1.);

    // Functions can also be run by index or name from the host.
    let value = vm
        .run(&mut TestContext::default(), double, (4f32,))
        .unwrap();
    assert_eq!(value.float().unwrap(), 8.);
    let value = vm
        .run(&mut TestContext::default(), c"double", (5f32,))
        .unwrap();
    assert_eq!(value.get::<f32>().unwrap(), 10.);

    // The initial values of function globals are exposed as indices.
    assert_eq!(
        function_index(&vm.initial_global(double_ptr).unwrap()),
        double
    );
}

/// Functions passed through the host still compare equal to the same function in QuakeC (e.g.
/// `self.think == SUB_Null` after the host has stored `think`).
#[test]
fn function_identity_survives_the_host() {
    let mut p = Program::new();
    let identity = p.builtin("identity", &[ArgSize::Scalar]);
    let identity_ptr = p.function_ptr(identity);
    let target = p.function("target", &[], 0, |_| vec![stmt(O::Done, 0, 0, 0)]);
    let target_ptr = p.function_ptr(target);
    let stored = p.global("stored", VmScalarType::Function, &[[0; 4]]);
    let eq = p.float("eq", -1.);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreFnc, target_ptr, 4, 0),
            stmt(O::Call1, identity_ptr, 0, 0),
            stmt(O::StoreFnc, 1, stored, 0),
            stmt(O::EqFnc, stored, target_ptr, eq),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();
    assert_eq!(ctx.recorded, [FunctionIndex(target).to_value()]);
    assert_eq!(read_float(&vm, eq), 1.);
}

#[test]
fn null_function() {
    let mut p = Program::new();
    let null_ptr = p.function_ptr(0);
    let (_, result) = run_main(
        p,
        vec![stmt(O::Call0, null_ptr, 0, 0), stmt(O::Done, 0, 0, 0)],
        &mut TestContext::default(),
    );

    let error = format!("{:#}", result.unwrap_err());
    assert!(error.contains("NULL function"), "{error}");
}

/// Globals defined by the progs (rather than the host) are stored in the VM, persist between calls
/// and can be reset.
#[test]
fn mutable_mod_globals() {
    let mut p = Program::new();
    let counter = p.float("counter", 10.);
    let one = p.float("one", 1.);
    // A temporary, which has no definition.
    let temp = 900;
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::AddF, counter, one, temp),
            stmt(O::StoreF, temp, counter, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    run(&vm, &mut TestContext::default(), main).unwrap();
    run(&vm, &mut TestContext::default(), main).unwrap();
    assert_eq!(read_float(&vm, counter), 12.);
    assert_eq!(read_float(&vm, temp), 12.);
    assert_eq!(vm.initial_global(counter).unwrap(), Value::Float(10.));

    vm.reset_globals();
    assert_eq!(read_float(&vm, counter), 10.);
}

/// Storing `void` (e.g. an uninitialized temporary) is a store of zero.
#[test]
fn void_stores() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    let float_field = p.field("float_field", VmScalarType::Float);
    let _vector_field = p.field("vector_field", VmType::Vector);
    let speed = p.field("speed", VmScalarType::Float);
    let global = p.float("global", 5.);
    let void_temp = 901;
    let ptr = 902;
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreF, void_temp, global, 0),
            stmt(O::Address, ent, float_field, ptr),
            stmt(O::StorePF, void_temp, ptr, 0),
            stmt(O::Address, ent, speed, ptr),
            stmt(O::StorePF, void_temp, ptr, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 1);

    let mut ctx = TestContext::default();
    ctx.fields
        .insert((1, "float_field".into()), Value::Float(100.));
    ctx.dynamic.insert((1, "speed".into()), Value::Float(100.));
    run(&vm, &mut ctx, main).unwrap();

    assert_eq!(read_float(&vm, global), 0.);
    assert_eq!(ctx.fields[&(1, "float_field".into())], Value::Float(0.));
    assert_eq!(ctx.dynamic[&(1, "speed".into())], Value::Float(0.));
}

/// fteqcc's `__NULL__` is a single nameless float immediate `0`, which it uses as the null of every
/// type. In Quake that's just the zero word, so it must act as a null function/string/entity.
#[test]
fn fteqcc_null_is_null_of_every_type() {
    let mut p = Program::new();
    let null = p.float("", 0.);
    let ent = p.entity("ent");
    let world = p.entity("world");
    let touch = p.field("touch", VmScalarType::Function);
    let null_fn = p.function_ptr(0);
    let some_fn = p.function_ptr(1);
    let some_s = p.string("some_s", "hello");
    let null_s = p.string_with_id("null_s", 0);
    let fn_ne = p.float("fn_ne", -1.);
    let null_fn_ne = p.float("null_fn_ne", -1.);
    let null_fn_eq = p.float("null_fn_eq", -1.);
    let s_ne = p.float("s_ne", -1.);
    let s_eq = p.float("s_eq", -1.);
    let null_s_eq = p.float("null_s_eq", -1.);
    let world_eq = p.float("world_eq", -1.);
    let ent_eq = p.float("ent_eq", -1.);
    let ptr = 900;
    let main = p.function("main", &[], 0, |_| {
        vec![
            // self.touch = __NULL__
            stmt(O::Address, ent, touch, ptr),
            stmt(O::StorePFnc, null, ptr, 0),
            stmt(O::NeFnc, some_fn, null, fn_ne),
            stmt(O::NeFnc, null_fn, null, null_fn_ne),
            stmt(O::EqFnc, null, null_fn, null_fn_eq),
            stmt(O::NeS, some_s, null, s_ne),
            stmt(O::EqS, some_s, null, s_eq),
            stmt(O::EqS, null_s, null, null_s_eq),
            stmt(O::EqEnt, world, null, world_eq),
            stmt(O::EqEnt, ent, null, ent_eq),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 1);

    let mut ctx = TestContext::default();
    ctx.dynamic
        .insert((1, "touch".into()), FunctionIndex(1).to_value());
    run(&vm, &mut ctx, main).unwrap();

    assert_eq!(
        ctx.dynamic[&(1, "touch".into())],
        Value::Function(crate::Function::NULL)
    );
    assert_eq!(read_float(&vm, fn_ne), 1.);
    assert_eq!(read_float(&vm, null_fn_ne), 0.);
    assert_eq!(read_float(&vm, null_fn_eq), 1.);
    assert_eq!(read_float(&vm, s_ne), 1.);
    assert_eq!(read_float(&vm, s_eq), 0.);
    assert_eq!(read_float(&vm, null_s_eq), 1.);
    assert_eq!(read_float(&vm, world_eq), 1.);
    assert_eq!(read_float(&vm, ent_eq), 0.);
}

/// Only the zero word is the null of every type: other mismatched values are still errors.
#[test]
fn non_null_type_mismatches_still_error() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    let touch = p.field("touch", VmScalarType::Function);
    let one = p.float("one", 1.);
    let ptr = 900;
    let body = vec![
        stmt(O::Address, ent, touch, ptr),
        stmt(O::StorePFnc, one, ptr, 0),
        stmt(O::Done, 0, 0, 0),
    ];
    let mut ctx = TestContext::default();
    let main = p.function("main", &[], 0, |_| body);
    let vm = p.build();
    set_entity(&vm, ent, 1);
    let error = format!("{:#}", run(&vm, &mut ctx, main).unwrap_err());
    assert!(error.contains("ScalarCastError"), "{error}");

    // `-0.0` isn't the zero word.
    let mut p = Program::new();
    let neg_zero = p.float("neg_zero", -0.);
    let some_fn = p.function_ptr(1);
    let out = p.float("out", -1.);
    let (_, result) = run_main(
        p,
        vec![
            stmt(O::NeFnc, some_fn, neg_zero, out),
            stmt(O::Done, 0, 0, 0),
        ],
        &mut TestContext::default(),
    );
    let error = format!("{:#}", result.unwrap_err());
    assert!(error.contains("ScalarCastError"), "{error}");
}

#[test]
fn null_scalar_conversions() {
    use crate::progs::{EntityField, Ptr, VmFunctionRef};

    assert!(
        StringRef::try_from(VmScalar::Entity(None))
            .unwrap()
            .is_null()
    );
    assert!(
        VmFunctionRef::try_from(VmScalar::Float(0.))
            .unwrap()
            .is_null()
    );
    assert_eq!(
        EntityRef::try_from(VmScalar::String(StringRef::default())).unwrap(),
        None
    );
    assert!(f32::try_from(VmScalar::EntityField(None, Ptr(0))).is_err());
    assert!(VmFunctionRef::try_from(VmScalar::Float(-0.)).is_err());
    assert!(StringRef::try_from(VmScalar::Float(1.)).is_err());
    // `EntityField` itself is unchanged: only a real field pointer converts.
    assert!(EntityField::try_from(VmScalar::Float(0.)).is_err());
}

#[test]
fn host_fields() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    let float_field = p.field("float_field", VmScalarType::Float);
    let vector_field = p.field("vector_field", VmType::Vector);
    let vector_field_y = p.field_ref("vector_field_y", 2);
    let value = p.float("value", 3.);
    let vector = p.vector("vector", [1., 2., 3.]);
    let loaded = p.float("loaded", 0.);
    let loaded_v = p.vector("loaded_v", [0.; 3]);
    let loaded_y = p.float("loaded_y", 0.);
    let ptr = 900;

    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::Address, ent, float_field, ptr),
            stmt(O::StorePF, value, ptr, 0),
            stmt(O::LoadF, ent, float_field, loaded),
            stmt(O::Address, ent, vector_field, ptr),
            stmt(O::StorePV, vector, ptr, 0),
            stmt(O::LoadV, ent, vector_field, loaded_v),
            stmt(O::LoadF, ent, vector_field_y, loaded_y),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 3);

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();

    assert_eq!(ctx.fields[&(3, "float_field".into())], Value::Float(3.));
    assert_eq!(read_float(&vm, loaded), 3.);
    assert_eq!(read_vector(&vm, loaded_v), [1., 2., 3.]);
    assert_eq!(read_float(&vm, loaded_y), 2.);
}

/// Fields defined by the progs (e.g. `pos1` in Quake) go through `get_dynamic`/`set_dynamic`.
#[test]
fn dynamic_fields() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    // Fill in the host's fields first, so that the progs fields come after them.
    let _float_field = p.field("float_field", VmScalarType::Float);
    let _vector_field = p.field("vector_field", VmType::Vector);
    let pos1 = p.field("pos1", VmType::Vector);
    let pos1_z = p.field_ref("pos1_z", 6);
    let speed = p.field("speed", VmScalarType::Float);
    let target = p.field("target", VmScalarType::String);
    let vector = p.vector("vector", [1., 2., 3.]);
    let value = p.float("value", 9.);
    let text = p.string("text", "door");
    let ptr = 900;
    let loaded_v = p.vector("loaded_v", [0.; 3]);
    let loaded_z = p.float("loaded_z", 0.);
    let loaded_speed = p.float("loaded_speed", 0.);
    let eq = p.float("eq", 0.);

    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::Address, ent, pos1, ptr),
            stmt(O::StorePV, vector, ptr, 0),
            stmt(O::LoadV, ent, pos1, loaded_v),
            stmt(O::LoadF, ent, pos1_z, loaded_z),
            stmt(O::Address, ent, speed, ptr),
            stmt(O::StorePF, value, ptr, 0),
            stmt(O::LoadF, ent, speed, loaded_speed),
            stmt(O::Address, ent, target, ptr),
            stmt(O::StorePS, text, ptr, 0),
            stmt(O::LoadS, ent, target, 903),
            stmt(O::EqS, 903, text, eq),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 2);

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();

    assert_eq!(
        ctx.dynamic[&(2, "pos1".into())],
        Value::Vector([1., 2., 3.].into())
    );
    assert_eq!(read_vector(&vm, loaded_v), [1., 2., 3.]);
    assert_eq!(read_float(&vm, loaded_z), 3.);
    assert_eq!(read_float(&vm, loaded_speed), 9.);
    assert_eq!(read_float(&vm, eq), 1.);
}

/// QuakeC can store field references and pointers into temporaries and parameters, and pass
/// fields to builtins (e.g. `find`).
#[test]
fn store_field_refs_and_pointers() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    let float_field = p.field("float_field", VmScalarType::Float);
    let record = p.builtin("record", &[ArgSize::Scalar]);
    let record_ptr = p.function_ptr(record);
    let value = p.float("value", 4.);
    let main = p.function("main", &[], 3, |locals| {
        let (field_copy, ptr) = (locals, locals + 1);
        vec![
            stmt(O::StoreFld, float_field, field_copy, 0),
            stmt(O::Address, ent, field_copy, ptr),
            stmt(O::StorePF, value, ptr, 0),
            stmt(O::StoreFld, float_field, 4, 0),
            stmt(O::Call1, record_ptr, 0, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 1);

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();
    assert_eq!(ctx.fields[&(1, "float_field".into())], Value::Float(4.));
    assert_eq!(ctx.recorded, [Value::Field(FieldRef { offset: 0 })]);
}

/// Field references can go to the host and back without losing their type.
#[test]
fn field_refs_round_trip_through_the_host() {
    let mut p = Program::new();
    let ent = p.entity("ent");
    // Fill in the host's fields first, so that `speed` comes after them.
    let _float_field = p.field("float_field", VmScalarType::Float);
    let _vector_field = p.field("vector_field", VmType::Vector);
    let speed = p.field("speed", VmScalarType::Float);
    let identity = p.builtin("identity", &[ArgSize::Scalar]);
    let identity_ptr = p.function_ptr(identity);
    let field_global = p.global("field_global", VmScalarType::FieldRef, &[[0; 4]]);
    let loaded = p.float("loaded", 0.);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreFld, speed, 4, 0),
            stmt(O::Call1, identity_ptr, 0, 0),
            stmt(O::StoreFld, 1, field_global, 0),
            stmt(O::LoadF, ent, field_global, loaded),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();
    set_entity(&vm, ent, 1);

    let mut ctx = TestContext::default();
    ctx.dynamic.insert((1, "speed".into()), Value::Float(320.));
    run(&vm, &mut ctx, main).unwrap();

    let speed_ref = Value::Field(FieldRef { offset: 4 });
    assert_eq!(ctx.recorded, std::slice::from_ref(&speed_ref));
    assert_eq!(vm.global(field_global).unwrap(), speed_ref);
    assert_eq!(read_float(&vm, loaded), 320.);
}

/// QuakeC can be re-entered from inside a builtin, and the caller carries on afterwards.
#[test]
fn nested_calls() {
    let mut p = Program::new();
    let callback = p.builtin("callback", &[ArgSize::Scalar, ArgSize::Scalar]);
    let callback_ptr = p.function_ptr(callback);
    let three = p.float("three", 3.);
    let one = p.float("one", 1.);
    let inner_result = p.float("inner_result", 0.);
    let after = p.float("after", 0.);
    let inner = p.function("inner", &[ArgSize::Scalar], 2, |locals| {
        vec![
            stmt(O::AddF, locals, one, locals + 1),
            stmt(O::Return, locals + 1, 0, 0),
        ]
    });
    let inner_ptr = p.function_ptr(inner);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreFnc, inner_ptr, 4, 0),
            stmt(O::StoreF, three, 7, 0),
            stmt(O::Call2, callback_ptr, 0, 0),
            stmt(O::StoreF, 1, inner_result, 0),
            // This must run after the nested call returns.
            stmt(O::StoreF, one, after, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    run(&vm, &mut TestContext::default(), main).unwrap();
    assert_eq!(read_float(&vm, inner_result), 4.);
    assert_eq!(read_float(&vm, after), 1.);
}

#[test]
fn runaway_loop() {
    let p = Program::new();
    let (_, result) = run_main(
        p,
        vec![jump(O::Goto, 0, 1), jump(O::Goto, 0, -1)],
        &mut TestContext::default(),
    );

    let error = format!("{:#}", result.unwrap_err());
    assert!(error.contains("runaway loop"), "{error}");
}

#[test]
fn call_depth_limit() {
    let mut p = Program::new();
    // Function pointers are indices, so we know what this one will be.
    let recurse_ptr = p.function_ptr(1);
    let recurse = p.function("recurse", &[], 0, |_| {
        vec![stmt(O::Call0, recurse_ptr, 0, 0), stmt(O::Done, 0, 0, 0)]
    });
    assert_eq!(recurse, 1);
    let vm = p.build();

    let error = format!(
        "{:#}",
        run(&vm, &mut TestContext::default(), recurse).unwrap_err()
    );
    assert!(error.contains("Call stack overflow"), "{error}");
    // Errors include the QuakeC call stack.
    assert!(error.contains("in recurse"), "{error}");
}

/// Locals start with their values from the `progs.dat`, and arguments are copied into them.
#[test]
fn locals_and_arguments() {
    let mut p = Program::new();
    let vector = p.vector("vector", [1., 2., 3.]);
    let five = p.float("five", 5.);
    let scale = p.function("scale", &[ArgSize::Vector, ArgSize::Scalar], 8, |locals| {
        let (v, f, out) = (locals, locals + 3, locals + 4);
        vec![stmt(O::MulVF, v, f, out), stmt(O::Return, out, 0, 0)]
    });
    let scale_ptr = p.function_ptr(scale);
    let out = p.vector("out", [0.; 3]);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreV, vector, 4, 0),
            stmt(O::StoreF, five, 7, 0),
            stmt(O::Call2, scale_ptr, 0, 0),
            stmt(O::StoreV, 1, out, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    run(&vm, &mut TestContext::default(), main).unwrap();
    assert_eq!(read_vector(&vm, out), [5., 10., 15.]);
}

/// Returning a scalar copies three values, the other two of which can be anything.
#[test]
fn scalar_return_values() {
    let mut p = Program::new();
    let text = p.string("text", "foo");
    let _ent = p.entity("ent");
    let get_text = p.function("get_text", &[], 0, |_| vec![stmt(O::Return, text, 0, 0)]);
    let hello = p.builtin("hello", &[]);
    let vm = p.build();

    let value = run(&vm, &mut TestContext::default(), get_text).unwrap();
    assert_eq!(&*value.string().unwrap(), c"foo");
    assert_eq!(value.scalar().unwrap(), Value::try_from("foo").unwrap());
    // Builtins can be run directly too.
    let value = run(&vm, &mut TestContext::default(), hello).unwrap();
    assert_eq!(&*value.string().unwrap(), c"hello");
    // Reading the return value as the wrong type is an error, but ignoring it never is.
    assert!(value.float().is_err());
    value.get::<()>().unwrap();
}

/// The caller says how to read a return value, so a float returned from a temporary is read as a
/// float even when the temporaries after it also hold floats (so all three return scalars do).
#[test]
fn float_return_values_next_to_floats() {
    let mut p = Program::new();
    let one = p.float("one", 1.);
    let two = p.float("two", 2.);
    let (tmp, next, next2) = (900, 901, 902);
    let get_float = p.function("get_float", &[], 0, |_| {
        vec![
            stmt(O::AddF, one, two, tmp),
            stmt(O::AddF, one, one, next),
            stmt(O::AddF, two, two, next2),
            stmt(O::Return, tmp, 0, 0),
        ]
    });
    let vm = p.build();

    let value = run(&vm, &mut TestContext::default(), get_float).unwrap();
    assert_eq!(value.float().unwrap(), 3.);
    assert_eq!(value.scalar().unwrap(), Value::Float(3.));
    // Read as a vector, it's the return slots as Quake would see them.
    assert_eq!(value.vector().unwrap(), glam::Vec3::new(3., 2., 4.));
}

/// Loads the real `progs.dat` from the path in `QCVM_PROGS`. If it isn't set, the test is skipped.
fn real_progs_vm() -> Option<QCVm> {
    let Some(path) = std::env::var_os("QCVM_PROGS") else {
        eprintln!("Skipping test: `QCVM_PROGS` isn't set");
        return None;
    };
    let bytes = std::fs::read(path).unwrap();
    Some(QCVm::load::<TestDefs, _>(std::io::Cursor::new(bytes)).unwrap())
}

/// Loads a real `progs.dat` (see [`real_progs_vm`]) and runs `main`.
#[test]
fn real_progs() {
    let Some(vm) = real_progs_vm() else {
        return;
    };

    let main = vm.functions().get_by_name(c"main").unwrap().index;
    run(&vm, &mut TestContext::default(), main).unwrap();

    // `StartFrame` is a function global defined by `progdefs.h`.
    let start_frame = vm
        .global_defs()
        .iter()
        .find(|def| &*def.name == c"StartFrame")
        .unwrap()
        .offset;
    let start_frame = function_index(&vm.initial_global(start_frame).unwrap());
    assert_eq!(
        &*vm.functions().get_by_index(start_frame).unwrap().name,
        c"StartFrame"
    );
}

/// `qcc` copies arguments of any type into the parameter slots with `STORE_V`.
#[test]
fn store_v_into_parameters_is_untyped() {
    let mut p = Program::new();
    let record = p.builtin("record", &[ArgSize::Scalar]);
    let record_ptr = p.function_ptr(record);
    let text = p.string("text", "hello");
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreV, text, 4, 0),
            stmt(O::Call1, record_ptr, 0, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();
    assert_eq!(ctx.recorded, [Value::try_from("hello").unwrap()]);
}

/// Outside of the parameter slots, `STORE_V` copies a vector, so it can't copy other types.
#[test]
fn store_v_elsewhere_is_typed() {
    for dest in [1, 900] {
        let mut p = Program::new();
        let text = p.string("text", "hello");
        let (_, result) = run_main(
            p,
            vec![stmt(O::StoreV, text, dest, 0), stmt(O::Done, 0, 0, 0)],
            &mut TestContext::default(),
        );
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("ScalarCastError"), "{error}");
    }
}

/// `qcc` sometimes says that functions with arguments have no locals, in which case the arguments
/// are in global memory.
#[test]
fn arguments_without_locals() {
    let mut p = Program::new();
    let one = p.float("one", 1.);
    let two = p.float("two", 2.);
    let out = p.float("out", 0.);
    let add = p.function("add", &[ArgSize::Scalar, ArgSize::Scalar], 0, |locals| {
        vec![
            stmt(O::AddF, locals, locals + 1, out),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let add_ptr = p.function_ptr(add);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreF, one, 4, 0),
            stmt(O::StoreF, two, 7, 0),
            stmt(O::Call2, add_ptr, 0, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    run(&vm, &mut TestContext::default(), main).unwrap();
    assert_eq!(read_float(&vm, out), 3.);
}

/// Builds a program where `call_without_args` calls `store` (which takes an argument) without any
/// arguments, after `set_param` has left a value in the first parameter slot.
fn missing_arguments_program(options: VmOptions) -> (QCVm, [i32; 2], u16) {
    let mut p = Program::new();
    let three = p.float("three", 3.);
    let out = p.float("out", 0.);
    let store = p.function("store", &[ArgSize::Scalar], 1, |locals| {
        vec![stmt(O::StoreF, locals, out, 0), stmt(O::Done, 0, 0, 0)]
    });
    let store_ptr = p.function_ptr(store);
    let set_param = p.function("set_param", &[], 0, |_| {
        vec![stmt(O::StoreF, three, 4, 0), stmt(O::Done, 0, 0, 0)]
    });
    let call_without_args = p.function("call_without_args", &[], 0, |_| {
        vec![stmt(O::Call0, store_ptr, 0, 0), stmt(O::Done, 0, 0, 0)]
    });
    (p.build_with(options), [set_param, call_without_args], out)
}

/// By default, calling a function with fewer arguments than it takes is an error.
#[test]
fn missing_arguments_error_by_default() {
    let (vm, [set_param, call_without_args], out) = missing_arguments_program(VmOptions::default());

    run(&vm, &mut TestContext::default(), set_param).unwrap();
    let error = format!(
        "{:#}",
        run(&vm, &mut TestContext::default(), call_without_args).unwrap_err()
    );
    assert!(
        error.contains("store called with 0 args, but it takes 1"),
        "{error}"
    );
    assert_eq!(read_float(&vm, out), 0.);

    // The same goes for calls from the host.
    let store = vm.functions().get_by_name(c"store").unwrap().index;
    let error = format!(
        "{:#}",
        run(&vm, &mut TestContext::default(), store).unwrap_err()
    );
    assert!(
        error.contains("store called with 0 args, but it takes 1"),
        "{error}"
    );
}

/// With [`MissingArguments::Stale`], parameter slots are global memory, so a function called with
/// fewer arguments than it takes gets the values left by the previous call.
#[test]
fn missing_arguments_use_stale_parameters_in_compat() {
    let (vm, [set_param, call_without_args], out) = missing_arguments_program(VmOptions {
        missing_arguments: MissingArguments::Stale,
    });

    run(&vm, &mut TestContext::default(), set_param).unwrap();
    run(&vm, &mut TestContext::default(), call_without_args).unwrap();
    assert_eq!(read_float(&vm, out), 3.);
}

/// Lists the calls in a real `progs.dat` (see [`real_progs_vm`]) that pass a QuakeC function fewer
/// arguments than it takes. These are bugs in the QuakeC, which only work with
/// [`MissingArguments::Stale`].
///
/// Only calls to constants (i.e. the function's own global) are checked, as other calls can't be
/// resolved statically. Builtins are variadic, so they aren't checked either.
#[test]
fn real_progs_missing_arguments() {
    let Some(vm) = real_progs_vm() else {
        return;
    };

    let functions = vm.functions();
    let mut found = vec![];
    for caller in functions.iter() {
        let crate::progs::functions::FunctionBody::Progs(body) = &caller.body else {
            continue;
        };

        for statement in body.statements.iter() {
            let Some(num_args) = (statement.opcode as usize).checked_sub(O::Call0 as usize) else {
                continue;
            };
            if num_args > O::Call8 as usize - O::Call0 as usize {
                continue;
            }

            let Ok(VmScalar::Function(crate::progs::VmFunctionRef::Ptr(ptr))) =
                vm.progs.globals.initial(statement.arg1 as u16 as usize)
            else {
                continue;
            };
            let Ok(callee) = functions.get_by_index(ptr.0) else {
                continue;
            };
            let is_constant = vm
                .global_defs()
                .iter()
                .any(|def| def.offset == statement.arg1 as u16 && def.name == callee.name);

            if is_constant && !callee.is_builtin() && num_args < callee.args.len() {
                found.push(format!(
                    "{} -> {} ({num_args} of {})",
                    caller.name.to_string_lossy(),
                    callee.name.to_string_lossy(),
                    callee.args.len()
                ));
            }
        }
    }

    // This is the only one in Quake's `id1/progs.dat`. If testing other progs, update this list.
    assert_eq!(found, ["ogre_smash12 -> ai_charge (0 of 1)"]);
}

/// Functions that `qcc` says have no locals (see [`QCFunctionDef::frame_range`]) still get their own
/// copy of their arguments, so a recursive call doesn't overwrite them.
#[test]
fn arguments_without_locals_are_per_frame() {
    let mut p = Program::new();
    // Function pointers are indices, so we know what this one will be.
    let rec_ptr = p.function_ptr(1);
    let one = p.float("one", 1.);
    let sum = p.float("sum", 0.);
    // void(float n) rec = { if (n) rec(n - 1); sum = sum + n; };
    let rec = p.function("rec", &[ArgSize::Scalar], 0, |n| {
        vec![
            jump(O::IfNot, n, 3),
            stmt(O::SubF, n, one, 4),
            stmt(O::Call1, rec_ptr, 0, 0),
            stmt(O::AddF, sum, n, sum),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    assert_eq!(rec, 1);
    let vm = p.build();

    vm.run(&mut TestContext::default(), rec, (2f32,)).unwrap();
    assert_eq!(read_float(&vm, sum), 3.);
}

/// Builtins are identified by number (`-first_statement`), not by their position in the function
/// table: numbers can have gaps (id1 has no `#5`), and several functions can alias one number
/// (LibreQuake's `bprint_e = #23`).
#[test]
fn builtins_are_identified_by_number() {
    let mut p = Program::new();
    let first = p.builtin("which_first", &[]);
    let bprint = p.numbered_builtin("which_bprint", 23);
    let bprint_e = p.numbered_builtin("which_bprint_e", 23);
    let spawn = p.numbered_builtin("which_spawn", 14);
    let ptrs = [first, bprint, bprint_e, spawn].map(|f| p.function_ptr(f));
    let main = p.function("main", &[], 0, |_| {
        let mut body = ptrs.map(|ptr| stmt(O::Call0, ptr, 0, 0)).to_vec();
        body.push(stmt(O::Done, 0, 0, 0));
        body
    });
    let vm = p.build();

    let numbers = [first, bprint, bprint_e, spawn]
        .map(|f| vm.functions().get_by_index(f).unwrap().builtin_number());
    assert_eq!(numbers, [Some(1), Some(23), Some(23), Some(14)]);
    assert_eq!(
        vm.functions().get_by_index(main).unwrap().builtin_number(),
        None
    );

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();
    assert_eq!(ctx.recorded, [1., 23., 23., 14.].map(Value::from).to_vec());
}

/// `progs.dat`s whose CRC isn't the defs' [`QCDefs::CRC`] are rejected.
#[test]
fn progs_crc_must_match_defs() {
    let header = |crc: i32| {
        let mut bytes = 6i32.to_le_bytes().to_vec();
        bytes.extend(crc.to_le_bytes());
        std::io::Cursor::new(bytes)
    };

    let err = QCVm::load::<TestDefs, _>(header(TestDefs::CRC + 1)).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match the expected CRC 5927"),
        "{err}"
    );

    // With the right CRC, loading gets past the header (and then fails as there are no lumps).
    let err = QCVm::load::<TestDefs, _>(header(TestDefs::CRC)).unwrap_err();
    assert!(!err.to_string().contains("CRC"), "{err}");
}

/// `fteqcc` writes 0 for the argument sizes of builtins, which Quake ignores.
#[test]
fn builtin_argument_sizes_are_ignored() {
    let mut sizes = [0; MAX_ARGS];
    let args = parse_arg_sizes(c"cvar", -45, 1, &sizes).unwrap();
    assert!(args.is_empty());

    sizes[..2].copy_from_slice(&[3, 1]);
    let args = parse_arg_sizes(c"setorigin", -2, 2, &sizes).unwrap();
    assert!(args.is_empty());

    // Too many arguments is still an error.
    parse_arg_sizes(c"cvar", -45, MAX_ARGS as i32 + 1, &sizes).unwrap_err();
}

/// QuakeC functions' argument sizes are used to copy their arguments, so must be valid.
#[test]
fn quakec_argument_sizes_are_checked() {
    let mut sizes = [0; MAX_ARGS];
    sizes[..2].copy_from_slice(&[3, 1]);
    let args = parse_arg_sizes(c"f", 10, 2, &sizes).unwrap();
    assert_eq!(args.as_slice(), [ArgSize::Vector, ArgSize::Scalar]);

    // Sizes past `argc` aren't read.
    sizes[2] = 0;
    parse_arg_sizes(c"f", 10, 2, &sizes).unwrap();

    let err = parse_arg_sizes(c"f", 10, 3, &sizes).unwrap_err();
    assert!(format!("{err}").contains("invalid argument size"), "{err}");
    sizes[2] = 2;
    parse_arg_sizes(c"f", 10, 3, &sizes).unwrap_err();
}

/// `fteqcc` emits definitions with types Quake doesn't know (e.g. `0xe` for the struct array
/// `intermission_levels[]` in LibreQuake), alongside definitions of their members.
#[test]
fn unknown_def_types_are_untyped() {
    assert_eq!(def_type("Global", c"s", 0xe), VmScalarType::Void.into());
    assert_eq!(def_type("Global", c"s", 9), VmScalarType::Void.into());
    assert_eq!(def_type("Global", c"s", 8), VmScalarType::Void.into());
    assert_eq!(def_type("Global", c"s", 1), VmScalarType::String.into());
    assert_eq!(def_type("Global", c"v", 3), VmType::Vector);

    // The struct and its first member share an offset, in either order.
    for struct_first in [true, false] {
        let mut p = Program::new();
        let id = p.string_id("start");
        let struct_def = GlobalDef {
            save: false,
            type_: def_type("Global", c"levels", 0xe),
            offset: p.next_global,
            name: cstring("levels"),
        };
        if struct_first {
            p.globals.push(struct_def.clone());
        }
        let member = p.string_with_id("levels[0].map_name", id);
        if !struct_first {
            p.globals.push(struct_def);
        }
        let vm = p.build();
        assert_eq!(
            vm.global(member).unwrap(),
            Value::try_from("start").unwrap()
        );
    }

    // The same goes for fields.
    for struct_first in [true, false] {
        let untyped = FieldDef {
            type_: def_type("Field", c"s", 0xe),
            offset: 0,
            name: cstring("s"),
        };
        let float = FieldDef {
            type_: VmScalarType::Float.into(),
            offset: 0,
            name: cstring("s.f"),
        };
        let defs = if struct_first {
            [untyped, float]
        } else {
            [float, untyped]
        };
        let entity_def = EntityTypeDef::new(defs);
        assert_eq!(entity_def.get(0).unwrap().type_, VmScalarType::Float);
    }
}

// Untyped globals =================================================================================

/// Optimizing compilers strip the defs of immediates, so globals without a typed def keep their
/// raw word, which is typed when it's read.
#[test]
fn untyped_globals_are_raw_bits() {
    let def = |name: &CStr, type_: VmType, offset: u16| GlobalDef {
        save: false,
        type_,
        offset,
        name: name.into(),
    };
    let raw = [
        0.5f32.to_le_bytes(),
        [0; 4],
        // Covered by a `void` def, which is untyped.
        0.25f32.to_le_bytes(),
        // Covered by a float def.
        10i32.to_le_bytes(),
        // Covered by a string def.
        1.5f32.to_le_bytes(),
    ]
    .concat();
    let globals = GlobalRegistry::new(
        [
            def(c"v", VmScalarType::Void.into(), 2),
            def(c"f", VmScalarType::Float.into(), 3),
            def(c"s", VmScalarType::String.into(), 4),
        ],
        &raw,
    )
    .unwrap();

    let initial = |i| globals.initial(i).unwrap().clone();
    assert_eq!(initial(0), VmScalar::Bits(0.5f32.to_bits()));
    assert_eq!(initial(1), VmScalar::Void);
    assert_eq!(initial(2), VmScalar::Bits(0.25f32.to_bits()));
    assert_eq!(initial(3), VmScalar::Float(f32::from_bits(10)));
    assert_eq!(
        initial(4),
        VmScalar::String(StringRef::Id(1.5f32.to_bits() as i32))
    );
}

#[test]
fn bits_convert_by_use() {
    use crate::progs::{FieldPtr, Ptr, VmFunctionRef};

    let bits = || VmScalar::Bits(10);
    assert_eq!(f32::try_from(bits()), Ok(f32::from_bits(10)));
    assert_eq!(StringRef::try_from(bits()), Ok(StringRef::Id(10)));
    assert_eq!(
        VmFunctionRef::try_from(bits()),
        Ok(VmFunctionRef::Ptr(Ptr(10)))
    );
    assert_eq!(FieldPtr::try_from(bits()), Ok(FieldPtr(Ptr(10))));
    assert!(EntityRef::try_from(bits()).is_err());

    assert!(!bits().is_null());
    assert!(!bits().is_false());
    // `-0.0` isn't null, but it is false.
    assert!(VmScalar::Bits((-0f32).to_bits()).is_false());
}

#[test]
fn untyped_float_operand() {
    let mut p = Program::new();
    let half = p.untyped(0.5f32.to_le_bytes());
    let one = p.float("one", 1.);
    let out = p.float("out", 0.);
    let (vm, result) = run_main(
        p,
        vec![stmt(O::AddF, half, one, out), stmt(O::Done, 0, 0, 0)],
        &mut TestContext::default(),
    );
    result.unwrap();
    assert_eq!(read_float(&vm, out), 1.5);
}

#[test]
fn untyped_string_operand() {
    let mut p = Program::new();
    let id = p.string_id("hello");
    let untyped = p.untyped(id.to_le_bytes());
    let typed = p.string_with_id("typed", id);
    let out = p.float("out", 0.);
    let (vm, result) = run_main(
        p,
        vec![stmt(O::EqS, untyped, typed, out), stmt(O::Done, 0, 0, 0)],
        &mut TestContext::default(),
    );
    result.unwrap();
    assert_eq!(read_float(&vm, out), 1.);
}

/// Untyped arguments are copied into the parameter slots as-is, and take the type of the builtin's
/// parameter. Without a type, they fall back to a float.
#[test]
fn untyped_builtin_arguments() {
    for (builtin, value, expected) in [
        ("record_string", None, Value::try_from("hello").unwrap()),
        ("record", Some(0.5f32), Value::Float(0.5)),
    ] {
        let mut p = Program::new();
        let record = p.builtin(builtin, &[ArgSize::Scalar]);
        let record_ptr = p.function_ptr(record);
        let bits = match value {
            Some(f) => f.to_le_bytes(),
            None => p.string_id("hello").to_le_bytes(),
        };
        let arg = p.untyped(bits);
        let main = p.function("main", &[], 0, |_| {
            vec![
                stmt(O::StoreF, arg, 4, 0),
                stmt(O::Call1, record_ptr, 0, 0),
                stmt(O::Done, 0, 0, 0),
            ]
        });
        let vm = p.build();

        let mut ctx = TestContext::default();
        run(&vm, &mut ctx, main).unwrap();
        assert_eq!(ctx.recorded, [expected], "{builtin}");
    }
}

#[test]
fn untyped_state_frame() {
    let mut p = Program::new();
    let frame = p.untyped(7f32.to_le_bytes());
    let think = p.function("think", &[], 0, |_| vec![stmt(O::Done, 0, 0, 0)]);
    let think_ptr = p.function_ptr(think);

    let mut ctx = TestContext::default();
    let (_, result) = run_main(
        p,
        vec![stmt(O::State, frame, think_ptr, 0), stmt(O::Done, 0, 0, 0)],
        &mut ctx,
    );
    result.unwrap();

    assert_eq!(ctx.states.len(), 1);
    assert_eq!(ctx.states[0].0, 7.);
    assert_eq!(function_index(&ctx.states[0].1), think);
}

/// A global initialized to 0 is `Void`, which is resolved to the zero of the builtin's parameter
/// type, as in Quake (e.g. `lightstyle(0, "m")` where the `0` is a shared zero global).
#[test]
fn void_builtin_argument_is_zero_of_parameter_type() {
    let mut p = Program::new();
    let record = p.builtin("record_float", &[ArgSize::Scalar]);
    let record_ptr = p.function_ptr(record);
    let zero = p.global("zero", VmScalarType::Void, &[[0; 4]]);
    let main = p.function("main", &[], 0, |_| {
        vec![
            stmt(O::StoreF, zero, 4, 0),
            stmt(O::Call1, record_ptr, 0, 0),
            stmt(O::Done, 0, 0, 0),
        ]
    });
    let vm = p.build();

    let mut ctx = TestContext::default();
    run(&vm, &mut ctx, main).unwrap();
    assert_eq!(ctx.recorded, [Value::Float(0.)]);
}

/// Every statement operand that is read with a fixed type, and whose initial value is untyped (see
/// [`untyped_globals_are_raw_bits`]), converts to that type, and refers to a valid string or
/// function. With an optimized `progs.dat` (see [`real_progs_vm`]), there are many such globals.
#[test]
fn real_progs_untyped_globals_convert() {
    use VmScalarType as T;

    let Some(vm) = real_progs_vm() else {
        return;
    };
    let globals = &vm.progs.globals;

    let mut errors = Vec::new();
    let mut checked = std::collections::HashSet::new();
    for function in vm.functions().iter() {
        let FunctionBody::Progs(body) = &function.body else {
            continue;
        };
        for statement in body.statements.iter() {
            let (a, b) = (statement.arg1, statement.arg2);
            // Vectors are 3 floats.
            let v = |addr: i16| [(addr, T::Float), (addr + 1, T::Float), (addr + 2, T::Float)];
            let reads: Vec<(i16, T)> = match statement.opcode {
                O::MulF
                | O::Div
                | O::AddF
                | O::SubF
                | O::EqF
                | O::NeF
                | O::Le
                | O::Ge
                | O::Lt
                | O::Gt
                | O::BitAnd
                | O::BitOr => vec![(a, T::Float), (b, T::Float)],
                O::MulV | O::AddV | O::SubV | O::EqV | O::NeV => [v(a), v(b)].concat(),
                O::MulFV => [&[(a, T::Float)][..], &v(b)].concat(),
                O::MulVF => [&v(a)[..], &[(b, T::Float)]].concat(),
                O::NotF | O::StorePF => vec![(a, T::Float)],
                O::NotV | O::StorePV => v(a).into(),
                O::EqS | O::NeS => vec![(a, T::String), (b, T::String)],
                O::NotS | O::StorePS => vec![(a, T::String)],
                O::EqEnt | O::NeEnt => vec![(a, T::Entity), (b, T::Entity)],
                O::NotEnt | O::StorePEnt => vec![(a, T::Entity)],
                O::EqFnc | O::NeFnc => vec![(a, T::Function), (b, T::Function)],
                O::NotFnc | O::StorePFnc => vec![(a, T::Function)],
                O::StorePFld => vec![(a, T::FieldRef)],
                O::LoadF
                | O::LoadV
                | O::LoadS
                | O::LoadEnt
                | O::LoadFld
                | O::LoadFnc
                | O::Address => vec![(a, T::Entity), (b, T::FieldRef)],
                O::Call0
                | O::Call1
                | O::Call2
                | O::Call3
                | O::Call4
                | O::Call5
                | O::Call6
                | O::Call7
                | O::Call8 => vec![(a, T::Function)],
                O::State => vec![(a, T::Float), (b, T::Function)],
                _ => vec![],
            };

            for (addr, ty) in reads {
                let addr = addr as u16 as usize;
                let Ok(&VmScalar::Bits(bits)) = globals.initial(addr) else {
                    continue;
                };
                checked.insert(addr);
                let result = VmScalar::bits_as(ty, bits)
                    .map_err(anyhow::Error::from)
                    .and_then(|value| match value {
                        VmScalar::String(s) => vm.progs.string_table.get(s).map(drop),
                        VmScalar::Function(crate::progs::VmFunctionRef::Ptr(ptr)) => {
                            vm.functions().get_by_index(ptr.0).map(drop)
                        }
                        _ => Ok(()),
                    });
                if let Err(e) = result {
                    errors.push(format!(
                        "{}: {:?} reads global {addr} ({bits:#x}) as {ty}: {e:#}",
                        function.name.to_string_lossy(),
                        statement.opcode,
                    ));
                }
            }
        }
    }

    eprintln!("Checked {} untyped globals", checked.len());
    assert!(errors.is_empty(), "{errors:#?}");
}
