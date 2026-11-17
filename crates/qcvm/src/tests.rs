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
    Address, EmptyAddress, EntityRef, EntityTypeDef, FieldDef, FunctionIndex, GlobalDef, MAX_ARGS,
    QCVm, Type, Value, VectorField,
    load::{LoadFn, Progs},
    ops::Opcode,
    progs::{
        StringRef, StringTable, VmScalar, VmScalarType, VmType,
        functions::{ArgSize, FunctionRegistry, Statement},
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

    fn build(mut self) -> QCVm {
        // Make sure that the locals (and the arguments of functions without locals, which come
        // after them) are inside the globals lump.
        let end = self.next_local + 3 * MAX_ARGS as u16;
        self.set_value(end, [0; 4]);

        let raw = self.values.iter().flatten().copied().collect::<Vec<u8>>();
        QCVm {
            progs: Progs {
                globals: GlobalRegistry::new(self.globals, &raw).unwrap(),
                entity_def: EntityTypeDef::new(self.fields),
                string_table: StringTable::new(self.strings).unwrap(),
                functions: FunctionRegistry::new(self.statements.into(), &self.functions).unwrap(),
            },
        }
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
    type FieldAddr = TestFieldAddr;

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
    type GlobalAddr = EmptyAddress;

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
            // Calls its argument from inside a builtin, as e.g. `walkmove` runs touch functions.
            b"callback" => builtin(&[Type::Function, Type::Float], |ctx, args| {
                let Value::Function(func) = &args[0] else {
                    panic!("Expected a function, found {:?}", args[0]);
                };
                let index = (&**func as &dyn std::any::Any)
                    .downcast_ref::<FunctionIndex>()
                    .expect("QuakeC functions should be passed as `FunctionIndex`");
                ctx.call(*index, (args[1].clone(),)).unwrap()
            }),
            b"hello" => builtin(&[], |_, _| Value::try_from("hello").unwrap()),
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

fn run(vm: &QCVm, ctx: &mut TestContext, function: i32) -> anyhow::Result<Value> {
    vm.run(ctx, function, ())
}

/// Builds a program with a single function `main` (no arguments or locals) and runs it.
fn run_main(
    program: Program,
    body: Vec<Statement>,
    ctx: &mut TestContext,
) -> (QCVm, anyhow::Result<Value>) {
    let mut program = program;
    let main = program.function("main", &[], 0, |_| body);
    let vm = program.build();
    let result = run(&vm, ctx, main);
    (vm, result)
}

fn function_index(value: &Value) -> i32 {
    let Value::Function(func) = value else {
        panic!("Expected a function, found {value:?}");
    };
    (&**func as &dyn std::any::Any)
        .downcast_ref::<FunctionIndex>()
        .expect("Expected a `FunctionIndex`")
        .0
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

/// `IF`/`IFNOT` test the raw bits of their argument, as in Quake.
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
    assert_eq!(value, Value::Float(8.));
    let value = vm
        .run(&mut TestContext::default(), c"double", (5f32,))
        .unwrap();
    assert_eq!(value, Value::Float(10.));

    // The initial values of function globals are exposed as indices.
    assert_eq!(
        function_index(&vm.initial_global(double_ptr).unwrap()),
        double
    );
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

/// Storing `void` (e.g. an uninitialised temporary) is a store of zero.
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
    // Field references are passed to the host as their offset.
    assert_eq!(ctx.recorded, [Value::Float(0.)]);
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
    assert_eq!(value, Value::try_from("foo").unwrap());
    // Builtins can be run directly too.
    let value = run(&vm, &mut TestContext::default(), hello).unwrap();
    assert_eq!(value, Value::try_from("hello").unwrap());
}

/// Loads a real `progs.dat`, from the path in `QCVM_PROGS`, and runs `main`.
#[test]
#[ignore = "needs a real progs.dat in `QCVM_PROGS`"]
fn real_progs() {
    let path = std::env::var_os("QCVM_PROGS").expect("`QCVM_PROGS` should be set");
    let bytes = std::fs::read(path).unwrap();
    let vm = QCVm::load(std::io::Cursor::new(bytes)).unwrap();

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

/// Parameter slots are global memory, so a function called with fewer arguments than it takes gets
/// the values left by the previous call, as in Quake.
#[test]
fn missing_arguments_use_stale_parameters() {
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
    let vm = p.build();

    run(&vm, &mut TestContext::default(), set_param).unwrap();
    run(&vm, &mut TestContext::default(), call_without_args).unwrap();
    assert_eq!(read_float(&vm, out), 3.);
}
