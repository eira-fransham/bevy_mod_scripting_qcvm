use std::{
    ffi::CStr,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    sync::Arc,
};

use arc_slice::ArcSlice;
use arrayvec::ArrayVec;
use byteorder::{LittleEndian, ReadBytesExt};
use num::FromPrimitive as _;
use tracing::debug;

use crate::{
    entity::EntityTypeDef,
    progs::{
        FieldDef, GlobalDef, StringTable, VmScalarType, VmType,
        functions::{ArgSize, FunctionRegistry, MAX_ARGS, Statement},
        globals::GlobalRegistry,
    },
};

const VERSION: i32 = 6;
/// The CRC of the `progdefs.h` this crate's Quake bindings expect.
const CRC: i32 = 5927;
const LUMP_COUNT: usize = 6;
const SAVE_GLOBAL: u16 = 1 << 15;

// the on-disk size of a bytecode statement
const STATEMENT_SIZE: usize = 8;

// the on-disk size of a function declaration
const FUNCTION_SIZE: usize = 36;

// the on-disk size of a global or field definition
const DEF_SIZE: usize = 8;

enum LumpId {
    Statements = 0,
    GlobalDefs = 1,
    Fielddefs = 2,
    Functions = 3,
    Strings = 4,
    Globals = 5,
}

#[derive(Copy, Clone, Debug)]
struct Lump {
    offset: usize,
    count: usize,
}

#[derive(Debug)]
pub(crate) struct LoadFn {
    pub offset: i32,
    pub name: Arc<CStr>,
    pub source: Arc<CStr>,
    pub locals: Range<usize>,
    pub args: ArrayVec<ArgSize, MAX_ARGS>,
}

/// A loaded `progs.dat`.
#[derive(Debug)]
pub struct Progs {
    /// Global values for QuakeC bytecode.
    pub globals: GlobalRegistry,

    pub entity_def: EntityTypeDef,

    pub string_table: StringTable,

    /// Function definitions and bodies.
    pub functions: FunctionRegistry,
}

/// Checks that we have read exactly up to `expected`.
fn expect_position<R: Seek>(src: &mut R, expected: u64) -> anyhow::Result<()> {
    let position = src.stream_position()?;
    if position != expected {
        anyhow::bail!("Malformed progs.dat: expected to be at offset {expected}, found {position}");
    }

    Ok(())
}

/// Parses the argument sizes (`parm_size`) of a function definition.
///
/// Builtins get no argument sizes: Quake ignores them (builtins use the caller's argument count),
/// and `fteqcc` writes 0 for them. QuakeC functions must have valid sizes, as they're used to copy
/// arguments into the function's locals.
pub(crate) fn parse_arg_sizes(
    name: &CStr,
    offset: i32,
    argc: i32,
    arg_sizes: &[u8; MAX_ARGS],
) -> anyhow::Result<ArrayVec<ArgSize, MAX_ARGS>> {
    let argc = usize::try_from(argc)
        .ok()
        .filter(|argc| *argc <= MAX_ARGS)
        .ok_or_else(|| anyhow::format_err!("Function {name:?} has {argc} arguments"))?;

    if offset < 0 {
        return Ok(ArrayVec::new());
    }

    arg_sizes[..argc]
        .iter()
        .map(|byte| {
            ArgSize::from_u8(*byte).ok_or_else(|| {
                anyhow::format_err!("Function {name:?} has an invalid argument size ({byte})")
            })
        })
        .collect()
}

/// Converts the type of a global or field definition (without `DEF_SAVEGLOBAL`).
///
/// Types that Quake doesn't know (e.g. `fteqcc`'s struct and union types) are treated as untyped,
/// as Quake ignores them. `fteqcc` also emits definitions for the members of structs, which
/// supply the actual types.
pub(crate) fn def_type(kind: &str, name: &CStr, type_: u16) -> VmType {
    VmType::from_u16(type_).unwrap_or_else(|| {
        debug!("{kind} {name:?} has an unsupported type ({type_:#x}), treating it as untyped");
        VmType::Scalar(VmScalarType::Void)
    })
}

impl Progs {
    /// Loads all data from a `progs.dat` file.
    ///
    /// This returns objects representing the necessary context to execute QuakeC bytecode.
    pub fn load<R>(mut src: R) -> anyhow::Result<Progs>
    where
        R: Read + Seek,
    {
        let version = src.read_i32::<LittleEndian>()?;
        if version != VERSION {
            anyhow::bail!("progs version {version} not supported (expected {VERSION})");
        }

        // Quake only uses the CRC to check compatibility with `progdefs.h`, i.e. the layout of the
        // globals and fields that the engine accesses directly.
        let crc = src.read_i32::<LittleEndian>()?;
        if crc != CRC {
            anyhow::bail!(
                "progs CRC {crc} does not match the expected CRC {CRC}: this progs.dat uses an \
                 incompatible global/field layout"
            );
        }

        let mut lumps = [Lump {
            offset: 0,
            count: 0,
        }; LUMP_COUNT];

        for (i, lump) in lumps.iter_mut().enumerate() {
            *lump = Lump {
                offset: src.read_i32::<LittleEndian>()? as usize,
                count: src.read_i32::<LittleEndian>()? as usize,
            };

            debug!("{:?}: {:?}", i, lump);
        }

        let ent_addr_count = src.read_i32::<LittleEndian>()? as usize;
        debug!("Field count: {}", ent_addr_count);

        // Read string data and construct StringTable

        let string_lump = &lumps[LumpId::Strings as usize];
        src.seek(SeekFrom::Start(string_lump.offset as u64))?;
        let mut strings = Vec::new();
        (&mut src)
            .take(string_lump.count as u64)
            .read_to_end(&mut strings)?;
        let string_table = StringTable::new(strings)?;

        expect_position(&mut src, (string_lump.offset + string_lump.count) as u64)?;

        // Read function definitions and statements and construct Functions

        let statement_lump = &lumps[LumpId::Statements as usize];
        src.seek(SeekFrom::Start(statement_lump.offset as u64))?;
        let mut statements = Vec::with_capacity(statement_lump.count);
        for _ in 0..statement_lump.count {
            statements.push(Statement::new(
                src.read_i16::<LittleEndian>()?,
                src.read_i16::<LittleEndian>()?,
                src.read_i16::<LittleEndian>()?,
                src.read_i16::<LittleEndian>()?,
            )?);
        }

        expect_position(
            &mut src,
            (statement_lump.offset + statement_lump.count * STATEMENT_SIZE) as u64,
        )?;

        let statements: ArcSlice<[Statement]> = statements.into();

        let function_lump = &lumps[LumpId::Functions as usize];
        src.seek(SeekFrom::Start(function_lump.offset as u64))?;
        let mut load_functions = Vec::with_capacity(function_lump.count);

        for _ in 0..function_lump.count {
            let offset = src.read_i32::<LittleEndian>()?;

            let arg_start = usize::try_from(src.read_i32::<LittleEndian>()?)?;
            let locals = usize::try_from(src.read_i32::<LittleEndian>()?)?;

            // This is always 0
            let _ = src.read_i32::<LittleEndian>()?;

            let name_id = src.read_i32::<LittleEndian>()?;
            let srcfile_id = src.read_i32::<LittleEndian>()?;

            let name = string_table.get(name_id)?;
            let source = string_table.get(srcfile_id)?;

            let argc = src.read_i32::<LittleEndian>()?;
            let mut arg_size_buf = [0; MAX_ARGS];
            src.read_exact(&mut arg_size_buf)?;

            let args = parse_arg_sizes(&name, offset, argc, &arg_size_buf)?;

            load_functions.push(LoadFn {
                offset,
                name,
                locals: arg_start..arg_start + locals,
                source,
                args,
            });
        }

        expect_position(
            &mut src,
            (function_lump.offset + function_lump.count * FUNCTION_SIZE) as u64,
        )?;

        // Function pointers are indices into the function lump, so keep `load_functions` in order.
        let functions = FunctionRegistry::new(statements, &load_functions)?;

        let globaldef_lump = &lumps[LumpId::GlobalDefs as usize];
        src.seek(SeekFrom::Start(globaldef_lump.offset as u64))?;
        let mut global_defs = Vec::new();
        for _ in 0..globaldef_lump.count {
            let type_ = src.read_u16::<LittleEndian>()?;
            let offset = src.read_u16::<LittleEndian>()?;
            let name_id = src.read_i32::<LittleEndian>()?;
            let name = string_table.get(name_id)?;

            global_defs.push(GlobalDef {
                save: type_ & SAVE_GLOBAL != 0,
                type_: def_type("Global", &name, type_ & !SAVE_GLOBAL),
                offset,
                name,
            });
        }

        expect_position(
            &mut src,
            (globaldef_lump.offset + globaldef_lump.count * DEF_SIZE) as u64,
        )?;

        let fielddef_lump = &lumps[LumpId::Fielddefs as usize];
        src.seek(SeekFrom::Start(fielddef_lump.offset as u64))?;
        let mut field_defs = Vec::new();
        for _ in 0..fielddef_lump.count {
            let type_ = src.read_u16::<LittleEndian>()?;
            let offset = src.read_u16::<LittleEndian>()?;
            let name_id = src.read_i32::<LittleEndian>()?;

            let name = string_table.get(name_id)?;

            if type_ & SAVE_GLOBAL != 0 {
                return Err(anyhow::Error::msg(
                    "Save flag not allowed in field definitions",
                ));
            }
            field_defs.push(FieldDef {
                type_: def_type("Field", &name, type_),
                offset,
                name,
            });
        }

        expect_position(
            &mut src,
            (fielddef_lump.offset + fielddef_lump.count * DEF_SIZE) as u64,
        )?;

        let globals_lump = &lumps[LumpId::Globals as usize];
        src.seek(SeekFrom::Start(globals_lump.offset as u64))?;

        // if globals_lump.count < GLOBAL_STATIC_COUNT {
        //     return Err(ProgsError::with_msg(
        //         "Global count lower than static global count",
        //     ));
        // }

        let mut global_values = vec![0; globals_lump.count * 4];
        src.read_exact(&mut global_values)?;

        let globals = GlobalRegistry::new(global_defs, &global_values)?;

        let entity_def = EntityTypeDef::new(field_defs);

        Ok(Progs {
            globals,
            entity_def,
            string_table,
            functions,
        })
    }
}
