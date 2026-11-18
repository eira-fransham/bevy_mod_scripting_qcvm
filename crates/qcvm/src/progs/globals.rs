use parking_lot::RwLock;

use tracing::debug;

use crate::progs::{GlobalDef, StringRef, VmScalar, VmScalarType, VmType};

/// QuakeC global memory that isn't provided by the host.
///
/// This covers every address in the `progs.dat` globals lump, not just the ones with definitions:
/// `qcc` allocates temporaries as globals without definitions. Locals are stored per call frame
/// instead, see `FunctionExecutionCtx`.
///
/// The values are behind a lock rather than requiring `&mut` access, since QuakeC can be re-entered
/// from a builtin (e.g. a touch function run by `walkmove`).
#[derive(Debug)]
pub struct GlobalRegistry {
    defs: Box<[GlobalDef]>,
    /// The values from the `progs.dat`, used to reset the globals and to initialize locals.
    initial: Box<[VmScalar]>,
    values: RwLock<Box<[VmScalar]>>,
}

impl GlobalRegistry {
    /// Builds the global memory from the definitions and the raw globals lump.
    ///
    /// `strings_len` is the size of the strings lump, used to classify globals without a
    /// definition (see [`classify_untyped`]).
    pub fn new<I>(defs: I, raw: &[u8], strings_len: usize) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = GlobalDef>,
    {
        let mut defs = defs.into_iter().collect::<Vec<_>>();
        defs.sort_by_key(|def| def.offset);

        let (chunks, rest) = raw.as_chunks::<4>();
        if !rest.is_empty() {
            anyhow::bail!("Globals lump size ({}) is not a multiple of 4", raw.len());
        }

        let mut initial = vec![VmScalar::Void; chunks.len()];
        let mut covered = vec![false; chunks.len()];

        for def in &defs {
            let types: &[VmScalarType] = match def.type_ {
                // Untyped defs (including fteqcc types we don't support) are classified like
                // globals without a def, see below.
                VmType::Scalar(VmScalarType::Void) => continue,
                VmType::Scalar(ty) => &[ty][..],
                VmType::Vector => &[VmScalarType::Float; 3][..],
            };

            for (i, ty) in types.iter().enumerate() {
                let offset = def.offset as usize + i;
                let bytes = chunks.get(offset).ok_or_else(|| {
                    anyhow::format_err!("Global {def} is outside of the globals lump")
                })?;

                initial[offset] = VmScalar::try_from_bytes(*ty, *bytes)
                    .map_err(|e| e.context(format!("Invalid initial value for global {def}")))?;
                covered[offset] = true;
            }
        }

        let (mut strings, mut floats) = (0usize, 0usize);
        for (offset, bytes) in chunks.iter().enumerate() {
            if covered[offset] {
                continue;
            }

            let value = classify_untyped(*bytes, strings_len);
            match value {
                VmScalar::String(_) => strings += 1,
                VmScalar::Float(_) => floats += 1,
                _ => {}
            }
            initial[offset] = value;
        }

        if strings + floats > 0 {
            debug!(
                "Classified {} nonzero globals without a typed def: {strings} strings, {floats} floats",
                strings + floats,
            );
        }

        Ok(Self {
            defs: defs.into(),
            values: RwLock::new(initial.clone().into()),
            initial: initial.into(),
        })
    }

    /// The global definitions, sorted by offset.
    pub fn defs(&self) -> &[GlobalDef] {
        &self.defs
    }

    /// The value of a global as specified in the `progs.dat`.
    pub fn initial(&self, index: usize) -> anyhow::Result<&VmScalar> {
        self.initial
            .get(index)
            .ok_or_else(|| anyhow::format_err!("Global {index} is out of range"))
    }

    /// The initial values of a range of globals, used to initialize locals.
    pub fn initial_range(&self, range: std::ops::Range<usize>) -> anyhow::Result<&[VmScalar]> {
        self.initial.get(range.clone()).ok_or_else(|| {
            anyhow::format_err!("Globals {}..{} are out of range", range.start, range.end)
        })
    }

    #[inline]
    pub fn get_value(&self, index: usize) -> anyhow::Result<VmScalar> {
        self.values
            .read()
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow::format_err!("Global {index} is out of range"))
    }

    #[inline]
    pub fn set_value(&self, index: usize, value: VmScalar) -> anyhow::Result<()> {
        *self
            .values
            .write()
            .get_mut(index)
            .ok_or_else(|| anyhow::format_err!("Global {index} is out of range"))? = value;

        Ok(())
    }

    /// Resets every global to its value in the `progs.dat`.
    pub fn reset(&self) {
        self.values.write().clone_from_slice(&self.initial);
    }
}

/// Infers the type of a global's initial value when it has no typed definition.
///
/// Optimizing compilers (e.g. `fteqcc -O2`) strip the definitions of immediates, so constants
/// such as `0.5`, `ATTN_STATIC` or `"knight/khurt.wav"` exist only as raw words in the globals
/// lump. Quake's globals are untyped, so the original engine reads them correctly regardless, but
/// we need a typed [`VmScalar`]. The only types an immediate can have that we can't tell apart from
/// a float are strings, and string immediates are offsets into the strings lump. As an `f32`, such
/// an offset is a positive denormal, which a real float constant never is, so:
///
/// - `0` is [`VmScalar::Void`], the zero of any type;
/// - a positive denormal that's in range of the strings lump is a string;
/// - anything else is a float.
///
/// Note that string offsets can point into the middle of another string (compilers merge common
/// suffixes), so we can't additionally require the offset to be the start of a string.
fn classify_untyped(bytes: [u8; 4], strings_len: usize) -> VmScalar {
    /// The smallest positive normal `f32`, as an `i32`.
    const MIN_NORMAL: i32 = 0x80_0000;

    let v = i32::from_le_bytes(bytes);
    let max_string = MIN_NORMAL.min(i32::try_from(strings_len).unwrap_or(i32::MAX));

    if v == 0 {
        VmScalar::Void
    } else if v > 0 && v < max_string {
        VmScalar::String(StringRef::Id(v))
    } else {
        VmScalar::Float(f32::from_le_bytes(bytes))
    }
}
