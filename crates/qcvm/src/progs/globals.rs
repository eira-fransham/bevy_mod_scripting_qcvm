use parking_lot::RwLock;

use crate::progs::{GlobalDef, VmScalar, VmScalarType, VmType};

/// QuakeC global memory that isn't provided by the host.
///
/// This covers every address in the `progs.dat` globals lump, not just the ones with definitions:
/// `qcc` allocates temporaries as globals without definitions, and optimizing compilers (e.g.
/// fteqcc) strip the definitions of immediates. Slots without a typed definition hold their raw
/// word (see [`VmScalar::Bits`]), which is typed lazily by whatever reads it. Locals are stored per
/// call frame instead, see `FunctionExecutionCtx`.
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
    pub fn new<I>(defs: I, raw: &[u8]) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = GlobalDef>,
    {
        let mut defs = defs.into_iter().collect::<Vec<_>>();
        defs.sort_by_key(|def| def.offset);

        let (chunks, rest) = raw.as_chunks::<4>();
        if !rest.is_empty() {
            anyhow::bail!("Globals lump size ({}) is not a multiple of 4", raw.len());
        }

        let mut initial = chunks
            .iter()
            .map(|chunk| VmScalar::from_bits(*chunk))
            .collect::<Vec<_>>();

        for def in &defs {
            let types: &[VmScalarType] = match def.type_ {
                // Already initialized as untyped. Skipping these also means they don't overwrite a
                // typed def at the same offset (e.g. fteqcc's struct defs share an offset with
                // their first member).
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
            }
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
