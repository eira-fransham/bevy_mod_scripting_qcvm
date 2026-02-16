use std::{ffi::CStr, sync::Arc};

use crate::{HashMap, VectorField};
use itertools::Either;

use crate::progs::{GlobalDef, VmScalar, VmScalarType};

#[derive(Clone, Debug)]
pub struct ScalarGlobal {
    pub def: Arc<GlobalDef>,
    pub field: VectorField,

    /// Should be same as `self.value.type_()`, but may get out of sync due to `Void`.
    pub type_: VmScalarType,
    pub value: VmScalar,
}

impl ScalarGlobal {
    fn new(def: Arc<GlobalDef>) -> Either<Self, [Self; 3]> {
        match VmScalarType::try_from(def.type_) {
            Ok(type_) => Either::Left(ScalarGlobal {
                def,
                field: Default::default(),
                type_,
                value: VmScalar::Void,
            }),
            Err(tys_and_offsets) => {
                Either::Right(tys_and_offsets.map(|(type_, offset)| ScalarGlobal {
                    def: def.clone(),
                    field: offset,
                    type_,
                    value: VmScalar::Void,
                }))
            }
        }
    }

    fn with_value_bytes(self, bytes: [u8; 4]) -> anyhow::Result<Self> {
        let value = VmScalar::try_from_bytes(self.type_, bytes)?;

        Ok(Self { value, ..self })
    }
}

#[derive(Debug)]
pub struct GlobalRegistry {
    // TODO: There's usually (always?) a strict range that contains all globals, so this can just be a vec and offset
    globals: HashMap<u16, ScalarGlobal>,
    // TODO: Expose this.
    #[expect(dead_code, reason = "TODO: Expose globals by name")]
    by_name: HashMap<Arc<CStr>, u16>,
}

impl GlobalRegistry {
    /// Constructs a new `Globals` object.
    pub fn new<I>(defs: I, values: &[u8]) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = GlobalDef>,
    {
        let (globals, infos) = defs
            .into_iter()
            .map(Arc::new)
            .flat_map(|def| {
                let value = values.get(def.offset as usize * 4..).unwrap_or(&[0; 12]);

                match ScalarGlobal::new(def.clone()) {
                    Either::Left(scalar) => {
                        let name = def.name.clone();

                        Either::Left(std::iter::once(
                            value
                                .get(..4)
                                // A kinda-janky way of making this fail if there
                                // are less than 4 elements left.
                                .unwrap_or_default()
                                .try_into()
                                .map_err(anyhow::Error::from)
                                .and_then(|value| scalar.with_value_bytes(value))
                                .map(|scalar| ((def.offset, scalar), (name, def.offset))),
                        ))
                    }
                    Either::Right(vector) => match value
                        .get(..12)
                        // A kinda-janky way of making this fail if there
                        // are less than 12 elements left.
                        .unwrap_or_default()
                        .as_chunks::<4>()
                        .0
                        .try_into()
                    {
                        Ok(values) => {
                            let values: [[u8; 4]; 3] = values;
                            Either::Right(
                                std::array::from_fn::<_, 3, _>(|i| {
                                    vector[i].clone().with_value_bytes(values[i]).map(|scalar| {
                                        ((def.offset, scalar), (def.name.clone(), def.offset))
                                    })
                                })
                                .into_iter(),
                            )
                        }
                        Err(e) => Either::Left(std::iter::once(Err(e.into()))),
                    },
                }
            })
            .collect::<anyhow::Result<(_, _)>>()?;

        Ok(Self {
            globals,
            by_name: infos,
        })
    }

    pub fn get_with_index(&self, index: u16) -> anyhow::Result<&ScalarGlobal> {
        self.globals
            .get(&index)
            .ok_or_else(|| anyhow::format_err!("No global with index {index}"))
    }

    pub fn get_with_index_mut(&mut self, index: u16) -> anyhow::Result<&mut ScalarGlobal> {
        self.globals
            .get_mut(&index)
            .ok_or_else(|| anyhow::format_err!("No global with index {index}"))
    }

    #[inline]
    pub fn get_value<P>(&self, ptr: P) -> anyhow::Result<VmScalar>
    where
        P: TryInto<u16>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.get(ptr).map(|glob| glob.value.clone())
    }

    // TODO
    #[expect(dead_code)]
    #[inline]
    pub fn get_vector<I>(&self, index: I) -> anyhow::Result<[f32; 3]>
    where
        I: TryInto<u16>,
        I::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        let index = index.try_into()?;

        Ok([
            self.get_with_index(index)?.value.clone().try_into()?,
            self.get_with_index(index + 1)?.value.clone().try_into()?,
            self.get_with_index(index + 2)?.value.clone().try_into()?,
        ])
    }

    #[inline]
    pub fn get<P>(&self, ptr: P) -> anyhow::Result<&ScalarGlobal>
    where
        P: TryInto<u16>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.get_with_index(ptr.try_into()?)
    }

    // TODO
    #[inline]
    #[allow(dead_code)]
    pub fn get_mut<P>(&mut self, ptr: P) -> anyhow::Result<&mut ScalarGlobal>
    where
        P: TryInto<u16>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.get_with_index_mut(ptr.try_into()?)
    }
}
