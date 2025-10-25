use std::{ffi::CStr, sync::Arc};

use hashbrown::HashMap;
use itertools::Either;

use crate::progs::{FieldName, GlobalDef, Scalar, ScalarType};

#[derive(Clone, Debug)]
pub struct Global {
    pub name: FieldName,

    /// Should be same as `self.value.type_()`, but may get out of sync due to `Void`.
    pub type_: ScalarType,
    pub value: Scalar,
}

impl Global {
    fn new(def: &GlobalDef) -> Either<Self, [Self; 3]> {
        match ScalarType::try_from(def.type_) {
            Ok(type_) => Either::Left(Global {
                name: def.name.clone().into(),
                type_,
                value: Scalar::Void,
            }),
            Err(tys_and_offsets) => Either::Right(tys_and_offsets.map(|(type_, offset)| Global {
                name: FieldName {
                    name: def.name.clone(),
                    offset: Some(offset),
                },
                type_,
                value: Scalar::Void,
            })),
        }
    }

    fn with_value_bytes(self, bytes: [u8; 4]) -> anyhow::Result<Self> {
        let value = Scalar::try_from_bytes(self.type_, bytes)?;

        Ok(Self { value, ..self })
    }
}

#[derive(Debug)]
pub struct GlobalRegistry {
    globals: HashMap<u16, Global>,
    // TODO: Expose this.
    #[expect(dead_code)]
    infos: HashMap<Arc<CStr>, u16>,
}

impl GlobalRegistry {
    /// Constructs a new `Globals` object.
    pub fn new<I>(defs: I, values: &[u8]) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = GlobalDef>,
    {
        let (globals, infos) = defs
            .into_iter()
            .flat_map(|def| {
                let value = values.get(def.offset as usize * 4..).unwrap_or(&[0; 12]);

                match Global::new(&def) {
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

        Ok(Self { globals, infos })
    }

    pub fn get_with_index(&self, index: u16) -> anyhow::Result<&Global> {
        self.globals
            .get(&index)
            .ok_or_else(|| anyhow::format_err!("No global with index {index}"))
    }

    #[inline]
    pub fn get_value<P>(&self, ptr: P) -> anyhow::Result<Scalar>
    where
        P: TryInto<u16>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.get(ptr).map(|glob| glob.value.clone())
    }

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
    pub fn get<P>(&self, ptr: P) -> anyhow::Result<&Global>
    where
        P: TryInto<u16>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.get_with_index(ptr.try_into()?)
    }
}
