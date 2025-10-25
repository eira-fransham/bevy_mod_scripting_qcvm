use std::sync::Arc;

use crate::progs::{FieldDef, FieldName, FieldOffset, Ptr, ScalarType, Type};

#[derive(Clone, Debug)]
pub struct ScalarFieldDef {
    pub name: FieldName,
    pub offset: u16,
    /// The type of the field.
    pub type_: ScalarType,
}

#[derive(Clone, Debug)]
pub struct EntityTypeDef {
    scalars: Arc<[Option<ScalarFieldDef>]>,
    // TODO: Expose this
    // offsets: HashMap<Arc<CStr>, Ptr>,
}

impl EntityTypeDef {
    /// `field_defs` should be sorted (see `impl Ord for FieldDef` for the precise ordering logic).
    pub fn new<F>(field_defs: F) -> EntityTypeDef
    where
        F: IntoIterator<Item = FieldDef>,
    {
        let mut fields = Vec::new();

        let mut push_def = |scalar_def: ScalarFieldDef| {
            let offset = scalar_def.offset as usize;
            if fields.len() <= offset {
                fields.resize_with(offset + 1, || None);
            }
            fields[offset] = Some(scalar_def);
        };

        for def in field_defs {
            match def.to_scalar() {
                Ok(scalar) => push_def(scalar),
                Err(scalars) => {
                    for scalar in scalars {
                        push_def(scalar);
                    }
                }
            }
        }

        Self {
            scalars: fields.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.scalars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_scalar(&self, field_ref: Ptr) -> anyhow::Result<&ScalarFieldDef> {
        let offset: usize = field_ref.0.try_into()?;

        self.scalars
            .get(offset)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::Error::msg(format!("No field at the offset {offset}")))
    }

    // TODO: It would be extremely easy to improve this, but it's good enough for now.
    pub fn get_vector(&self, field_ref: Ptr) -> anyhow::Result<FieldDef> {
        let [x, y, z] = std::array::from_fn(|i| self.get_scalar(Ptr(field_ref.0 + i as i32)));
        match [x?, y?, z?] {
            [
                ScalarFieldDef {
                    name:
                        FieldName {
                            name,
                            offset: Some(FieldOffset::X),
                            ..
                        },
                    offset,
                    ..
                },
                ScalarFieldDef {
                    name:
                        FieldName {
                            offset: Some(FieldOffset::Y),
                            ..
                        },
                    ..
                },
                ScalarFieldDef {
                    name:
                        FieldName {
                            offset: Some(FieldOffset::Z),
                            ..
                        },
                    ..
                },
            ] => Ok(FieldDef {
                type_: Type::Vector,
                offset: *offset,
                name: name.clone(),
            }),
            [
                ScalarFieldDef {
                    type_,
                    name: FieldName { name, .. },
                    ..
                },
                ..,
            ] => anyhow::bail!(
                "Type mismatch for {}, expected vector, found {}",
                name.to_string_lossy(),
                type_
            ),
        }
    }
}
