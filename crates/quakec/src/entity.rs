use std::{convert::identity, sync::Arc};

use crate::progs::{FieldDef, FieldName, Ptr, ScalarType};

#[derive(Clone, Debug)]
pub struct ScalarFieldDef {
    pub name: FieldName,
    pub offset: u16,
    /// The type of the field.
    pub type_: ScalarType,
}

#[derive(Clone, Debug)]
pub struct EntityTypeDef {
    fields: Arc<[Option<ScalarFieldDef>]>,
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
            fields: fields.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, field_ref: Ptr) -> anyhow::Result<ScalarFieldDef> {
        let offset: usize = field_ref.0.try_into()?;

        self.fields
            .get(offset)
            .cloned()
            .and_then(identity)
            .ok_or_else(|| anyhow::Error::msg(format!("No field at the offset {offset}")))
    }
}
