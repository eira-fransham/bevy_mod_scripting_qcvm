use std::{ffi::CStr, sync::Arc};

use crate::progs::{FieldDef, Ptr, VectorField, VmScalarType};

#[derive(Clone, Debug)]
pub struct ScalarFieldDef {
    pub field: Option<VectorField>,
    /// The type of the field.
    pub type_: VmScalarType,
    pub def: FieldDef,
}

impl ScalarFieldDef {
    pub fn offset(&self) -> u16 {
        self.def.offset + self.field.map(|f| f as u16).unwrap_or_default()
    }

    pub fn name(&self) -> Arc<CStr> {
        self.def.name.clone()
    }
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
            let offset = scalar_def.offset() as usize;
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

    pub fn get(&self, field_ref: Ptr) -> anyhow::Result<&ScalarFieldDef> {
        let offset: usize = field_ref.0.try_into()?;

        self.scalars
            .get(offset)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::Error::msg(format!("No field at the offset {offset}")))
    }
}
