use std::{ffi::CStr, sync::Arc};

use crate::progs::{FieldDef, VectorField, VmScalarType, VmType};

/// A single scalar of an entity field: either a scalar field, or one component of a vector field.
#[derive(Clone, Debug)]
pub struct ScalarFieldDef {
    /// For components of vector fields, the component. Otherwise, [`VectorField::XOrScalar`].
    pub field: VectorField,
    /// The type of the scalar.
    pub type_: VmScalarType,
    /// The definition of the whole field.
    pub def: FieldDef,
}

impl ScalarFieldDef {
    /// The offset of this scalar.
    pub fn offset(&self) -> u16 {
        self.def.offset + self.field as u16
    }

    /// The name of the whole field.
    pub fn name(&self) -> Arc<CStr> {
        self.def.name.clone()
    }

    /// Whether this is a component of a vector field.
    pub fn is_vector_component(&self) -> bool {
        self.def.type_ == VmType::Vector
    }
}

/// The entity fields defined by a `progs.dat`, indexed by offset.
#[derive(Clone, Debug)]
pub struct EntityTypeDef {
    defs: Arc<[FieldDef]>,
    scalars: Arc<[Option<ScalarFieldDef>]>,
}

impl EntityTypeDef {
    /// Builds the field layout from the field definitions in a `progs.dat`.
    ///
    /// `qcc` defines `foo_x`, `foo_y` and `foo_z` float fields aliasing each vector field `foo`. When
    /// fields overlap like this, the vector field is used, so that vector loads find it.
    pub fn new<F>(field_defs: F) -> EntityTypeDef
    where
        F: IntoIterator<Item = FieldDef>,
    {
        let defs = field_defs.into_iter().collect::<Vec<_>>();
        let mut scalars: Vec<Option<ScalarFieldDef>> = Vec::new();

        for def in &defs {
            let parts = match def.to_scalar() {
                Ok(scalar) => vec![scalar],
                Err(scalars) => scalars.into(),
            };

            for scalar in parts {
                let offset = scalar.offset() as usize;
                if scalars.len() <= offset {
                    scalars.resize_with(offset + 1, || None);
                }

                let slot = &mut scalars[offset];
                let keep_existing = slot.as_ref().is_some_and(|existing| {
                    existing.is_vector_component() && !scalar.is_vector_component()
                });
                if !keep_existing {
                    *slot = Some(scalar);
                }
            }
        }

        Self {
            defs: defs.into(),
            scalars: scalars.into(),
        }
    }

    /// The scalar at a given field offset, if a field is defined there.
    pub fn get(&self, offset: u16) -> Option<&ScalarFieldDef> {
        self.scalars.get(offset as usize)?.as_ref()
    }

    /// All field definitions, in the order they appear in the `progs.dat`.
    pub fn field_defs(&self) -> &[FieldDef] {
        &self.defs
    }
}
