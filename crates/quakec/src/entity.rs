use std::{convert::identity, error::Error, ffi::CStr, fmt, sync::Arc};

use hashbrown::HashMap;
use num_derive::FromPrimitive;

use crate::progs::{FieldDef, FieldName, Ptr, ScalarType};

pub const MAX_ENT_LEAVES: usize = 16;

pub const STATIC_ADDRESS_COUNT: usize = 105;

#[derive(Debug)]
pub enum EntityError {
    Io(::std::io::Error),
    Address(isize),
    Other(String),
    NoVacantSlots,
}

impl EntityError {
    pub fn with_msg<S>(msg: S) -> Self
    where
        S: AsRef<str>,
    {
        EntityError::Other(msg.as_ref().to_owned())
    }
}

impl fmt::Display for EntityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EntityError::Io(ref err) => {
                write!(f, "I/O error: ")?;
                err.fmt(f)
            }
            EntityError::Address(val) => write!(f, "Invalid address ({val})"),
            EntityError::Other(ref msg) => write!(f, "{msg}"),
            EntityError::NoVacantSlots => write!(f, "No vacant slots"),
        }
    }
}

impl Error for EntityError {}

impl From<::std::io::Error> for EntityError {
    fn from(error: ::std::io::Error) -> Self {
        EntityError::Io(error)
    }
}

#[derive(Clone, Debug)]
pub struct ScalarFieldInfo {
    pub name: FieldName,
    pub offset: u16,
    /// The type of the field.
    pub type_: ScalarType,
}

#[derive(Clone, Debug)]
pub struct EntityTypeDef {
    fields: Arc<[Option<ScalarFieldInfo>]>,
    offsets: HashMap<Arc<CStr>, Ptr>,
}

impl EntityTypeDef {
    /// `field_defs` should be sorted (see `impl Ord for FieldDef` for the precise ordering logic).
    pub fn new<F>(field_defs: F) -> Result<EntityTypeDef, EntityError>
    where
        F: IntoIterator<Item = FieldDef>,
    {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn get(&self, field_ref: Ptr) -> anyhow::Result<ScalarFieldInfo> {
        let offset: usize = field_ref.0.try_into()?;

        self.fields
            .get(offset)
            .cloned()
            .and_then(identity)
            .ok_or_else(|| anyhow::Error::msg(format!("No field at the offset {offset}")))
    }
}

#[derive(Debug, FromPrimitive, PartialEq)]
pub enum EntitySolid {
    Not = 0,
    Trigger = 1,
    BBox = 2,
    SlideBox = 3,
    Bsp = 4,
}
