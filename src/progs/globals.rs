use std::{error::Error, ffi::CStr, fmt, sync::Arc};

use bevy_mod_scripting_bindings::{ScriptGlobalMakerFn, ScriptValue};
use hashbrown::HashMap;
use itertools::Either;

use crate::progs::{FieldName, GlobalDef, Ptr, ScalarKind, ScalarType, ValueKind};

pub const GLOBAL_STATIC_START: usize = 28;
pub const GLOBAL_DYNAMIC_START: usize = 64;

pub const GLOBAL_STATIC_COUNT: usize = GLOBAL_DYNAMIC_START - GLOBAL_STATIC_START;

pub const GLOBAL_ADDR_RETURN: usize = 1;
pub const GLOBAL_ADDR_ARG_START: usize = 4;

#[derive(Debug)]
pub enum GlobalsError {
    Io(::std::io::Error),
    Address(isize),
    Other(String),
}

impl GlobalsError {
    pub fn with_msg<S>(msg: S) -> Self
    where
        S: AsRef<str>,
    {
        GlobalsError::Other(msg.as_ref().to_owned())
    }
}

impl fmt::Display for GlobalsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GlobalsError::Io(ref err) => {
                write!(f, "I/O error: ")?;
                err.fmt(f)
            }
            GlobalsError::Address(val) => write!(f, "Invalid address ({val})"),
            GlobalsError::Other(ref msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for GlobalsError {}

impl From<::std::io::Error> for GlobalsError {
    fn from(error: ::std::io::Error) -> Self {
        GlobalsError::Io(error)
    }
}

pub trait GlobalAddr {
    /// The type of value referenced by this address.
    type Value;

    /// Loads the value at this address.
    fn load(&self, globals: &GlobalRegistry) -> Result<Self::Value, GlobalsError>;

    /// Stores a value at this address.
    fn store(&self, globals: &mut GlobalRegistry, value: Self::Value) -> Result<(), GlobalsError>;
}

pub enum Access {
    ReadOnly,
    ReadWrite,
}

pub enum External {
    ReadExt,
    ReadWriteExt,
    WriteExt,
}

pub struct GlobalValue {
    value: Result<ScalarKind, Arc<ScriptGlobalMakerFn<ScriptValue>>>,
}

#[derive(Clone, Debug)]
pub struct Global {
    pub name: FieldName,

    /// Should be same as `self.value.type_()`, but may get out of sync due to `Void`.
    pub type_: ScalarType,
    pub value: ScalarKind,
}

impl Global {
    fn new(def: &GlobalDef) -> Result<Self, [Self; 3]> {
        match ScalarType::try_from(def.type_) {
            Ok(type_) => Ok(Global {
                name: def.name.clone().into(),
                type_,
                value: ScalarKind::Void,
            }),
            Err(tys_and_offsets) => Err(tys_and_offsets.map(|(type_, offset)| Global {
                name: FieldName {
                    name: def.name.clone(),
                    offset: Some(offset),
                },
                type_,
                value: ScalarKind::Void,
            })),
        }
    }

    fn with_value_bytes(self, bytes: [u8; 4]) -> anyhow::Result<Self> {
        let value = ScalarKind::try_from_bytes(self.type_, bytes)?;

        Ok(Self { value, ..self })
    }
}

#[derive(Debug)]
pub struct GlobalRegistry {
    globals: HashMap<u16, Global>,
    infos: HashMap<Arc<CStr>, u16>,
}

impl GlobalRegistry {
    /// Constructs a new `Globals` object.
    pub fn new<'a, I>(defs: I, values: &[u8]) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = GlobalDef>,
    {
        let (globals, infos) = defs
            .into_iter()
            .flat_map(|def| {
                let value = values.get(def.offset as usize * 4..).unwrap_or(&[0; 12]);

                match Global::new(&def) {
                    Ok(scalar) => {
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
                    Err(vector) => match value
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
            .ok_or_else(|| anyhow::Error::msg(format!("No global with index {index}")))
    }

    #[inline]
    pub fn get_value<P>(&self, ptr: P) -> anyhow::Result<ScalarKind>
    where
        P: TryInto<Ptr>,
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
        P: TryInto<Ptr>,
        P::Error: snafu::Error + Into<anyhow::Error> + Send + Sync + 'static,
    {
        match ptr.try_into()? {
            Ptr::Id(index) => self.get_with_index(index.try_into()?),
            Ptr::Name(cstr) => {
                let offset = self
                    .infos
                    .get(&cstr)
                    .ok_or_else(|| anyhow::Error::msg(format!("No global with name {cstr:?}")))?;

                self.get_with_index(*offset)
            }
        }
    }
}
