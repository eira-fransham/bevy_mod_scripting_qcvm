use std::{ffi::CStr, sync::Arc};

#[derive(Clone, Debug)]
pub struct StringTable {
    /// Interned string data.
    data: Arc<[u8]>,
}

impl StringTable {
    pub fn new<D: Into<Arc<[u8]>>>(data: D) -> StringTable {
        StringTable { data: data.into() }
    }

    pub fn get(&self, start: usize) -> Option<&CStr> {
        CStr::from_bytes_until_nul(self.data.get(start..)?).ok()
    }
}
