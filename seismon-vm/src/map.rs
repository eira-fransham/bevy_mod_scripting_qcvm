use std::{
    borrow::Borrow,
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
};

pub struct MemoHash<T, H = DefaultHasher> {
    value: T,
    hash: u64,
    _phantom: PhantomData<H>,
}

impl<T, H> MemoHash<T, H>
where
    T: Hash,
    H: Default + Hasher,
{
    pub fn new(value: T) -> Self {
        value.into()
    }
}

impl<T, H> MemoHash<T, H> {
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T, H> From<T> for MemoHash<T, H>
where
    T: Hash,
    H: Default + Hasher,
{
    fn from(value: T) -> Self {
        let hash = BuildHasherDefault::<H>::new().hash_one(&value);

        Self {
            value,
            hash,
            _phantom: PhantomData,
        }
    }
}

impl<T, H> Deref for MemoHash<T, H> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, U, H> AsRef<U> for MemoHash<T, H>
where
    T: AsRef<U>,
{
    fn as_ref(&self) -> &U {
        self.value.as_ref()
    }
}

impl<T, H> Borrow<T> for MemoHash<T, H> {
    fn borrow(&self) -> &T {
        &self.value
    }
}

struct BuildIdentityHasher<H = DefaultHasher> {
    _phantom: PhantomData<H>,
}

impl<H: Hasher> Default for BuildIdentityHasher<H> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<H: Hasher> BuildHasher for BuildIdentityHasher<H> {
    type Hasher = IdentityHasher<H>;

    // Required method
    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher {
            value: 0,
            _phantom: PhantomData,
        }
    }
}

struct IdentityHasher<H = DefaultHasher> {
    value: u64,
    _phantom: PhantomData<H>,
}

impl<H> Hasher for IdentityHasher<H>
where
    H: Hasher,
{
    fn finish(&self) -> u64 {
        self.value
    }

    fn write(&mut self, bytes: &[u8]) {
        self.value = u64::from_ne_bytes(bytes.try_into().unwrap());
    }

    fn write_u64(&mut self, i: u64) {
        self.value = i;
    }

    fn write_u8(&mut self, i: u8) {
        self.value = i as u64;
    }

    fn write_u16(&mut self, i: u16) {
        self.value = i as u64;
    }

    fn write_u32(&mut self, i: u32) {
        self.value = i as u64;
    }

    fn write_u128(&mut self, i: u128) {
        self.value = i as u64;
    }

    fn write_usize(&mut self, i: usize) {
        self.value = i as u64;
    }

    fn write_i8(&mut self, i: i8) {
        self.value = i as u64;
    }

    fn write_i16(&mut self, i: i16) {
        self.value = i as u64;
    }

    fn write_i32(&mut self, i: i32) {
        self.value = i as u64;
    }

    fn write_i64(&mut self, i: i64) {
        self.value = i as u64;
    }

    fn write_i128(&mut self, i: i128) {
        self.value = i as u64;
    }

    fn write_isize(&mut self, i: isize) {
        self.value = i as u64;
    }
}
