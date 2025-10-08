pub type Scalar = [u8; 4];

pub enum Value {
    Scalar(Scalar),
    Vec3([Scalar; 3]),
}

pub trait IntoValue: Copy + 'static {
    fn into_seismon_value(self) -> Value;
}
