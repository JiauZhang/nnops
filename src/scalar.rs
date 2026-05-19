use crate::common::Index;
use crate::data_type::{DataType, DtypeValue, sizeof_dtype, type_cast};
use crate::tensor::Tensor;
use crate::device::DeviceType;

#[derive(Debug, Clone)]
pub enum Scalar {
    Bool(bool),
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
}

impl Scalar {
    pub fn dtype(&self) -> DataType {
        match self {
            Scalar::Bool(_) => DataType::Bool,
            Scalar::Uint8(_) => DataType::Uint8,
            Scalar::Int8(_) => DataType::Int8,
            Scalar::Uint16(_) => DataType::Uint16,
            Scalar::Int16(_) => DataType::Int16,
            Scalar::Uint32(_) => DataType::Uint32,
            Scalar::Int32(_) => DataType::Int32,
            Scalar::Uint64(_) => DataType::Uint64,
            Scalar::Int64(_) => DataType::Int64,
            Scalar::Float32(_) => DataType::Float32,
            Scalar::Float64(_) => DataType::Float64,
        }
    }

    pub fn itemsize(&self) -> Index {
        sizeof_dtype(self.dtype())
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Scalar::Bool(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 1) },
            Scalar::Uint8(v) => unsafe { std::slice::from_raw_parts(v, 1) },
            Scalar::Int8(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 1) },
            Scalar::Uint16(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 2) },
            Scalar::Int16(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 2) },
            Scalar::Uint32(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 4) },
            Scalar::Int32(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 4) },
            Scalar::Float32(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 4) },
            Scalar::Uint64(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 8) },
            Scalar::Int64(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 8) },
            Scalar::Float64(v) => unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast(), 8) },
        }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Scalar::Bool(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 1) },
            Scalar::Uint8(v) => unsafe { std::slice::from_raw_parts_mut(v, 1) },
            Scalar::Int8(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 1) },
            Scalar::Uint16(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 2) },
            Scalar::Int16(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 2) },
            Scalar::Uint32(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 4) },
            Scalar::Int32(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 4) },
            Scalar::Float32(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 4) },
            Scalar::Uint64(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 8) },
            Scalar::Int64(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 8) },
            Scalar::Float64(v) => unsafe { std::slice::from_raw_parts_mut(std::ptr::from_mut(v).cast(), 8) },
        }
    }

    pub fn get<T: DtypeValue>(&self) -> Option<T> {
        if self.dtype() != T::DTYPE {
            return None;
        }
        let bytes = self.as_bytes();
        unsafe { Some(std::ptr::read_unaligned(bytes.as_ptr() as *const T)) }
    }

    pub fn astype(&self, dtype: DataType) -> Self {
        let mut result = zero_for_dtype(dtype);
        let src_slice = self.as_bytes();
        let dst_itemsize = result.itemsize();
        let src_itemsize = self.itemsize();
        let dst_slice = result.as_bytes_mut();
        type_cast(dst_slice, src_slice, dst_itemsize, src_itemsize, 1, self.dtype(), dtype);
        result
    }

    pub fn to_tensor(&self) -> Tensor {
        let t = Tensor::with_device_type(self.dtype(), &vec![1], DeviceType::Cpu);
        t.buffer.copy_from_cpu(self.as_bytes());
        t
    }
}

fn zero_for_dtype(dtype: DataType) -> Scalar {
    match dtype {
        DataType::Bool => Scalar::Bool(false),
        DataType::Uint8 => Scalar::Uint8(0),
        DataType::Int8 => Scalar::Int8(0),
        DataType::Uint16 => Scalar::Uint16(0),
        DataType::Int16 => Scalar::Int16(0),
        DataType::Uint32 => Scalar::Uint32(0),
        DataType::Int32 => Scalar::Int32(0),
        DataType::Uint64 => Scalar::Uint64(0),
        DataType::Int64 => Scalar::Int64(0),
        DataType::Float32 => Scalar::Float32(0.0),
        DataType::Float64 => Scalar::Float64(0.0),
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self { Scalar::Bool(v) }
}
impl From<u8> for Scalar {
    fn from(v: u8) -> Self { Scalar::Uint8(v) }
}
impl From<i8> for Scalar {
    fn from(v: i8) -> Self { Scalar::Int8(v) }
}
impl From<u16> for Scalar {
    fn from(v: u16) -> Self { Scalar::Uint16(v) }
}
impl From<i16> for Scalar {
    fn from(v: i16) -> Self { Scalar::Int16(v) }
}
impl From<u32> for Scalar {
    fn from(v: u32) -> Self { Scalar::Uint32(v) }
}
impl From<i32> for Scalar {
    fn from(v: i32) -> Self { Scalar::Int32(v) }
}
impl From<u64> for Scalar {
    fn from(v: u64) -> Self { Scalar::Uint64(v) }
}
impl From<i64> for Scalar {
    fn from(v: i64) -> Self { Scalar::Int64(v) }
}
impl From<f32> for Scalar {
    fn from(v: f32) -> Self { Scalar::Float32(v) }
}
impl From<f64> for Scalar {
    fn from(v: f64) -> Self { Scalar::Float64(v) }
}