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

    pub fn data_ptr(&self) -> *const u8 {
        match self {
            Scalar::Bool(v) => (v as *const bool) as *const u8,
            Scalar::Uint8(v) => v as *const u8,
            Scalar::Int8(v) => (v as *const i8) as *const u8,
            Scalar::Uint16(v) => (v as *const u16) as *const u8,
            Scalar::Int16(v) => (v as *const i16) as *const u8,
            Scalar::Uint32(v) => (v as *const u32) as *const u8,
            Scalar::Int32(v) => (v as *const i32) as *const u8,
            Scalar::Uint64(v) => (v as *const u64) as *const u8,
            Scalar::Int64(v) => (v as *const i64) as *const u8,
            Scalar::Float32(v) => (v as *const f32) as *const u8,
            Scalar::Float64(v) => (v as *const f64) as *const u8,
        }
    }

    pub fn itemsize(&self) -> Index {
        sizeof_dtype(self.dtype())
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.data_ptr(), self.itemsize() as usize)
        }
    }

    pub fn get<T: DtypeValue>(&self) -> Option<T> {
        if self.dtype() != T::DTYPE {
            return None;
        }
        unsafe {
            Some(std::ptr::read(self.data_ptr() as *const T))
        }
    }

    pub fn astype(&self, dtype: DataType) -> Self {
        let result = zero_for_dtype(dtype);
        let src_slice = self.as_bytes();
        let dst_slice = unsafe {
            std::slice::from_raw_parts_mut(result.data_ptr() as *mut u8, result.itemsize() as usize)
        };
        type_cast(dst_slice, src_slice, result.itemsize(), self.itemsize(), 1, self.dtype(), dtype);
        result
    }

    pub fn to_tensor(&self) -> Tensor {
        let t = Tensor::with_device_type(self.dtype(), &vec![1], DeviceType::Cpu);
        let src_slice = self.as_bytes();
        let dst_slice = unsafe {
            std::slice::from_raw_parts_mut(t.data_ptr() as *mut u8, t.itemsize() as usize)
        };
        type_cast(dst_slice, src_slice, t.itemsize(), self.itemsize(), 1, self.dtype(), self.dtype());
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