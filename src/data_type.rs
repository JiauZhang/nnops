use crate::common::Index;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool = 0,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
}

pub const NUM_DATA_TYPES: usize = 11;

pub const fn sizeof_dtype(dtype: DataType) -> Index {
    match dtype {
        DataType::Bool => 1,
        DataType::Uint8 => 1,
        DataType::Int8 => 1,
        DataType::Uint16 => 2,
        DataType::Int16 => 2,
        DataType::Uint32 => 4,
        DataType::Int32 => 4,
        DataType::Uint64 => 8,
        DataType::Int64 => 8,
        DataType::Float32 => 4,
        DataType::Float64 => 8,
    }
}

pub trait DtypeValue: Copy + 'static {
    const DTYPE: DataType;
}

impl DtypeValue for bool { const DTYPE: DataType = DataType::Bool; }
impl DtypeValue for u8 { const DTYPE: DataType = DataType::Uint8; }
impl DtypeValue for i8 { const DTYPE: DataType = DataType::Int8; }
impl DtypeValue for u16 { const DTYPE: DataType = DataType::Uint16; }
impl DtypeValue for i16 { const DTYPE: DataType = DataType::Int16; }
impl DtypeValue for u32 { const DTYPE: DataType = DataType::Uint32; }
impl DtypeValue for i32 { const DTYPE: DataType = DataType::Int32; }
impl DtypeValue for u64 { const DTYPE: DataType = DataType::Uint64; }
impl DtypeValue for i64 { const DTYPE: DataType = DataType::Int64; }
impl DtypeValue for f32 { const DTYPE: DataType = DataType::Float32; }
impl DtypeValue for f64 { const DTYPE: DataType = DataType::Float64; }

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarBinaryOpType {
    Add = 0,
    Sub,
    Mul,
    Div,
}

pub const NUM_SCALAR_BINARY_OP_TYPES: usize = 4;

macro_rules! type_cast_body {
    ($dst:expr, $src:expr, $dst_stride:expr, $src_stride:expr, $size:expr, $from_ty:ty, $to_ty:ty) => {{
        let fs = $src_stride as usize;
        let ts = $dst_stride as usize;
        for i in 0..$size as usize {
            unsafe {
                let from_val = *($src.as_ptr().add(i * fs) as *const $from_ty);
                *($dst.as_mut_ptr().add(i * ts) as *mut $to_ty) = from_val as $to_ty;
            }
        }
    }};
}

macro_rules! type_cast_body_float_to_int {
    ($dst:expr, $src:expr, $dst_stride:expr, $src_stride:expr, $size:expr, $from_ty:ty, $to_ty:ty, $min:expr, $range:expr) => {{
        let fs = $src_stride as usize;
        let ts = $dst_stride as usize;
        for i in 0..$size as usize {
            unsafe {
                let from_val = *($src.as_ptr().add(i * fs) as *const $from_ty);
                let truncated = from_val.trunc() as i128;
                let wrapped = ((truncated + $min as i128).rem_euclid($range) + $min as i128) as $to_ty;
                *($dst.as_mut_ptr().add(i * ts) as *mut $to_ty) = wrapped;
            }
        }
    }};
}

macro_rules! type_cast_body_float_to_bool {
    ($dst:expr, $src:expr, $dst_stride:expr, $src_stride:expr, $size:expr, $from_ty:ty) => {{
        let fs = $src_stride as usize;
        let ts = $dst_stride as usize;
        for i in 0..$size as usize {
            unsafe {
                let from_val = *($src.as_ptr().add(i * fs) as *const $from_ty);
                *($dst.as_mut_ptr().add(i * ts) as *mut u8) = (from_val != 0.0) as u8;
            }
        }
    }};
}

macro_rules! type_cast_body_int_to_bool {
    ($dst:expr, $src:expr, $dst_stride:expr, $src_stride:expr, $size:expr, $from_ty:ty) => {{
        let fs = $src_stride as usize;
        let ts = $dst_stride as usize;
        for i in 0..$size as usize {
            unsafe {
                let from_val = *($src.as_ptr().add(i * fs) as *const $from_ty);
                *($dst.as_mut_ptr().add(i * ts) as *mut u8) = (from_val != 0) as u8;
            }
        }
    }};
}

macro_rules! type_cast_body_float_to_uint {
    ($dst:expr, $src:expr, $dst_stride:expr, $src_stride:expr, $size:expr, $from_ty:ty, $to_ty:ty, $max:expr) => {{
        let fs = $src_stride as usize;
        let ts = $dst_stride as usize;
        let max_val: $to_ty = $max;
        for i in 0..$size as usize {
            unsafe {
                let from_val = *($src.as_ptr().add(i * fs) as *const $from_ty);
                let clipped = if from_val < 0.0 { 0.0 } else { from_val };
                let truncated = clipped.trunc();
                let result = if truncated > max_val as $from_ty { max_val } else { truncated as $to_ty };
                *($dst.as_mut_ptr().add(i * ts) as *mut $to_ty) = result;
            }
        }
    }};
}

#[inline]
pub fn type_cast(dst: &mut [u8], src: &[u8], dst_stride: Index, src_stride: Index, size: Index, from: DataType, to: DataType) {
    match (from, to) {
        (DataType::Bool, DataType::Bool) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u8),
        (DataType::Bool, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u8),
        (DataType::Bool, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i8),
        (DataType::Bool, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u16),
        (DataType::Bool, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i16),
        (DataType::Bool, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u32),
        (DataType::Bool, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i32),
        (DataType::Bool, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u64),
        (DataType::Bool, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i64),
        (DataType::Bool, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, f32),
        (DataType::Bool, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, f64),

        (DataType::Uint8, DataType::Bool) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u8),
        (DataType::Uint8, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u8),
        (DataType::Uint8, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i8),
        (DataType::Uint8, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u16),
        (DataType::Uint8, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i16),
        (DataType::Uint8, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u32),
        (DataType::Uint8, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i32),
        (DataType::Uint8, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, u64),
        (DataType::Uint8, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, i64),
        (DataType::Uint8, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, f32),
        (DataType::Uint8, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u8, f64),

        (DataType::Int8, DataType::Bool) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, u8),
        (DataType::Int8, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, u8),
        (DataType::Int8, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, i8),
        (DataType::Int8, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, u16),
        (DataType::Int8, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, i16),
        (DataType::Int8, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, u32),
        (DataType::Int8, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, i32),
        (DataType::Int8, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, u64),
        (DataType::Int8, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, i64),
        (DataType::Int8, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, f32),
        (DataType::Int8, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i8, f64),

        (DataType::Uint16, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, u16),
        (DataType::Uint16, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, u8),
        (DataType::Uint16, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, i8),
        (DataType::Uint16, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, u16),
        (DataType::Uint16, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, i16),
        (DataType::Uint16, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, u32),
        (DataType::Uint16, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, i32),
        (DataType::Uint16, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, u64),
        (DataType::Uint16, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, i64),
        (DataType::Uint16, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, f32),
        (DataType::Uint16, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u16, f64),

        (DataType::Int16, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, i16),
        (DataType::Int16, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, u8),
        (DataType::Int16, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, i8),
        (DataType::Int16, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, u16),
        (DataType::Int16, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, i16),
        (DataType::Int16, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, u32),
        (DataType::Int16, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, i32),
        (DataType::Int16, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, u64),
        (DataType::Int16, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, i64),
        (DataType::Int16, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, f32),
        (DataType::Int16, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i16, f64),

        (DataType::Uint32, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, u32),
        (DataType::Uint32, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, u8),
        (DataType::Uint32, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, i8),
        (DataType::Uint32, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, u16),
        (DataType::Uint32, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, i16),
        (DataType::Uint32, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, u32),
        (DataType::Uint32, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, i32),
        (DataType::Uint32, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, u64),
        (DataType::Uint32, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, i64),
        (DataType::Uint32, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, f32),
        (DataType::Uint32, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u32, f64),

        (DataType::Int32, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, i32),
        (DataType::Int32, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, u8),
        (DataType::Int32, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, i8),
        (DataType::Int32, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, u16),
        (DataType::Int32, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, i16),
        (DataType::Int32, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, u32),
        (DataType::Int32, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, i32),
        (DataType::Int32, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, u64),
        (DataType::Int32, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, i64),
        (DataType::Int32, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, f32),
        (DataType::Int32, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i32, f64),

        (DataType::Uint64, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, u64),
        (DataType::Uint64, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, u8),
        (DataType::Uint64, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, i8),
        (DataType::Uint64, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, u16),
        (DataType::Uint64, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, i16),
        (DataType::Uint64, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, u32),
        (DataType::Uint64, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, i32),
        (DataType::Uint64, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, u64),
        (DataType::Uint64, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, i64),
        (DataType::Uint64, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, f32),
        (DataType::Uint64, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, u64, f64),

        (DataType::Int64, DataType::Bool) => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, i64),
        (DataType::Int64, DataType::Uint8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, u8),
        (DataType::Int64, DataType::Int8) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, i8),
        (DataType::Int64, DataType::Uint16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, u16),
        (DataType::Int64, DataType::Int16) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, i16),
        (DataType::Int64, DataType::Uint32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, u32),
        (DataType::Int64, DataType::Int32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, i32),
        (DataType::Int64, DataType::Uint64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, u64),
        (DataType::Int64, DataType::Int64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, i64),
        (DataType::Int64, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, f32),
        (DataType::Int64, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, i64, f64),

        (DataType::Float32, DataType::Bool) => type_cast_body_float_to_bool!(dst, src, dst_stride, src_stride, size, f32),
        (DataType::Float32, DataType::Uint8) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, u8, 0, 256),
        (DataType::Float32, DataType::Int8) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, i8, -128, 256),
        (DataType::Float32, DataType::Uint16) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, u16, 0, 65536),
        (DataType::Float32, DataType::Int16) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, i16, -32768, 65536),
        (DataType::Float32, DataType::Uint32) => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, f32, u32, 4294967295),
        (DataType::Float32, DataType::Int32) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, i32, -2147483648, 4294967296),
        (DataType::Float32, DataType::Uint64) => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, f32, u64, 18446744073709551615),
        (DataType::Float32, DataType::Int64) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f32, i64, -9223372036854775808, 18446744073709551616),
        (DataType::Float32, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, f32, f32),
        (DataType::Float32, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, f32, f64),

        (DataType::Float64, DataType::Bool) => type_cast_body_float_to_bool!(dst, src, dst_stride, src_stride, size, f64),
        (DataType::Float64, DataType::Uint8) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, u8, 0, 256),
        (DataType::Float64, DataType::Int8) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, i8, -128, 256),
        (DataType::Float64, DataType::Uint16) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, u16, 0, 65536),
        (DataType::Float64, DataType::Int16) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, i16, -32768, 65536),
        (DataType::Float64, DataType::Uint32) => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, f64, u32, 4294967295),
        (DataType::Float64, DataType::Int32) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, i32, -2147483648, 4294967296),
        (DataType::Float64, DataType::Uint64) => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, f64, u64, 18446744073709551615),
        (DataType::Float64, DataType::Int64) => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, f64, i64, -9223372036854775808, 18446744073709551616),
        (DataType::Float64, DataType::Float32) => type_cast_body!(dst, src, dst_stride, src_stride, size, f64, f32),
        (DataType::Float64, DataType::Float64) => type_cast_body!(dst, src, dst_stride, src_stride, size, f64, f64),
    }
}

static PROMOTE_TYPES: [[DataType; NUM_DATA_TYPES]; NUM_DATA_TYPES] = {
    use DataType::*;
    [
        /*       b1  u1  i1  u2  i2  u4  i4  u8  i8  f4  f8 */
        /* b1 */ [Bool, Uint8, Int8, Uint16, Int16, Uint32, Int32, Uint64, Int64, Float32, Float64],
        /* u1 */ [Uint8, Uint8, Int16, Uint16, Int16, Uint32, Int32, Uint64, Int64, Float32, Float64],
        /* i1 */ [Int8, Int16, Int8, Int32, Int16, Int64, Int32, Float64, Int64, Float32, Float64],
        /* u2 */ [Uint16, Uint16, Int32, Uint16, Int32, Uint32, Int32, Uint64, Int64, Float32, Float64],
        /* i2 */ [Int16, Int16, Int16, Int32, Int16, Int64, Int32, Float64, Int64, Float32, Float64],
        /* u4 */ [Uint32, Uint32, Int64, Uint32, Int64, Uint32, Int64, Uint64, Int64, Float64, Float64],
        /* i4 */ [Int32, Int32, Int32, Int32, Int32, Int64, Int32, Float64, Int64, Float64, Float64],
        /* u8 */ [Uint64, Uint64, Float64, Uint64, Float64, Uint64, Float64, Uint64, Float64, Float64, Float64],
        /* i8 */ [Int64, Int64, Int64, Int64, Int64, Int64, Int64, Float64, Int64, Float64, Float64],
        /* f4 */ [Float32, Float32, Float32, Float32, Float32, Float64, Float64, Float64, Float64, Float32, Float64],
        /* f8 */ [Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
    ]
};

pub fn get_promote_type(op_type: ScalarBinaryOpType, ltype: DataType, rtype: DataType) -> DataType {
    if op_type == ScalarBinaryOpType::Div
        && ltype != DataType::Float32
        && ltype != DataType::Float64
        && rtype != DataType::Float32
        && rtype != DataType::Float64
    {
        return DataType::Float64;
    }
    PROMOTE_TYPES[ltype as usize][rtype as usize]
}

macro_rules! binary_op_body {
    ($out:expr, $lhs:expr, $rhs:expr, $out_stride:expr, $lhs_stride:expr, $rhs_stride:expr, $size:expr, $func:ident, $type:ty) => {{
        let out_s = $out_stride as usize;
        let lhs_s = $lhs_stride as usize;
        let rhs_s = $rhs_stride as usize;
        for i in 0..$size as usize {
            unsafe {
                let l = *($lhs.as_ptr().add(i * lhs_s) as *const $type);
                let r = *($rhs.as_ptr().add(i * rhs_s) as *const $type);
                *($out.as_mut_ptr().add(i * out_s) as *mut $type) = $func(l, r);
            }
        }
    }};
}

trait WrappingArith: Copy {
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
}

macro_rules! impl_wrapping_int {
    ($ty:ty) => {
        impl WrappingArith for $ty {
            fn wrapping_add(self, other: Self) -> Self { self.wrapping_add(other) }
            fn wrapping_sub(self, other: Self) -> Self { self.wrapping_sub(other) }
            fn wrapping_mul(self, other: Self) -> Self { self.wrapping_mul(other) }
        }
    };
}

impl_wrapping_int!(u8);
impl_wrapping_int!(i8);
impl_wrapping_int!(u16);
impl_wrapping_int!(i16);
impl_wrapping_int!(u32);
impl_wrapping_int!(i32);
impl_wrapping_int!(u64);
impl_wrapping_int!(i64);

impl WrappingArith for f32 {
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
}

impl WrappingArith for f64 {
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
}

fn add_op<T: WrappingArith>(l: T, r: T) -> T { l.wrapping_add(r) }
fn sub_op<T: WrappingArith>(l: T, r: T) -> T { l.wrapping_sub(r) }
fn mul_op<T: WrappingArith>(l: T, r: T) -> T { l.wrapping_mul(r) }
fn div_op<T: std::ops::Div<Output = T>>(l: T, r: T) -> T { l / r }

pub fn scalar_binary_op(
    op_type: ScalarBinaryOpType, dtype: DataType,
    out: &mut [u8], lhs: &[u8], rhs: &[u8],
    out_stride: Index, lhs_stride: Index, rhs_stride: Index, size: Index,
) {
    match (op_type, dtype) {
        (ScalarBinaryOpType::Add, DataType::Bool) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, u8),
        (ScalarBinaryOpType::Add, DataType::Uint8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, u8),
        (ScalarBinaryOpType::Add, DataType::Int8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, i8),
        (ScalarBinaryOpType::Add, DataType::Uint16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, u16),
        (ScalarBinaryOpType::Add, DataType::Int16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, i16),
        (ScalarBinaryOpType::Add, DataType::Uint32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, u32),
        (ScalarBinaryOpType::Add, DataType::Int32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, i32),
        (ScalarBinaryOpType::Add, DataType::Uint64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, u64),
        (ScalarBinaryOpType::Add, DataType::Int64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, i64),
        (ScalarBinaryOpType::Add, DataType::Float32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, f32),
        (ScalarBinaryOpType::Add, DataType::Float64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, f64),

        (ScalarBinaryOpType::Sub, DataType::Bool) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, u8),
        (ScalarBinaryOpType::Sub, DataType::Uint8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, u8),
        (ScalarBinaryOpType::Sub, DataType::Int8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, i8),
        (ScalarBinaryOpType::Sub, DataType::Uint16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, u16),
        (ScalarBinaryOpType::Sub, DataType::Int16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, i16),
        (ScalarBinaryOpType::Sub, DataType::Uint32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, u32),
        (ScalarBinaryOpType::Sub, DataType::Int32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, i32),
        (ScalarBinaryOpType::Sub, DataType::Uint64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, u64),
        (ScalarBinaryOpType::Sub, DataType::Int64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, i64),
        (ScalarBinaryOpType::Sub, DataType::Float32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, f32),
        (ScalarBinaryOpType::Sub, DataType::Float64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, f64),

        (ScalarBinaryOpType::Mul, DataType::Bool) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, u8),
        (ScalarBinaryOpType::Mul, DataType::Uint8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, u8),
        (ScalarBinaryOpType::Mul, DataType::Int8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, i8),
        (ScalarBinaryOpType::Mul, DataType::Uint16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, u16),
        (ScalarBinaryOpType::Mul, DataType::Int16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, i16),
        (ScalarBinaryOpType::Mul, DataType::Uint32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, u32),
        (ScalarBinaryOpType::Mul, DataType::Int32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, i32),
        (ScalarBinaryOpType::Mul, DataType::Uint64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, u64),
        (ScalarBinaryOpType::Mul, DataType::Int64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, i64),
        (ScalarBinaryOpType::Mul, DataType::Float32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, f32),
        (ScalarBinaryOpType::Mul, DataType::Float64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, f64),

        (ScalarBinaryOpType::Div, DataType::Bool) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Uint8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Int8) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Uint16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Int16) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Uint32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Int32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Uint64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Int64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),
        (ScalarBinaryOpType::Div, DataType::Float32) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f32),
        (ScalarBinaryOpType::Div, DataType::Float64) => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, f64),

        }
}