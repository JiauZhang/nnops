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
                let from_val = *($src.as_ptr().add(i * fs).cast::<$from_ty>());
                *($dst.as_mut_ptr().add(i * ts).cast::<$to_ty>()) = from_val as $to_ty;
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
                let from_val = *($src.as_ptr().add(i * fs).cast::<$from_ty>());
                let truncated = from_val.trunc() as i128;
                let wrapped = ((truncated + $min as i128).rem_euclid($range) + $min as i128) as $to_ty;
                *($dst.as_mut_ptr().add(i * ts).cast::<$to_ty>()) = wrapped;
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
                let from_val = *($src.as_ptr().add(i * fs).cast::<$from_ty>());
                *($dst.as_mut_ptr().add(i * ts).cast::<u8>()) = (from_val != 0.0) as u8;
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
                let from_val = *($src.as_ptr().add(i * fs).cast::<$from_ty>());
                *($dst.as_mut_ptr().add(i * ts).cast::<u8>()) = (from_val != 0) as u8;
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
                let from_val = *($src.as_ptr().add(i * fs).cast::<$from_ty>());
                let clipped = if from_val < 0.0 { 0.0 } else { from_val };
                let truncated = clipped.trunc();
                let result = if truncated > max_val as $from_ty { max_val } else { truncated as $to_ty };
                *($dst.as_mut_ptr().add(i * ts).cast::<$to_ty>()) = result;
            }
        }
    }};
}

macro_rules! define_type_cast_from_int_fn {
    ($fn_name:ident, $from_ty:ty) => {
        #[inline(always)]
        fn $fn_name(dst: &mut [u8], src: &[u8], dst_stride: Index, src_stride: Index, size: Index, to: DataType) {
            match to {
                DataType::Bool => type_cast_body_int_to_bool!(dst, src, dst_stride, src_stride, size, $from_ty),
                DataType::Uint8 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, u8),
                DataType::Int8 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, i8),
                DataType::Uint16 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, u16),
                DataType::Int16 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, i16),
                DataType::Uint32 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, u32),
                DataType::Int32 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, i32),
                DataType::Uint64 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, u64),
                DataType::Int64 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, i64),
                DataType::Float32 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, f32),
                DataType::Float64 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, f64),
            }
        }
    };
}

macro_rules! define_type_cast_from_float_fn {
    ($fn_name:ident, $from_ty:ty) => {
        #[inline(always)]
        fn $fn_name(dst: &mut [u8], src: &[u8], dst_stride: Index, src_stride: Index, size: Index, to: DataType) {
            match to {
                DataType::Bool => type_cast_body_float_to_bool!(dst, src, dst_stride, src_stride, size, $from_ty),
                DataType::Uint8 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, u8, 0, 256),
                DataType::Int8 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, i8, -128, 256),
                DataType::Uint16 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, u16, 0, 65536),
                DataType::Int16 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, i16, -32768, 65536),
                DataType::Uint32 => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, $from_ty, u32, 4294967295),
                DataType::Int32 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, i32, -2147483648, 4294967296),
                DataType::Uint64 => type_cast_body_float_to_uint!(dst, src, dst_stride, src_stride, size, $from_ty, u64, 18446744073709551615),
                DataType::Int64 => type_cast_body_float_to_int!(dst, src, dst_stride, src_stride, size, $from_ty, i64, -9223372036854775808, 18446744073709551616),
                DataType::Float32 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, f32),
                DataType::Float64 => type_cast_body!(dst, src, dst_stride, src_stride, size, $from_ty, f64),
            }
        }
    };
}

define_type_cast_from_int_fn!(type_cast_from_bool, u8);
define_type_cast_from_int_fn!(type_cast_from_uint8, u8);
define_type_cast_from_int_fn!(type_cast_from_int8, i8);
define_type_cast_from_int_fn!(type_cast_from_uint16, u16);
define_type_cast_from_int_fn!(type_cast_from_int16, i16);
define_type_cast_from_int_fn!(type_cast_from_uint32, u32);
define_type_cast_from_int_fn!(type_cast_from_int32, i32);
define_type_cast_from_int_fn!(type_cast_from_uint64, u64);
define_type_cast_from_int_fn!(type_cast_from_int64, i64);
define_type_cast_from_float_fn!(type_cast_from_float32, f32);
define_type_cast_from_float_fn!(type_cast_from_float64, f64);

#[inline]
pub fn type_cast(dst: &mut [u8], src: &[u8], dst_stride: Index, src_stride: Index, size: Index, from: DataType, to: DataType) {
    match from {
        DataType::Bool => type_cast_from_bool(dst, src, dst_stride, src_stride, size, to),
        DataType::Uint8 => type_cast_from_uint8(dst, src, dst_stride, src_stride, size, to),
        DataType::Int8 => type_cast_from_int8(dst, src, dst_stride, src_stride, size, to),
        DataType::Uint16 => type_cast_from_uint16(dst, src, dst_stride, src_stride, size, to),
        DataType::Int16 => type_cast_from_int16(dst, src, dst_stride, src_stride, size, to),
        DataType::Uint32 => type_cast_from_uint32(dst, src, dst_stride, src_stride, size, to),
        DataType::Int32 => type_cast_from_int32(dst, src, dst_stride, src_stride, size, to),
        DataType::Uint64 => type_cast_from_uint64(dst, src, dst_stride, src_stride, size, to),
        DataType::Int64 => type_cast_from_int64(dst, src, dst_stride, src_stride, size, to),
        DataType::Float32 => type_cast_from_float32(dst, src, dst_stride, src_stride, size, to),
        DataType::Float64 => type_cast_from_float64(dst, src, dst_stride, src_stride, size, to),
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
                let l = *($lhs.as_ptr().add(i * lhs_s).cast::<$type>());
                let r = *($rhs.as_ptr().add(i * rhs_s).cast::<$type>());
                *($out.as_mut_ptr().add(i * out_s).cast::<$type>()) = $func(l, r);
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

macro_rules! define_binary_op_on_type {
    ($fn_name:ident, $op_type:ty, $div_type:ty) => {
        #[inline(always)]
        fn $fn_name(
            op_type: ScalarBinaryOpType,
            out: &mut [u8], lhs: &[u8], rhs: &[u8],
            out_stride: Index, lhs_stride: Index, rhs_stride: Index, size: Index,
        ) {
            match op_type {
                ScalarBinaryOpType::Add => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, add_op, $op_type),
                ScalarBinaryOpType::Sub => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, sub_op, $op_type),
                ScalarBinaryOpType::Mul => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, mul_op, $op_type),
                ScalarBinaryOpType::Div => binary_op_body!(out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size, div_op, $div_type),
            }
        }
    };
}

define_binary_op_on_type!(binary_op_bool, u8, f64);
define_binary_op_on_type!(binary_op_uint8, u8, f64);
define_binary_op_on_type!(binary_op_int8, i8, f64);
define_binary_op_on_type!(binary_op_uint16, u16, f64);
define_binary_op_on_type!(binary_op_int16, i16, f64);
define_binary_op_on_type!(binary_op_uint32, u32, f64);
define_binary_op_on_type!(binary_op_int32, i32, f64);
define_binary_op_on_type!(binary_op_uint64, u64, f64);
define_binary_op_on_type!(binary_op_int64, i64, f64);
define_binary_op_on_type!(binary_op_float32, f32, f32);
define_binary_op_on_type!(binary_op_float64, f64, f64);

pub fn scalar_binary_op(
    op_type: ScalarBinaryOpType, dtype: DataType,
    out: &mut [u8], lhs: &[u8], rhs: &[u8],
    out_stride: Index, lhs_stride: Index, rhs_stride: Index, size: Index,
) {
    match dtype {
        DataType::Bool => binary_op_bool(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Uint8 => binary_op_uint8(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Int8 => binary_op_int8(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Uint16 => binary_op_uint16(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Int16 => binary_op_int16(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Uint32 => binary_op_uint32(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Int32 => binary_op_int32(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Uint64 => binary_op_uint64(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Int64 => binary_op_int64(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Float32 => binary_op_float32(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
        DataType::Float64 => binary_op_float64(op_type, out, lhs, rhs, out_stride, lhs_stride, rhs_stride, size),
    }
}