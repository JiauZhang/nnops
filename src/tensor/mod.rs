mod meta;
mod buffer;
mod accessor;
mod iterator;
mod util;

pub use meta::{TensorMeta, Slice};
pub use buffer::TensorBuffer;
pub use accessor::TensorAccessor;
pub use iterator::TensorIterator;
pub use util::tensor_from;

use crate::common::{Index, TensorShape, TensorStride, is_broadcastable, is_broadcastable_to, broadcast_shape, broadcast_to};
use crate::data_type::{DataType, DtypeValue, type_cast};
use crate::device::{Device, DeviceType};
use std::sync::Arc;
use std::fmt;

pub struct Tensor {
    pub(crate) meta: TensorMeta,
    pub(crate) buffer: Arc<TensorBuffer>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            meta: self.meta.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

impl Tensor {
    pub fn new() -> Self {
        let device = Device::from_type(DeviceType::Cpu);
        Tensor {
            meta: TensorMeta::new(),
            buffer: Arc::new(TensorBuffer::new(&device, 0)),
        }
    }

    pub fn with_device(dtype: DataType, dims: &TensorShape, device: Device) -> Self {
        nnops_check!(!(dims.is_empty() || dims.iter().any(|&d| d <= 0)), "invalid shape info!");
        let meta = TensorMeta::with_dtype_shape(dtype, dims);
        let size = meta.nbytes();
        let buffer = Arc::new(TensorBuffer::new(&device, size));
        Tensor { meta, buffer }
    }

    pub fn with_device_type(dtype: DataType, dims: &TensorShape, device_type: DeviceType) -> Self {
        let device = Device::from_type(device_type);
        Self::with_device(dtype, dims, device)
    }

    pub fn with_meta_buffer(meta: TensorMeta, buffer: Arc<TensorBuffer>) -> Self {
        Tensor { meta, buffer }
    }

    pub fn fill_from(&mut self, value: &Tensor) {
        nnops_check!(
            is_broadcastable_to(&value.meta.dims, &self.meta.dims, 0),
            "could not broadcast tensor from shape {} into shape {}",
            value.meta.shape_as_string(),
            self.meta.shape_as_string()
        );
        let type_value = value.astype(self.meta.dtype).broadcast_to_shape(&self.meta.dims, 0);
        clone_impl(&type_value, 0, self, 0, 0);
    }

    pub fn dtype(&self) -> DataType {
        self.meta.dtype
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.buffer.data_ptr()
    }

    pub fn data_ptr_with_offset(&self, offset: Index) -> *const u8 {
        unsafe {
            self.buffer.data_ptr().offset((self.meta.offset + offset) as isize * self.itemsize() as isize)
        }
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        Arc::get_mut(&mut self.buffer).map(|b| b.data_mut_ptr()).unwrap_or_else(|| {
            panic!("cannot mutate tensor buffer: multiple references exist");
        })
    }

    pub fn ndim(&self) -> usize {
        self.meta.ndim()
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
    }

    pub fn device(&self) -> Device {
        Device::from_type(self.buffer.device_type())
    }

    pub fn device_type(&self) -> DeviceType {
        self.buffer.device_type()
    }

    pub fn nelems(&self) -> usize {
        self.meta.nelems
    }

    pub fn nbytes(&self) -> usize {
        self.meta.nbytes()
    }

    pub fn itemsize(&self) -> Index {
        self.meta.itemsize()
    }

    pub fn offset(&self) -> Index {
        self.meta.offset
    }

    pub fn is_contiguous(&self) -> bool {
        self.meta.is_contiguous()
    }

    pub fn shape(&self) -> &TensorShape {
        &self.meta.dims
    }

    pub fn shape_at(&self, index: Index) -> Index {
        let ndim = self.ndim() as Index;
        nnops_check!(index >= -ndim && index < ndim, "shape index is out of bounds");
        let idx = if index < 0 { (index + ndim) as usize } else { index as usize };
        self.meta.dims[idx]
    }

    pub fn stride(&self) -> &TensorStride {
        &self.meta.strides
    }

    pub fn stride_at(&self, index: Index) -> Index {
        let ndim = self.ndim() as Index;
        nnops_check!(index >= -ndim && index < ndim, "stride index is out of bounds");
        let idx = if index < 0 { (index + ndim) as usize } else { index as usize };
        self.meta.strides[idx]
    }

    pub fn clone_tensor(&self) -> Self {
        let mut tensor = Tensor::with_device_type(self.meta.dtype, &self.meta.dims, self.device_type());
        clone_impl(self, 0, &mut tensor, 0, 0);
        tensor
    }

    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            self.clone_tensor()
        }
    }

    pub fn astype(&self, dtype: DataType) -> Self {
        if self.meta.dtype == dtype {
            return self.clone();
        }
        let mut tensor = Tensor::with_device_type(dtype, &self.meta.dims, self.device_type());
        clone_impl(self, 0, &mut tensor, 0, 0);
        tensor
    }

    pub fn to_device(&self, device_type: DeviceType) -> Self {
        if device_type == self.device_type() {
            return self.clone();
        }
        let dev = Device::from_type(device_type);
        let mut dst = Tensor::with_device_type(self.meta.dtype, &self.meta.dims, device_type);
        let src = self.contiguous();
        if src.device_type() == DeviceType::Cpu {
            dev.copy_from_cpu(src.data_ptr(), dst.data_mut_ptr(), src.nbytes());
        } else {
            let src_dev = src.device();
            src_dev.copy_to_cpu(src.data_ptr(), dst.data_mut_ptr(), src.nbytes());
        }
        dst
    }

    pub fn reshape(&self, dims: &mut TensorShape) -> Self {
        let mut tensor = self.contiguous();
        tensor.meta.reshape_inplace(dims);
        tensor
    }

    pub fn reshape_inplace(&mut self, dims: &mut TensorShape) {
        self.meta.reshape_inplace(dims);
    }

    pub fn permute(&self, index: &TensorShape) -> Self {
        let meta = self.meta.permute(index);
        Tensor::with_meta_buffer(meta, self.buffer.clone())
    }

    pub fn transpose(&self, dim0: Index, dim1: Index) -> Self {
        let meta = self.meta.transpose(dim0, dim1);
        Tensor::with_meta_buffer(meta, self.buffer.clone())
    }

    pub fn is_broadcastable_tensors(t1: &Tensor, t2: &Tensor) -> bool {
        is_broadcastable(&t1.meta.dims, &t2.meta.dims, 0)
    }

    pub fn broadcast_shape_tensors(t1: &Tensor, t2: &Tensor) -> TensorShape {
        broadcast_shape(&t1.meta.dims, &t2.meta.dims, 0)
    }

    pub fn is_broadcast(&self) -> bool {
        self.meta.strides.iter().any(|&s| s == 0)
    }

    pub fn broadcast_to_shape(&self, shape: &TensorShape, offset: usize) -> Self {
        let (new_shape, new_strides) = broadcast_to(&self.meta.dims, &self.meta.strides, shape, offset);
        let mut meta = self.meta.clone();
        meta.dims = new_shape;
        meta.strides = new_strides;
        Tensor::with_meta_buffer(meta, self.buffer.clone())
    }

    pub fn to_string(&self) -> String {
        if self.meta.nelems == 0 {
            return String::new();
        }
        let mut ret = String::new();
        self.to_string_impl(&mut ret, 0, 0);
        ret
    }

    fn to_string_impl(&self, ret: &mut String, dim: usize, offset: Index) {
        if dim < self.ndim() - 1 {
            ret.push('[');
            for i in 0..self.meta.dims[dim] {
                self.to_string_impl(ret, dim + 1, offset + i * self.meta.strides[dim]);
            }
            let len = ret.len();
            if len >= 2 {
                let c = ret.as_bytes()[len - 2];
                if c == b',' as u8 {
                    ret.truncate(len - 1);
                }
            }
            if dim == 0 {
                let len = ret.len();
                if len >= 1 {
                    ret.truncate(len - 1);
                }
            } else {
                ret.push(',');
                ret.push('\n');
            }
            return;
        }

        ret.push('[');
        let data_ptr = self.data_ptr_with_offset(offset);
        let stride = self.meta.strides[dim] as isize * self.itemsize() as isize;
        for i in 0..self.meta.dims[dim] {
            unsafe {
                let ptr = data_ptr.offset(i as isize * stride);
                let val = read_as_f64(ptr, self.meta.dtype);
                ret.push_str(&format!("{}, ", val));
            }
        }
        let len = ret.len();
        if len >= 2 {
            ret.truncate(len - 1);
        }
        if dim == 0 {
            let len = ret.len();
            if len >= 1 {
                ret.truncate(len - 1);
            }
        } else {
            ret.push(',');
            ret.push('\n');
        }
    }

    pub fn to_repr(&self) -> String {
        let mut ret = "Tensor(".to_string();
        let s = self.to_string();
        ret.push_str(&s);
        ret.push(')');
        ret
    }

    pub fn shape_as_string(&self) -> String {
        self.meta.shape_as_string()
    }

    pub fn unravel_index(&self, idx: Index) -> TensorShape {
        crate::common::unravel_index(idx, &self.meta.dims)
    }

    pub fn ravel_index(&self, dims: &TensorShape) -> Index {
        crate::common::ravel_index(dims, &self.meta.dims)
    }

    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    pub fn buffer(&self) -> Arc<TensorBuffer> {
        self.buffer.clone()
    }

    pub fn as_slice<T: DtypeValue>(&self) -> Option<&[T]> {
        if !self.is_contiguous() || self.meta.dtype != T::DTYPE {
            return None;
        }
        let offset = (self.meta.offset * self.itemsize()) as usize;
        let byte_len = self.nelems() * std::mem::size_of::<T>();
        if offset + byte_len > self.buffer.size() {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts(
                self.buffer.data_ptr().add(offset) as *const T,
                self.nelems(),
            ))
        }
    }

    pub fn as_slice_mut<T: DtypeValue>(&mut self) -> Option<&mut [T]> {
        if !self.is_contiguous() || self.meta.dtype != T::DTYPE {
            return None;
        }
        let offset = (self.meta.offset * self.itemsize()) as usize;
        let nelems = self.nelems();
        let byte_len = nelems * std::mem::size_of::<T>();
        let buffer = Arc::get_mut(&mut self.buffer)?;
        if offset + byte_len > buffer.size() {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                buffer.data_mut_ptr().add(offset) as *mut T,
                nelems,
            ))
        }
    }

    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut(usize) -> f64,
    {
        match self.meta.dtype {
            DataType::Float32 => {
                if let Some(slice) = self.as_slice_mut::<f32>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as f32;
                    }
                }
            }
            DataType::Float64 => {
                if let Some(slice) = self.as_slice_mut::<f64>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i);
                    }
                }
            }
            DataType::Int32 => {
                if let Some(slice) = self.as_slice_mut::<i32>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as i32;
                    }
                }
            }
            DataType::Int64 => {
                if let Some(slice) = self.as_slice_mut::<i64>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as i64;
                    }
                }
            }
            DataType::Uint8 => {
                if let Some(slice) = self.as_slice_mut::<u8>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as u8;
                    }
                }
            }
            DataType::Int8 => {
                if let Some(slice) = self.as_slice_mut::<i8>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as i8;
                    }
                }
            }
            DataType::Uint16 => {
                if let Some(slice) = self.as_slice_mut::<u16>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as u16;
                    }
                }
            }
            DataType::Int16 => {
                if let Some(slice) = self.as_slice_mut::<i16>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as i16;
                    }
                }
            }
            DataType::Uint32 => {
                if let Some(slice) = self.as_slice_mut::<u32>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as u32;
                    }
                }
            }
            DataType::Uint64 => {
                if let Some(slice) = self.as_slice_mut::<u64>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = f(i) as u64;
                    }
                }
            }
            DataType::Bool => {
                if let Some(slice) = self.as_slice_mut::<u8>() {
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = if f(i) != 0.0 { 1 } else { 0 };
                    }
                }
            }
        }
    }
}

fn read_as_f64(ptr: *const u8, dtype: DataType) -> f64 {
    unsafe {
        match dtype {
            DataType::Bool => *ptr as f64,
            DataType::Uint8 => *(ptr as *const u8) as f64,
            DataType::Int8 => *(ptr as *const i8) as f64,
            DataType::Uint16 => *(ptr as *const u16) as f64,
            DataType::Int16 => *(ptr as *const i16) as f64,
            DataType::Uint32 => *(ptr as *const u32) as f64,
            DataType::Int32 => *(ptr as *const i32) as f64,
            DataType::Uint64 => *(ptr as *const u64) as f64,
            DataType::Int64 => *(ptr as *const i64) as f64,
            DataType::Float32 => *(ptr as *const f32) as f64,
            DataType::Float64 => *(ptr as *const f64),
        }
    }
}

pub fn clone_impl(src: &Tensor, src_offset: Index, dst: &mut Tensor, dst_offset: Index, axis: usize) {
    if axis < src.ndim() - 1 {
        for i in 0..src.meta.dims[axis] {
            clone_impl(
                src,
                src_offset + i * src.meta.strides[axis],
                dst,
                dst_offset + i * dst.meta.strides[axis],
                axis + 1,
            );
        }
        return;
    }

    let loop_size = src.meta.dims[axis];
    let src_slice = unsafe {
        std::slice::from_raw_parts(
            src.data_ptr_with_offset(src_offset),
            (loop_size as usize) * src.itemsize() as usize,
        )
    };
    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(
            dst.data_ptr_with_offset(dst_offset) as *mut u8,
            (loop_size as usize) * dst.itemsize() as usize,
        )
    };
    type_cast(
        dst_slice,
        src_slice,
        dst.meta.strides[axis] * dst.itemsize(),
        src.meta.strides[axis] * src.itemsize(),
        loop_size,
        src.meta.dtype,
        dst.meta.dtype,
    );
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={}, dtype={:?})", self.shape_as_string(), self.meta.dtype)
    }
}