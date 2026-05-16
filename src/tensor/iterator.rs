use crate::common::{Index, TensorShape};
use crate::data_type::DataType;
use crate::tensor::meta::TensorMeta;
use crate::tensor::buffer::TensorBuffer;
use std::sync::Arc;

pub struct TensorIterator {
    meta: TensorMeta,
    index: TensorShape,
    offset: Index,
    buffer: Arc<TensorBuffer>,
}

impl TensorIterator {
    pub fn new(meta: &TensorMeta, buffer: Arc<TensorBuffer>) -> Self {
        let index = vec![0i64; meta.ndim()];
        TensorIterator {
            meta: meta.clone(),
            index,
            offset: 0,
            buffer,
        }
    }

    pub fn with_range(meta: &TensorMeta, buffer: Arc<TensorBuffer>, start: usize, stop: usize) -> Self {
        let mut new_meta = TensorMeta::new();
        new_meta.dims = meta.dims[start..stop].to_vec();
        new_meta.strides = meta.strides[start..stop].to_vec();
        new_meta.nelems = 1;
        for i in start..stop {
            new_meta.nelems *= meta.dims[i] as usize;
        }
        new_meta.dtype = meta.dtype;
        new_meta.offset = meta.offset;
        let index = vec![0i64; new_meta.ndim()];
        TensorIterator {
            meta: new_meta,
            index,
            offset: 0,
            buffer,
        }
    }

    pub fn with_range_offset(meta: &TensorMeta, buffer: Arc<TensorBuffer>, start: usize, stop: usize, offset: Index) -> Self {
        let mut iter = Self::with_range(meta, buffer, start, stop);
        iter.meta.offset += offset;
        iter
    }

    pub fn shape(&self) -> &TensorShape {
        &self.meta.dims
    }

    pub fn stride(&self) -> &TensorShape {
        &self.meta.strides
    }

    pub fn ndim(&self) -> usize {
        self.meta.ndim()
    }

    pub fn data_ptr(&self, offset: Index) -> *const u8 {
        unsafe {
            self.buffer.data_ptr().offset(
                ((self.meta.offset + offset) * self.meta.itemsize()) as isize
            )
        }
    }

    pub fn offset(&self) -> Index {
        self.offset
    }

    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    pub fn buffer(&self) -> Arc<TensorBuffer> {
        self.buffer.clone()
    }

    pub fn dtype(&self) -> DataType {
        self.meta.dtype
    }

    pub fn advance(&mut self) {
        let shape = &self.meta.dims;
        let stride = &self.meta.strides;
        let mut ax = self.ndim() as isize - 1;

        while ax >= 0 {
            if self.index[ax as usize] < shape[ax as usize] - 1 {
                self.index[ax as usize] += 1;
                self.offset += stride[ax as usize];
                return;
            } else {
                self.offset -= self.index[ax as usize] * stride[ax as usize];
                self.index[ax as usize] = 0;
                ax -= 1;
            }
        }

        self.offset = -1;
    }

    pub fn current(&self) -> *const u8 {
        self.data_ptr(self.offset)
    }

    pub fn is_end(&self) -> bool {
        self.offset == -1
    }

    pub fn end(&mut self) {
        self.offset = -1;
    }
}

impl Iterator for TensorIterator {
    type Item = *const u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_end() {
            return None;
        }
        let ptr = self.current();
        self.advance();
        Some(ptr)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_end() {
            return (0, Some(0));
        }
        let remaining = self.meta.nelems;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TensorIterator {}