use crate::common::{Index, TensorShape};
use crate::tensor::Tensor;

pub struct TensorAccessor {
    tensor: Tensor,
}

impl TensorAccessor {
    pub fn new(tensor: &Tensor) -> Self {
        TensorAccessor { tensor: tensor.clone() }
    }

    pub fn data_ptr_unsafe(&self, dims: &TensorShape) -> *const u8 {
        let mut offset: Index = 0;
        for (i, &d) in dims.iter().enumerate().take(self.tensor.ndim()) {
            offset += d * self.tensor.stride_at(i as Index);
        }
        self.tensor.data_ptr_with_offset(offset)
    }

    pub fn data_ptr_unsafe_with_offset(&self, anchor: &TensorShape, offset: Index, dim: Index) -> *const u8 {
        let ptr = self.data_ptr_unsafe(anchor) as *mut u8;
        unsafe {
            ptr.offset((offset * self.tensor.stride_at(dim) * self.tensor.itemsize()) as isize) as *const u8
        }
    }

    pub fn data_ptr_unsafe_from_anchor(&self, anchor_ptr: *const u8, offset: Index, dim: Index) -> *const u8 {
        unsafe {
            (anchor_ptr as *mut u8).offset((offset * self.tensor.stride_at(dim) * self.tensor.itemsize()) as isize) as *const u8
        }
    }
}