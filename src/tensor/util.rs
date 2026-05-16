use crate::tensor::Tensor;
use crate::tensor::iterator::TensorIterator;

pub fn tensor_from(iter: &TensorIterator) -> Tensor {
    Tensor::with_meta_buffer(iter.meta().clone(), iter.buffer())
}