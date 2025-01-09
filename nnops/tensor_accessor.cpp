#include <nnops/tensor_accessor.h>

namespace nnops {

class Tensor;

TensorAccessor::TensorAccessor(const Tensor &tensor) : tensor_(&tensor) {}

void *TensorAccessor::data_ptr_unsafe(const TensorShape &dims) {
    index_t offset = 0;
    for (int i = 0; i < tensor_->shape().size(); i++)
        offset += dims[i] * tensor_->stride()[i];
    return (void *)((char *)tensor_->data_ptr() + offset * tensor_->itemsize());
}

void *TensorAccessor::data_ptr_unsafe(const TensorShape &anchor, index_t offset, index_t dim) {
    char *ptr = (char *)data_ptr_unsafe(anchor);
    ptr += offset * tensor_->stride()[dim] * tensor_->itemsize();
    return (void *)ptr;
}

void *TensorAccessor::data_ptr_unsafe(const void *anchor_ptr, index_t offset, index_t dim) {
    char *ptr = (char *)anchor_ptr;
    ptr += offset * tensor_->stride()[dim] * tensor_->itemsize();
    return (void *)ptr;
}

} // namespace nnops