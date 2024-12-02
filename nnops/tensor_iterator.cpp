#include <nnops/tensor_iterator.h>

namespace nnops {

TensorIterator::TensorIterator(const Tensor &tensor) {
    shape_ = tensor.shape();
    offset_ = TensorShape(shape_.size(), 0);
    stride_ = tensor.stride();
}

} // namespace nnops