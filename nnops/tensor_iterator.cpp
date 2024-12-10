#include <nnops/tensor_iterator.h>

namespace nnops {

class Tensor;

TensorIterator::TensorIterator(const Tensor &tensor) : tensor_(tensor) {
    index_ = TensorShape(tensor_.shape().size(), 0);
    offset_ = tensor_.meta().offset();
}

TensorIterator &TensorIterator::operator++() {
    const TensorShape &shape = tensor_.shape();
    const TensorStride &stride = tensor_.stride();
    int ax = shape.size() - 1;

    while (ax >= 0) {
        if (index_[ax] < shape[ax] - 1) {
            index_[ax]++;
            offset_ += stride[ax];
            return *this;
        } else {
            offset_ -= index_[ax] * stride[ax];
            index_[ax] = 0;
            ax--;
        }
    }

    this->end();

    return *this;
}

bool TensorIterator::operator!=(const TensorIterator &other) {
    return offset_ != other.offset();
}

void *TensorIterator::operator*() {
    char *data_ptr = reinterpret_cast<char *>(tensor_.data_ptr()) + offset_ * sizeof_dtype(tensor_.dtype());
    return reinterpret_cast<void *>(data_ptr);
}

} // namespace nnops