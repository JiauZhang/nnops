#include <nnops/common.h>
#include <nnops/tensor_iterator.h>

namespace nnops {

TensorIterator::TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer) {
    tensor_meta_ = tensor_meta;
    tensor_buffer_ = buffer;
    index_ = TensorShape(this->shape().size(), 0);
    offset_ = 0;
}

TensorIterator &TensorIterator::operator++() {
    const TensorShape &shape = tensor_meta_.shape();
    const TensorStride &stride = tensor_meta_.stride();
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

TensorPartialIterator::TensorPartialIterator(
    const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer, index_t start, index_t stop
    ) : TensorIterator(tensor_meta, buffer) {
    NNOPS_CHECK(!(start < 0 || stop > tensor_meta_.ndim() || start >= stop), "invalid TensorPartialIterator parameter.");
    start_ = start;
    stop_ = stop;
}

TensorPartialIterator &TensorPartialIterator::operator++() {
    const TensorShape &shape = tensor_meta_.shape();
    const TensorStride &stride = tensor_meta_.stride();
    int ax = stop_ - 1;

    while (ax >= start_) {
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

} // namespace nnops