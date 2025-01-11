#include <nnops/common.h>
#include <nnops/tensor_iterator.h>
#include <stdexcept>

namespace nnops {

TensorIterator::TensorIterator(const Tensor &tensor) : tensor_(&tensor) {
    index_ = TensorShape(tensor_->shape().size(), 0);
    offset_ = 0;
}

TensorIterator &TensorIterator::operator++() {
    const TensorShape &shape = tensor_->shape();
    const TensorStride &stride = tensor_->stride();
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

TensorPartialIterator::TensorPartialIterator(const Tensor &tensor, index_t start, index_t stop): TensorIterator(tensor) {
    if (start < 0 || stop > tensor.ndim() || start >= stop) {
        const std::string info = "invalid TensorPartialIterator parameter.";
        throw std::runtime_error(info);
    }

    start_ = start;
    stop_ = stop;
}

TensorPartialIterator &TensorPartialIterator::operator++() {
    const TensorShape &shape = tensor_->shape();
    const TensorStride &stride = tensor_->stride();
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

Tensor TensorPartialIterator::tensor() {
    TensorMeta meta;
    meta.dtype_ = tensor_->dtype();
    meta.offset_ = this->offset_;
    meta.dims_.resize(tensor_->ndim() - stop_ + start_);
    meta.strides_.resize(tensor_->ndim() - stop_ + start_);
    int idx = 0;
    meta.nelems_ = 1;
    for (int i = 0; i < start_; i++) {
        meta.dims_[idx] = tensor_->shape()[i];
        meta.nelems_ *= meta.dims_[idx];
        meta.strides_[idx] = tensor_->stride()[i];
        ++idx;
    }
    for (int i = stop_; i < tensor_->ndim(); i++) {
        meta.dims_[idx] = tensor_->shape()[i];
        meta.nelems_ *= meta.dims_[idx];
        meta.strides_[idx] = tensor_->stride()[i];
        ++idx;
    }
    meta.nbytes_ = meta.nelems_ * tensor_->itemsize();
    return Tensor(meta, tensor_->buffer());
}

} // namespace nnops