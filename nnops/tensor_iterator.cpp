#include <nnops/common.h>
#include <nnops/tensor_iterator.h>

namespace nnops {

TensorIterator::TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer) {
    tensor_meta_ = tensor_meta;
    tensor_buffer_ = buffer;
    index_ = TensorShape(this->shape().size(), 0);
    offset_ = 0;
}

TensorIterator::TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer, index_t start, index_t stop) {
    auto &dims = tensor_meta_.dims_;
    auto &strides = tensor_meta_.strides_;
    auto &nelems = tensor_meta_.nelems_;
    auto &offset = tensor_meta_.offset_;
    dims.resize(stop - start);
    strides.resize(stop - start);
    nelems = 1;
    for (int i = start; i < stop; i++) {
        nelems *= tensor_meta.dims_[i];
        dims[i - start] = tensor_meta.dims_[i];
        strides[i - start] = tensor_meta.strides_[i];
    }
    tensor_meta_.dtype_ = tensor_meta.dtype_;
    offset = tensor_meta.offset();
    tensor_buffer_ = buffer;
    index_ = TensorShape(this->shape().size(), 0);
    offset_ = 0;
}

TensorIterator::TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer, index_t start, index_t stop, index_t offset) {
    new (this) TensorIterator(tensor_meta, buffer, start, stop);
    this->tensor_meta_.offset_ += offset;
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

} // namespace nnops