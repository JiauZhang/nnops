#include <nnops/tensor_util.h>

namespace nnops {

Tensor tensor_from(const TensorPartialIterator &iter) {
    TensorMeta meta;
    meta.dtype_ = iter.dtype();
    meta.offset_ = iter.offset();
    meta.dims_.resize(iter.ndim() - iter.stop() + iter.start());
    meta.strides_.resize(iter.ndim() - iter.stop() + iter.start());
    int idx = 0;
    meta.nelems_ = 1;
    for (int i = 0; i < iter.start(); i++) {
        meta.dims_[idx] = iter.shape()[i];
        meta.nelems_ *= meta.dims_[idx];
        meta.strides_[idx] = iter.stride()[i];
        ++idx;
    }
    for (int i = iter.stop(); i < iter.ndim(); i++) {
        meta.dims_[idx] = iter.shape()[i];
        meta.nelems_ *= meta.dims_[idx];
        meta.strides_[idx] = iter.stride()[i];
        ++idx;
    }

    return Tensor(meta, iter.buffer());
}

} // namespace nnops