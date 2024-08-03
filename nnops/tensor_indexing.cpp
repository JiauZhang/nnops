#include <nnops/tensor_indexing.h>

namespace nnops {

void slice_inplace(TensorMeta &meta, Slice &slice) {

}

void index_inplace(TensorMeta &meta, int dim) {
    auto &shape_ = meta.dims_;
    auto &strides_ = meta.strides_;
    auto &offset_ = meta.offset_;
    auto &nelems_ = meta.nelems_;
    auto &nbytes_ = meta.nbytes_;

    if (dim < 0)
        dim += shape_[0];

    offset_ += dim * strides_[0];
    nelems_ = 1;

    for (int i=0; i<shape_.size()-1; i++) {
        shape_[i] = shape_[i+1];
        strides_[i] = strides_[i+1];
        nelems_ *= shape_[i];
    }
    shape_.pop_back();
    strides_.pop_back();
    nbytes_ = nelems_ * sizeof_dtype(meta.dtype_);
}

} // namespace nnops
