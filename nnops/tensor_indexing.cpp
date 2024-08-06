#include <nnops/tensor_indexing.h>

namespace nnops {

void slice_inplace(TensorMeta &meta, Slice &slice, int axis) {
    auto &shape_ = meta.dims_;
    auto &strides_ = meta.strides_;
    auto &offset_ = meta.offset_;
    auto &nelems_ = meta.nelems_;
    auto &nbytes_ = meta.nbytes_;
    auto &start = slice.start_, &stop = slice.stop_, &step = slice.step_;

    offset_ += start.value() * strides_[axis];
    nelems_ /= shape_[axis];
    strides_[axis] *= step.value();

    if ((start.value() < stop.value()) && (step.value() > 0)) {
        shape_[axis] = (stop.value() - start.value() - 1) / step.value() + 1;
    } else if ((start.value() > stop.value()) && (step.value() < 0)) {
        shape_[axis] = (start.value() - stop.value() - 1) / (-step.value()) + 1;
    } else {
        shape_[axis] = 0;
    }

    nelems_ *= shape_[axis];
    nbytes_ = nelems_ * sizeof_dtype(meta.dtype_);
}

void index_inplace(TensorMeta &meta, int index, int axis) {
    auto &shape_ = meta.dims_;
    auto &strides_ = meta.strides_;
    auto &offset_ = meta.offset_;
    auto &nelems_ = meta.nelems_;
    auto &nbytes_ = meta.nbytes_;

    if (index < 0)
        index += shape_[axis];

    offset_ += index * strides_[axis];
    nelems_ /= shape_[axis];

    for (int i=axis; i<shape_.size()-1; i++) {
        shape_[i] = shape_[i+1];
        strides_[i] = strides_[i+1];
    }
    shape_.pop_back();
    strides_.pop_back();
    nbytes_ = nelems_ * sizeof_dtype(meta.dtype_);
}

void slice_inplace(Tensor &tensor, Slice &slice, int axis) {
    slice_inplace(tensor.tensor_meta_, slice, axis);
}

void index_inplace(Tensor &tensor, int index, int axis) {
    auto &shape = tensor.shape();

    if (index >= shape[axis] || index < -shape[axis]) {
        std::string info = "index_inplace " + std::to_string(index) + " is out of bounds for axis "
            + std::to_string(axis) + " with size " + std::to_string(shape[axis]);
        throw std::runtime_error(info);
    }

    index_inplace(tensor.tensor_meta_, index, axis);
}

} // namespace nnops
