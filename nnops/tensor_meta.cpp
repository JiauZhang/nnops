#include <nnops/tensor_meta.h>
#include <nnops/common.h>
#include <string>
#include <stdexcept>

namespace nnops {

std::string TensorMeta::shape_as_string(const TensorShape &dims) {
    std::string shape_str;

    shape_str += '[';
    for (auto dim: dims)
        shape_str += std::to_string(dim) + ", ";
    auto len = shape_str.size();
    shape_str.resize(len-1);
    shape_str[len-2] = ']';

    return shape_str;
}

void TensorMeta::reshape_inplace(TensorShape &indices) {
    int value, idx = indices.size(), count = 0, nelems = 1;

    for (int i=0; i<indices.size(); i++) {
        value = indices[i];
        if (value < 0) {
            idx = i;
            count++;
        } else {
            nelems *= value;
        }
    }

    if (count > 1) {
        throw std::runtime_error("can only specify one unknown dimension!");
    } else if (count == 1) {
        if (nelems == 0 || this->nelems_ % nelems) {
reshape_error:
            std::string info = "cannot reshape tensor of shape " 
                + nnops::TensorMeta::shape_as_string(this->dims_)
                + " into shape " + nnops::TensorMeta::shape_as_string(indices);
            throw std::runtime_error(info);
        }

        indices[idx] = this->nelems_ / nelems;
        nelems *= indices[idx];
    }

    if (nelems != this->nelems_)
        goto reshape_error;

    nelems = 1;
    dims_ = indices;
    strides_.resize(dims_.size());
    for (int i=dims_.size()-1; i>=0; i--) {
        strides_[i] = nelems;
        nelems *= dims_[i];
    }
}

void TensorMeta::index_inplace(int index, int axis) {
    TensorShape &shape = dims_;

    NNOPS_CHECK(
        !(index >= shape[axis] || index < -shape[axis]),
        "index %d is out of bounds for axis %d with size %d",
        index, axis, shape[axis]
    );

    if (index < 0)
        index += shape[axis];

    offset_ += index * strides_[axis];
    nelems_ /= shape[axis];

    for (int i=axis; i<shape.size()-1; i++) {
        shape[i] = shape[i+1];
        strides_[i] = strides_[i+1];
    }
    shape.pop_back();
    strides_.pop_back();
}

void TensorMeta::slice_inplace(Slice &slice, int axis) {
    auto &shape_ = dims_;
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
}

bool TensorMeta::is_contiguous() const {
    int nelems = 1;

    for (int i=this->dims_.size()-1; i>=0; i--) {
        if (nelems != this->strides_[i])
            return false;
        nelems *= this->dims_[i];
    }

    return true;
}

TensorMeta TensorMeta::permute(const TensorShape &index) const {
    NNOPS_CHECK(index.size() == this->shape().size(), "axes size don't match!");
    int len = index.size();
    TensorShape count(len, 0), indices = index;
    for (int i = 0; i < len; i++) {
        auto &idx = indices[i];
        NNOPS_CHECK(!(idx < -len || idx >= len), "axis %d is out of bounds for tensor of dimension %d", idx, len);
        if (idx < 0)
            idx += len;
        ++count[idx];
        NNOPS_CHECK(count[idx] <= 1, "repeated axis in permute");
    }

    TensorMeta meta = *this;
    for (int i = 0; i < this->shape().size(); i++) {
        meta.dims_[i] = this->shape()[indices[i]];
        meta.strides_[i] = this->stride()[indices[i]];
    }
    return meta;
}

TensorMeta TensorMeta::transpose(index_t dim0, index_t dim1) const {
    const int len = 2, size = this->ndim();
    index_t dims[len] = {dim0, dim1};
    for (int i = 0; i < len; i++) {
        auto &dim = dims[i];
        NNOPS_CHECK(!(dim < -size || dim >= size), "axis %d is out of bounds for tensor of dimension %d", dim, size);
        if (dim < 0)
            dim += size;
    }

    NNOPS_CHECK(dims[0] != dims[1], "repeated axis in transpose");

    TensorMeta meta = *this;
    for (int i = 0; i < len; i++) {
        auto idx = i ^ 1;
        meta.dims_[dims[i]] = this->shape()[dims[idx]];
        meta.strides_[dims[i]] = this->stride()[dims[idx]];
    }
    return meta;
}

} // namespace nnops
