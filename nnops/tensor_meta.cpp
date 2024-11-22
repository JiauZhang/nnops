#include <nnops/tensor_meta.h>
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

bool TensorMeta::is_contiguous() {
    int nelems = 1;

    for (int i=this->dims_.size()-1; i>=0; i--) {
        if (nelems != this->strides_[i])
            return false;
        nelems *= this->dims_[i];
    }

    return true;
}

} // namespace nnops
