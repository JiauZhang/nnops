#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>
#include <nnops/tensor_indexing.h>

namespace nnops {

Tensor::Tensor(): tensor_buffer_(nullptr) {}

Tensor::Tensor(DataType &dtype, std::vector<int> &dims, std::string &device) {
    tensor_buffer_ = nullptr;
    tensor_meta_.offset_ = 0;
    tensor_meta_.dims_ = dims;
    tensor_meta_.dtype_ = dtype;

    auto &nelems_ = tensor_meta_.nelems_;
    auto &strides_ = tensor_meta_.strides_;
    auto &shape = this->shape();

    nelems_ = 1;
    strides_.resize(shape.size());

    for (int i=shape.size()-1; i>=0; i--) {
        strides_[i] = nelems_;
        nelems_ *= shape[i];
    }

    if (shape.size() == 0 || nelems_ <= 0)
        throw std::runtime_error("invalid shape info!");

    tensor_meta_.nbytes_ = nelems_ * sizeof_dtype(dtype);
    Device *device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");

    void *data_ptr_ = nullptr;
    data_ptr_ = device_->malloc(tensor_meta_.nbytes_);
    if (data_ptr_ == nullptr)
        throw std::runtime_error("alloc tensor memory failed!");
    else
        tensor_buffer_ = new TensorBuffer(data_ptr_, device_);
}

Tensor::Tensor(const Tensor &other) {
    tensor_meta_ = other.tensor_meta_;
    tensor_buffer_ = other.tensor_buffer_;
    tensor_buffer_->inc_ref();
}

Tensor::Tensor(const Tensor &other, const std::vector<int> &dims) {
    tensor_meta_ = other.tensor_meta_;
    tensor_meta_.dims_ = dims;
    tensor_buffer_ = other.tensor_buffer_;
    tensor_buffer_->inc_ref();
}

Tensor::~Tensor() {
    if (tensor_buffer_) {
        tensor_buffer_->dec_ref();
        tensor_buffer_ = nullptr;
    }
}

Tensor Tensor::reshape(std::vector<int> &dims) {
    int nelems = 1;
    for (auto dim: dims)
        nelems *= dim;

    if (nelems != this->nelems()) {
        std::string info = "Can't reshape tensor from shape "
            + shape_as_string(this->shape()) + " to " + shape_as_string(dims);
        throw std::runtime_error(info);
    }

    Tensor _tensor;
    auto &_dims = _tensor.tensor_meta_.dims_;
    auto &_strides = _tensor.tensor_meta_.strides_;

    _tensor.tensor_meta_ = tensor_meta_;
    _dims = dims;
    _strides.resize(_dims.size());
    nelems = 1;

    for (int i=_dims.size()-1; i>=0; i--) {
        _strides[i] = nelems;
        nelems *= _dims[i];
    }
    _tensor.tensor_buffer_ = tensor_buffer_;
    _tensor.tensor_buffer_->inc_ref();

    return _tensor;
}

template<typename T>
void to_string_impl(Tensor *tensor, std::string *prefix, std::string *ret, int dim, int offset) {
    std::string cur_prefix;

    if (ret->size() && ret->back() == '\n') {
        if (prefix)
            cur_prefix += *prefix;
        for (int i=0; i<dim; i++)
            cur_prefix += ' ';
    }
    cur_prefix += '[';

    if (dim < tensor->ndim() - 1) {
        *ret += cur_prefix;
        for (int i=0; i<tensor->shape()[dim]; i++)
            to_string_impl<T>(tensor, prefix, ret, dim+1, offset+i*(tensor->stride()[dim]));
        auto len = ret->size();
        (*ret)[len-2] = ']';

        if (dim == 0) {
            ret->resize(len-1);
        } else {
            (*ret)[len-1] = ',';
            *ret += '\n';
        }
        return;
    }

    T *data_ptr = (T *)tensor->tensor_buffer_->data_ptr_ + offset;
    auto &stride = tensor->stride();

    if (tensor->ndim() == 0) {
        *ret += std::to_string(*data_ptr);
        return;
    }

    *ret += cur_prefix;
    for (int i=0; i<tensor->shape()[dim]; i++) {
        *ret += std::to_string(*data_ptr) + ", ";
        data_ptr += stride[dim];
    }

    auto len = ret->size();
    (*ret)[len-2] = ']';

    if (dim == 0) {
        ret->resize(len-1);
    } else {
        (*ret)[len-1] = ',';
        *ret += '\n';
    }
}

#define TO_STRING_TEMPLATE_GEN(dtype, type)                  \
    case dtype: {                                            \
        int offset = this->tensor_meta_.offset_;             \
        to_string_impl<type>(this, prefix, ret, 0, offset);  \
        break;                                               \
    }

void Tensor::to_string(std::string *prefix, std::string *ret) {
    if (tensor_meta_.nelems_ == 0)
        return;

    switch (tensor_meta_.dtype_) {
        DATATYPE_GEN_TEMPLATE(TO_STRING_TEMPLATE_GEN)
        default:
            throw std::runtime_error("invalid type");
    }
}

std::string Tensor::to_string() {
    std::string ret;
    this->to_string(nullptr, &ret);
    return ret;
}

std::string Tensor::to_repr() {
    std::string ret = "Tensor(";
    std::string prefix(ret.size(), ' ');

    this->to_string(&prefix, &ret);
    ret += ')';

    return ret;
}

std::string Tensor::shape_as_string(const std::vector<int> &dims) {
    std::string shape_str;

    shape_str += '[';
    for (auto dim: dims)
        shape_str += std::to_string(dim) + ", ";
    auto len = shape_str.size();
    shape_str.resize(len-1);
    shape_str[len-2] = ']';

    return shape_str;
}

} // namespace nnops
