#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>
#include <nnops/tensor_indexing.h>

namespace nnops {

Tensor::Tensor(): tensor_buffer_(nullptr) {}

Tensor::Tensor(DataType dtype, std::vector<int> &dims, std::string &device) {
    Device *device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, std::vector<int> &dims, DeviceType device) {
    Device *device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, std::vector<int> &dims, Device *device) {
    Device *device_ = device;

    if (device_ == nullptr)
        throw std::runtime_error("device is invalid!");
    init_tensor(dtype, dims, device_);
}

void Tensor::init_tensor(DataType &dtype, std::vector<int> &dims, Device *device) {
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
    tensor_buffer_ = new TensorBuffer(device, tensor_meta_.nbytes_);
}

Tensor::Tensor(const Tensor &other) {
    tensor_meta_ = other.tensor_meta_;
    tensor_buffer_ = other.tensor_buffer_;
    tensor_buffer_->inc_ref();
}

Tensor::~Tensor() {
    if (tensor_buffer_) {
        tensor_buffer_->dec_ref();
        tensor_buffer_ = nullptr;
    }
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

    T *data_ptr = (T *)tensor->data_ptr() + offset;
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

Tensor &Tensor::operator=(Tensor &other) {
    if (this != &other) {
        tensor_meta_ = other.tensor_meta_;
        tensor_buffer_ = other.tensor_buffer_;
        tensor_buffer_->inc_ref();
    }
    return *this;
}

void tensor_clone_impl(Tensor *src, int src_offset, Tensor *dst, int dst_offset, int axis) {
    if (axis < src->ndim() - 1) {
        for (int i=0; i<src->shape()[axis]; i++)
            tensor_clone_impl(
                src, src_offset + i * (src->stride()[axis]),
                dst, dst_offset + i * (dst->stride()[axis]),
                axis + 1
            );
    }

    int itemsize = sizeof_dtype(src->dtype());
    unsigned char *src_ptr = (unsigned char *)src->data_ptr() + src_offset * itemsize;
    unsigned char *dst_ptr = (unsigned char *)dst->data_ptr() + dst_offset * itemsize;
    auto &src_stride = src->stride();
    auto &dst_stride = dst->stride();
    for (int i=0; i<src->shape()[axis]; i++) {
        for (int j=0; j<itemsize; j++)
            dst_ptr[j] = src_ptr[j];
        src_ptr += src_stride[axis] * itemsize;
        dst_ptr += dst_stride[axis] * itemsize;
    }
}

Tensor Tensor::clone() {
    Tensor tensor(this->dtype(), this->shape(), this->device());
    tensor_clone_impl(this, this->offset(), &tensor, tensor.offset(), 0);
    return tensor;
}

Tensor Tensor::contiguous() {
    if (this->is_contiguous()) {
        return *this;
    } else {
        return this->clone();
    }
}

Tensor Tensor::reshape(std::vector<int> &dims) {
    Tensor tensor = this->contiguous();
    tensor.reshape_inplace(dims);
    return tensor;
}

bool Tensor::is_broadcastable(std::vector<int> &s1, std::vector<int> &s2) {
    int dims = std::min(s1.size(), s2.size());

    for (int i=0; i<dims; i++)
        if (s1[i] != s2[i] && s1[i] != 1 && s2[i] != 1)
            return false;
    return true;
}

std::vector<int> Tensor::broadcast_shape(std::vector<int> &s1, std::vector<int> &s2) {
    int dims = std::min(s1.size(), s2.size());
    std::vector<int> shape, *shape_long, *shape_short;

    if (dims == s1.size()) {
        shape_long = &s2;
        shape_short = &s1;
    } else {
        shape_long = &s1;
        shape_short = &s2;
    }
    dims = shape_long->size();
    shape.resize(dims);
    for (int i=0; i<dims; i++)
        shape[i] = shape_long->data()[i];
    dims = shape_long->size() - 1;
    for (int i=shape_short->size()-1; i>=0; i--) {
        if (shape_short->data()[i] != shape[dims])
            shape[dims] *= shape_short->data()[i];
        dims--;
    }

    return shape;
}

bool Tensor::is_broadcast() {
    int dims = this->ndim();
    auto &strides = this->stride();
    for (int i=0; i<dims; i++)
        if (strides[i] == 0)
            return true;
    return false;
}

Tensor Tensor::broadcast_to(Tensor &t, std::vector<int> &shape) {
    Tensor tb = t;

    return tb;
}

} // namespace nnops
