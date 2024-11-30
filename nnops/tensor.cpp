#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

namespace nnops {

Tensor::Tensor(): tensor_buffer_(nullptr) {}

Tensor::Tensor(DataType dtype, TensorShape &dims, std::string &device) {
    Device *device_ = Device::get_device(device);
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, TensorShape &dims, DeviceType device) {
    Device *device_ = Device::get_device(device);
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, TensorShape &dims, Device *device) {
    init_tensor(dtype, dims, device);
}

void Tensor::init_tensor(DataType &dtype, TensorShape &dims, Device *device) {
    if (device == nullptr)
        throw std::runtime_error("device is invalid!");

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
    set_meta(other.meta());
    set_buffer(other.buffer());
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
        set_meta(other.meta());
        set_buffer(other.buffer());
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
        return;
    }

    auto cast_op = get_cast_op(src->dtype(), dst->dtype());
    int src_itemsize = sizeof_dtype(src->dtype());
    int dst_itemsize = sizeof_dtype(dst->dtype());
    unsigned char *src_ptr = (unsigned char *)src->data_ptr() + src_offset * src_itemsize;
    unsigned char *dst_ptr = (unsigned char *)dst->data_ptr() + dst_offset * dst_itemsize;
    auto &src_stride = src->stride();
    auto &dst_stride = dst->stride();
    for (int i=0; i<src->shape()[axis]; i++) {
        cast_op(src_ptr, dst_ptr);
        src_ptr += src_stride[axis] * src_itemsize;
        dst_ptr += dst_stride[axis] * dst_itemsize;
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

Tensor Tensor::reshape(TensorShape &dims) {
    Tensor tensor = this->contiguous();
    tensor.reshape_inplace(dims);
    return tensor;
}

bool Tensor::is_broadcastable(TensorShape &s1, TensorShape &s2) {
    int dims = std::min(s1.size(), s2.size()), s1_size = s1.size() - 1, s2_size = s2.size() - 1;

    for (int i=0; i<dims; i++)
        if (s1[s1_size-i] != s2[s2_size-i] && s1[s1_size-i] != 1 && s2[s2_size-i] != 1)
            return false;
    return true;
}

TensorShape Tensor::broadcast_shape(TensorShape &s1, TensorShape &s2) {
    int dims = std::min(s1.size(), s2.size());
    TensorShape shape, *shape_long, *shape_short;

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

Tensor Tensor::broadcast_to(Tensor &t, TensorShape &shape) {
    TensorShape &ts = t.shape();
    std::string info = "Can not broadcast Tensor from shape " + TensorMeta::shape_as_string(ts)
        + " to " + TensorMeta::shape_as_string(shape);

    if (ts.size() > shape.size())
        throw std::runtime_error(info);

    int dims = std::min(ts.size(), shape.size()), ts_size = ts.size() - 1, s_size = shape.size() - 1;
    for (int i=0; i<dims; i++)
        if (shape[s_size-i] != ts[ts_size-i] && ts[ts_size-i] != 1)
            throw std::runtime_error(info);

    Tensor tb = t;
    TensorStride &strides = tb.stride(), &tb_shape = tb.shape();
    int offset = shape.size() - ts.size();

    for (int i=0; i<=ts_size; i++)
        if (tb_shape[i] == 1)
            strides[i] = 0;
    tb_shape = shape;
    strides.resize(shape.size());
    for (int i=s_size; i>=offset; i--)
        strides[i] = strides[i-offset];
    for (int i=0; i<offset; i++)
        strides[i] = 0;

    return tb;
}

Tensor Tensor::astype(DataType dtype) {
    Tensor tensor(dtype, this->shape(), this->device());
    tensor_clone_impl(this, this->offset(), &tensor, tensor.offset(), 0);
    return tensor;
}

const TensorMeta &Tensor::meta() const {
    return tensor_meta_;
}

void Tensor::set_meta(const TensorMeta &meta) {
    tensor_meta_ = meta;
}

TensorBuffer *Tensor::buffer() const {
    return tensor_buffer_;
}

void Tensor::set_buffer(TensorBuffer *buf) {
    tensor_buffer_ = buf;
    tensor_buffer_->inc_ref();
}

} // namespace nnops
