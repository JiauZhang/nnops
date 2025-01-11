#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>
#include <stdexcept>

namespace nnops {

class TensorIterator;

Tensor::Tensor() : tensor_buffer_(nullptr) {}

Tensor::Tensor(DataType dtype, const TensorShape &dims, std::string &device) {
    Device *device_ = Device::get_device(device);
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, const TensorShape &dims, DeviceType device) {
    Device *device_ = Device::get_device(device);
    init_tensor(dtype, dims, device_);
}

Tensor::Tensor(DataType dtype, const TensorShape &dims, Device *device) {
    init_tensor(dtype, dims, device);
}

void Tensor::init_tensor(DataType &dtype, const TensorShape &dims, Device *device) {
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

Tensor::Tensor(const Tensor &other) : tensor_buffer_(nullptr){
    set_meta(other.meta());
    set_buffer(other.buffer());
}

Tensor::Tensor(const TensorMeta &meta, TensorBuffer *buf) : tensor_buffer_(nullptr){
    set_meta(meta);
    set_buffer(buf);
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

    T *data_ptr = (T *)tensor->data_ptr();
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
        DATATYPE_GEN_TEMPLATE_LOOPx1(TO_STRING_TEMPLATE_GEN)
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

void tensor_clone_impl(const Tensor *src, int src_offset, Tensor *dst, int dst_offset, int axis) {
    if (axis < src->ndim() - 1) {
        for (int i=0; i<src->shape()[axis]; i++)
            tensor_clone_impl(
                src, src_offset + i * (src->stride()[axis]),
                dst, dst_offset + i * (dst->stride()[axis]),
                axis + 1
            );
        return;
    }

    const auto &cast_op = get_cast_op(src->dtype(), dst->dtype());
    const index_t loop = src->shape()[axis];
    void *args[2] = {
        (void *)((unsigned char *)src->data_ptr() + src_offset * src->itemsize()),
        (void *)((unsigned char *)dst->data_ptr() + dst_offset * dst->itemsize()),
    };
    const index_t strides[2] = {
        src->stride()[axis] * src->itemsize(),
        dst->stride()[axis] * dst->itemsize(),
    };
    cast_op(args, strides, loop);
}

Tensor Tensor::clone() {
    Tensor tensor(this->dtype(), this->shape(), this->device());
    tensor_clone_impl(this, 0, &tensor, 0, 0);
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

bool Tensor::is_broadcastable(const TensorShape &s1, const TensorShape &s2) {
    int dims = std::min(s1.size(), s2.size()), s1_size = s1.size() - 1, s2_size = s2.size() - 1;

    for (int i=0; i<dims; i++)
        if (s1[s1_size-i] != s2[s2_size-i] && s1[s1_size-i] != 1 && s2[s2_size-i] != 1)
            return false;
    return true;
}

TensorShape Tensor::broadcast_shape(const TensorShape &s1, const TensorShape &s2) {
    int dims = std::min(s1.size(), s2.size());
    TensorShape shape;
    const TensorShape *shape_long, *shape_short;

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

Tensor Tensor::broadcast_to(const Tensor &t, const TensorShape &shape) {
    const TensorShape &ts = t.shape();
    std::string info = "Can not broadcast Tensor from shape " + TensorMeta::shape_as_string(ts)
        + " to " + TensorMeta::shape_as_string(shape);

    if (ts.size() > shape.size())
        throw std::runtime_error(info);

    int dims = std::min(ts.size(), shape.size()), ts_size = ts.size() - 1, s_size = shape.size() - 1;
    for (int i=0; i<dims; i++)
        if (shape[s_size-i] != ts[ts_size-i] && ts[ts_size-i] != 1)
            throw std::runtime_error(info);

    Tensor tb = t;
    TensorStride strides = tb.stride();
    const TensorShape &tb_shape = tb.shape();
    int offset = shape.size() - ts.size();

    for (int i=0; i<=ts_size; i++)
        if (tb_shape[i] == 1)
            strides[i] = 0;
    tb.set_shape(shape);
    strides.resize(shape.size());
    for (int i=s_size; i>=offset; i--)
        strides[i] = strides[i-offset];
    for (int i=0; i<offset; i++)
        strides[i] = 0;
    tb.set_stride(strides);

    return tb;
}

Tensor Tensor::permute(TensorShape &index) {
    if (index.size() != this->shape().size())
        throw std::runtime_error("axes size don't match!");
    int len = index.size();
    TensorShape count(len, 0);
    for (int i = 0; i < len; i++) {
        auto &idx = index[i];
        if (idx < -len || idx >= len) {
            std::string info = "axis " + std::to_string(idx) + " is out of bounds for tensor of dimension "
                + std::to_string(len);
            throw std::runtime_error(info);
        }
        if (idx < 0)
            idx += len;
        ++count[idx];
        if (count[idx] > 1)
            throw std::runtime_error("repeated axis in permute");
    }

    TensorMeta meta = this->meta();
    for (int i = 0; i < meta.shape().size(); i++) {
        meta.dims_[i] = this->meta().shape()[index[i]];
        meta.strides_[i] = this->meta().stride()[index[i]];
    }
    return Tensor(meta, this->buffer());
}

Tensor Tensor::astype(DataType dtype) {
    Tensor tensor(dtype, this->shape(), this->device());
    tensor_clone_impl(this, 0, &tensor, 0, 0);
    return tensor;
}

Tensor Tensor::to(DeviceType device) {
    if (device == this->device()->get_device_type())
        return *this;
    Device *dev = Device::get_device(device);
    if (dev == nullptr)
        throw std::runtime_error("device is invalid!");
    Tensor dst(this->dtype(), this->shape(), dev), src = this->contiguous();
    if (src.device()->get_device_type() == DeviceType::CPU) {
        dev->copy_from_cpu(src.data_ptr(), dst.data_ptr(), src.nbytes());
    } else {
        dev = src.device();
        dev->copy_to_cpu(src.data_ptr(), dst.data_ptr(), src.nbytes());
    }
    return dst;
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
    if (tensor_buffer_ != buf) {
        if (tensor_buffer_)
            tensor_buffer_->dec_ref();
        tensor_buffer_ = buf;
        tensor_buffer_->inc_ref();
    }
}

TensorShape Tensor::unravel_index(index_t idx, const TensorShape &shape) {
    TensorShape indices(shape.size()), strides_contig(shape.size());
    strides_contig[strides_contig.size() - 1] = 1;
    for (int i = strides_contig.size()-2; i >= 0; i--)
        strides_contig[i] = strides_contig[i + 1] * shape[i + 1];
    index_t nelems = strides_contig[0] * shape[0];

    if (idx > nelems) {
        std::string info = "index " + std::to_string(idx) + "is out of bounds for TensorShape with size"
            + std::to_string(nelems);
        throw std::runtime_error(info);
    }

    for (int i = 0; i < indices.size(); i++) {
        indices[i] = idx / strides_contig[i];
        idx -= indices[i] * strides_contig[i];
    }
    return indices;
}

index_t Tensor::ravel_index(const TensorShape &indices, const TensorShape &shape) {
    if (indices.size() != shape.size()) {
        std::string info = "parameter indices must be a sequence of length " + std::to_string(shape.size());
        throw std::runtime_error(info);
    }
    TensorShape strides_contig(shape.size());
    index_t idx = 0;
    strides_contig[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--)
        strides_contig[i] = strides_contig[i + 1] * shape[i + 1];
    for (int i = 0; i < shape.size(); i++) {
        if (indices[i] >= shape[i]) {
            std::string info = "indices[" + std::to_string(i) + "]: " + std::to_string(indices[i])
                + "is out of bounds for shape[" + std::to_string(i) + "]: " + std::to_string(shape[i]);
            throw std::runtime_error(info);
        }
    }
    for (int i = 0; i < shape.size(); i++)
        idx += indices[i] * strides_contig[i];
    return idx;
}

} // namespace nnops
