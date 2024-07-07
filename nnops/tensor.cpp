#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

using namespace std;

Tensor::Tensor(DataType &dtype, vector<int> &dims, string &device) {
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
    auto &device_ = tensor_meta_.device_;
    device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");

    alloc_buffer(tensor_meta_);
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
        if (tensor_buffer_->is_zero()) {
            tensor_meta_.device_->free(tensor_buffer_->data_ptr_);
            free(tensor_buffer_);
            tensor_buffer_ = nullptr;
        }
    }
}

void Tensor::alloc_buffer(TensorMeta &meta) {
    void *data_ptr_ = nullptr;
    auto &device_ = meta.device_;
    data_ptr_ = device_->malloc(meta.nbytes_);
    if (data_ptr_ == nullptr) {
        tensor_buffer_ = nullptr;
        throw std::runtime_error("alloc tensor memory failed!");
    } else {
        tensor_buffer_ = new TensorBuffer(data_ptr_);
    }
}

void Tensor::reshape(vector<int> &dims) {
    tensor_meta_.reshape(dims);
}

template<typename T>
void to_string_impl(Tensor *tensor, std::string *ret, int dim) {
    std::string prefix;
    if (ret->size() && ret->back() == '\n')
        for (int i=0; i<dim; i++)
            prefix += ' ';
    prefix += '[';

    if (dim < tensor->ndim() - 1) {
        *ret += prefix;
        for (int i=0; i<tensor->shape()[dim]; i++)
            to_string_impl<T>(tensor, ret, dim+1);
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

    T *data_ptr = (T *)tensor->tensor_buffer_->data_ptr_;
    *ret += prefix;

    for (int i=0; i<tensor->shape()[dim]; i++)
        *ret += std::to_string(data_ptr[i]) + ", ";

    auto len = ret->size();
    (*ret)[len-2] = ']';
    (*ret)[len-1] = ',';
    *ret += '\n';
}

#define TO_STRING_TEMPLATE_GEN(dtype, type)      \
    case dtype: {                                \
        to_string_impl<type>(this, &ret, 0);     \
        break;                                   \
    }

std::string Tensor::to_string() {
    std::string ret;
    switch (tensor_meta_.dtype_) {
        DATATYPE_GEN_TEMPLATE(TO_STRING_TEMPLATE_GEN)
        default:
            throw std::runtime_error("invalid type");
    }

    return ret;
}

Tensor Tensor::operator[](std::vector<int> &dims) {
    return Tensor(*this, this->shape());
}
