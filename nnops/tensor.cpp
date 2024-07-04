#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

using namespace std;

Tensor::Tensor(DataType &dtype, vector<int> &dims, string &device) {
    tensor_meta_.dims_ = dims;
    tensor_meta_.dtype_ = dtype;

    auto &nelems_ = tensor_meta_.nelems_;
    nelems_ = 1;
    for (auto dim: this->shape())
        nelems_ *= dim;

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
std::string to_string_impl(Tensor *tensor) {
    T *data_ptr = (T *)tensor->tensor_buffer_->data_ptr_;
    std::string ret = "[";

    for (int i=0; i<tensor->tensor_meta_.nelems_; i++)
        ret += std::to_string(data_ptr[i]) + ", ";

    auto len = ret.size();
    ret.resize(len - 1);
    ret[len - 2] = ']';

    return ret;
}

#define TO_STRING_TEMPLATE_GEN(dtype, type)      \
    case dtype: {                                \
        ret = to_string_impl<type>(this);        \
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
