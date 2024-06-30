#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

using namespace std;

Tensor::Tensor(DataType &dtype, vector<int> &dims, string &device) {
    meta_.dims_ = dims;
    meta_.dtype_ = dtype;

    auto &nelems_ = meta_.nelems_;
    nelems_ = 1;
    for (auto dim: meta_.get_dims())
        nelems_ *= dim;

    meta_.nbytes_ = nelems_ * sizeof_dtype(dtype);
    auto &device_ = meta_.device_;
    device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");

    alloc_buffer(meta_);
}

Tensor::Tensor(Tensor &other) {
    meta_ = other.meta_;
    tensor_buffer_ = other.tensor_buffer_;
    tensor_buffer_->inc_ref();
}

Tensor::~Tensor() {
    if (tensor_buffer_) {
        tensor_buffer_->dec_ref();
        if (tensor_buffer_->is_zero())
            meta_.device_->free(tensor_buffer_->data_ptr_);
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
    meta_.reshape(dims);
}
