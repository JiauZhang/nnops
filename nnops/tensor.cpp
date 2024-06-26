#include <data_type.h>
#include <tensor_shape.h>
#include <tensor.h>

using namespace std;

Tensor::Tensor(DataType &dtype, TensorShape &shape, string &device) {
    init_tensor(dtype, shape, device);
}

Tensor::Tensor(DataType &dtype, vector<int> &dims, string &device) {
    auto tshape_ = TensorShape(dims);
    init_tensor(dtype, tshape_, device);
}

Tensor::Tensor(Tensor &other) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    device_ = other.device_;
    nbytes_ = other.nbytes_;
    nelems_ = other.nelems_;
    tensor_buffer_ = other.tensor_buffer_;
    tensor_buffer_->inc_ref();
}

Tensor::~Tensor() {
    if (tensor_buffer_) {
        tensor_buffer_->dec_ref();
        if (tensor_buffer_->is_zero())
            device_->free(tensor_buffer_->data_ptr_);
    }
}

void Tensor::init_tensor(DataType &dtype, TensorShape &shape, std::string &device) {
    dtype_ = dtype;
    shape_ = shape;
    device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!");

    nelems_ = 1;
    for (auto dim: shape_.get_dims())
        nelems_ *= dim;

    nbytes_ = nelems_ * sizeof_dtype(dtype_);

    void *data_ptr_ = nullptr;
    data_ptr_ = device_->malloc(nbytes_);
    if (data_ptr_ == nullptr) {
        tensor_buffer_ = nullptr;
        throw std::runtime_error("alloc tensor memory failed!");
    } else {
        tensor_buffer_ = new TensorBuffer(data_ptr_);
    }
}

void Tensor::reshape(vector<int> &dims) {
    shape_.set_dims(dims);
}

void Tensor::reshape(TensorShape &shape) {
    shape_.set_dims(shape);
}
