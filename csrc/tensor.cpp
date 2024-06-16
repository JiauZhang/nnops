#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <data_type.h>
#include <tensor_shape.h>
#include <tensor.h>
#include <nanobind/stl/string.h>

using namespace std;
namespace nb = nanobind;

Tensor::Tensor(DataType &dtype, TensorShape &shape, string &device) {
    init_tensor(dtype, shape, device);
}

Tensor::Tensor(DataType &dtype, vector<int> &dims, string &device) {
    auto tshape_ = TensorShape(dims);
    init_tensor(dtype, tshape_, device);
}

Tensor::~Tensor() {
    if (data_ptr_)
        device_->free(data_ptr_);
}

void Tensor::init_tensor(DataType &dtype, TensorShape &shape, std::string &device) {
    dtype_ = dtype;
    shape_ = shape;
    device_ = Device::get_device(device);

    if (device_ == nullptr)
        throw std::runtime_error("get device failed!\n");

    nelems_ = 1;
    for (auto dim: shape_.get_dims())
        nelems_ *= dim;

    nbytes_ = nelems_ * sizeof_dtype(dtype_);

    data_ptr_ = nullptr;
    data_ptr_ = device_->malloc(nbytes_);
    if (data_ptr_ == nullptr)
        throw std::runtime_error("alloc tensor memory failed!\n");
}

void Tensor::reshape(vector<int> &dims) {
    shape_.set_dims(dims);
}

void Tensor::reshape(TensorShape &shape) {
    shape_.set_dims(shape);
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())
        .def(nb::init<DataType &, TensorShape &, string &>())
        .def(nb::init<DataType &, vector<int> &, string &>())
        .def_ro("dtype", &Tensor::dtype_)
        .def("reshape", [](Tensor &self, vector<int> &dims) { self.reshape(dims); })
        .def("reshape", [](Tensor &self, TensorShape &shape) { self.reshape(shape); })
        .def_prop_ro("ndim", [](Tensor &t) { return t.shape_.ndim(); })
        .def_prop_ro("shape", [](Tensor &t) { return t.shape_.get_dims(); });
}