#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "data_type.h"
#include "tensor_shape.h"
#include "tensor.h"

using namespace std;
namespace nb = nanobind;

void Tensor::reshape(vector<int> &dims) {
    shape_.set_dims(dims);
}

void Tensor::reshape(TensorShape &shape) {
    shape_.set_dims(shape);
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())
        .def("reshape", [](Tensor &self, vector<int> &dims) { self.reshape(dims); })
        .def("reshape", [](Tensor &self, TensorShape &shape) { self.reshape(shape); })
        .def_prop_ro("shape", [](Tensor &t) { return t.shape_.get_dims(); });
}