#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "tensor_shape.h"

using namespace std;
namespace nb = nanobind;

void DEFINE_TENSOR_SHAPE_MODULE(nb::module_ & (m)) {
    nb::class_<TensorShape>(m, "TensorShape")
        .def(nb::init<>())
        .def(nb::init<TensorShape &>())
        .def(nb::init<vector<int> &>())
        .def("get_dims", &TensorShape::get_dims)
        .def("set_dims", [](TensorShape &self, vector<int> &dims) { self.set_dims(dims); })
        .def("set_dims", [](TensorShape &self, TensorShape &shape) { self.set_dims(shape); })
        .def_prop_ro("ndim", [](TensorShape &shape) { return shape.ndim(); });
}