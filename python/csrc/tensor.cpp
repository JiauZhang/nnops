#include <tensor.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<Tensor &>())
        .def(nb::init<DataType &, TensorShape &, string &>())
        .def(nb::init<DataType &, vector<int> &, string &>())
        .def_ro("dtype", &Tensor::dtype_)
        .def("reshape", [](Tensor &self, vector<int> &dims) { self.reshape(dims); })
        .def("reshape", [](Tensor &self, TensorShape &shape) { self.reshape(shape); })
        .def_prop_ro("data_ptr", [](Tensor &t) { return t.tensor_buffer_->data_ptr_; })
        .def_prop_ro("ref_count", [](Tensor &t) { return t.tensor_buffer_->count(); })
        .def_prop_ro("ndim", [](Tensor &t) { return t.shape_.ndim(); })
        .def_prop_ro("nbytes", [](Tensor &t) { return t.nbytes_; })
        .def_prop_ro("nelems", [](Tensor &t) { return t.nelems_; })
        .def_prop_ro("shape", [](Tensor &t) { return t.shape_.get_dims(); });
}