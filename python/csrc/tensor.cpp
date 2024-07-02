#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdio.h>

namespace nb = nanobind;

void at(Tensor &t, vector<int> &dims) {
    switch (t.meta_.dtype_) {
        case DataType::TYPE_FLOAT32: {
            auto value = t.at<DataType::TYPE_FLOAT32>(dims);
            printf("%s\n", std::to_string(value).c_str());
            break;
        } case DataType::TYPE_INT32: {
            auto value = t.at<DataType::TYPE_INT32>(dims);
            printf("%s\n", std::to_string(value).c_str());
            break;
        } case DataType::TYPE_INT16: {
            auto value = t.at<DataType::TYPE_INT16>(dims);
            printf("%s\n", std::to_string(value).c_str());
            break;
        } default: {
            break;
        }
    }
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<Tensor &>())
        .def(nb::init<DataType &, vector<int> &, string &>())
        .def_prop_ro("dtype", [](Tensor &t) { return t.meta_.dtype_; })
        .def("reshape", [](Tensor &self, vector<int> &dims) { self.reshape(dims); })
        .def("at", [](Tensor &self, vector<int> &dims) { at(self, dims); })
        .def_prop_ro("data_ptr", [](Tensor &t) { return t.tensor_buffer_->data_ptr_; })
        .def_prop_ro("ref_count", [](Tensor &t) { return t.tensor_buffer_->count(); })
        .def_prop_ro("ndim", [](Tensor &t) { return t.meta_.ndim(); })
        .def_prop_ro("nbytes", [](Tensor &t) { return t.meta_.nbytes_; })
        .def_prop_ro("nelems", [](Tensor &t) { return t.meta_.nelems_; })
        .def_prop_ro("shape", [](Tensor &t) { return t.meta_.get_dims(); });
}