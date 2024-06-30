#include <nnops/tensor_meta.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

using namespace std;
namespace nb = nanobind;

void DEFINE_TENSOR_SHAPE_MODULE(nb::module_ & (m)) {
    nb::class_<TensorMeta>(m, "TensorMeta")
        .def(nb::init<>())
        .def("get_dims", &TensorMeta::get_dims)
        .def("set_dims", [](TensorMeta &self, vector<int> &dims) { self.set_dims(dims); })
        .def_prop_ro("ndim", [](TensorMeta &self) { return self.ndim(); });
}