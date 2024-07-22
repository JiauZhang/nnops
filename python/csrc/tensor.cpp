#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdio.h>
#include <Python.h>

namespace nb = nanobind;

namespace pynnops {

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<nnops::Tensor>(m, "Tensor")
        .def(nb::init<nnops::Tensor &>())
        .def(nb::init<nnops::DataType &, std::vector<int> &, std::string &>())
        .def("__str__", [](nnops::Tensor &self) { return self.to_string(); })
        .def("__repr__", &nnops::Tensor::to_repr)
        .def("getitem", [](nb::handle h, std::vector<int> &dims) {
            PyObject *ob_self = h.ptr();
            nnops::Tensor *self = nb::inst_ptr<nnops::Tensor>(ob_self);
            PyTypeObject *tp_self = ob_self->ob_type;
            PyObject *ob_new = nb::detail::nb_inst_alloc(tp_self);
            nb::handle h_new(ob_new);

            ob_new->ob_refcnt = 0;
            nnops::Tensor *ptr_new = nb::inst_ptr<nnops::Tensor>(h_new);
            nnops::Tensor &&sub_tensor = (*self)[dims];
            ptr_new->tensor_meta_ = sub_tensor.tensor_meta_;
            ptr_new->tensor_buffer_ = sub_tensor.tensor_buffer_;
            ptr_new->tensor_buffer_->inc_ref();
            nb::inst_mark_ready(h_new);

            return h_new;
        })
        .def("reshape", [](nnops::Tensor &self, std::vector<int> &dims) { return self.reshape(dims); })
        .def_prop_ro("dtype", [](nnops::Tensor &t) { return t.dtype(); })
        .def_prop_ro("data_ptr", [](nnops::Tensor &t) { return t.data_ptr(); })
        .def_prop_ro("ref_count", [](nnops::Tensor &t) { return t.ref_count(); })
        .def_prop_ro("ndim", [](nnops::Tensor &t) { return t.ndim(); })
        .def_prop_ro("nbytes", [](nnops::Tensor &t) { return t.nbytes(); })
        .def_prop_ro("nelems", [](nnops::Tensor &t) { return t.nelems(); })
        .def_prop_ro("stride", [](nnops::Tensor &t) { return t.stride(); })
        .def_prop_ro("shape", [](nnops::Tensor &t) { return t.shape(); });
}

} // namespace pynnops