#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdio.h>
#include <Python.h>

namespace nb = nanobind;

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<Tensor &>())
        .def(nb::init<DataType &, vector<int> &, string &>())
        .def("__str__", [](Tensor &self) { return self.to_string(); })
        .def("__repr__", &Tensor::to_repr)
        .def("getitem", [](nb::handle h, std::vector<int> &dims) {
            PyObject *ob_self = h.ptr();
            Tensor *self = nb::inst_ptr<Tensor>(ob_self);
            PyTypeObject *tp_self = ob_self->ob_type;
            PyObject *ob_new = nb::detail::nb_inst_alloc(tp_self);
            nb::handle h_new(ob_new);

            ob_new->ob_refcnt = 0;
            Tensor *ptr_new = nb::inst_ptr<Tensor>(h_new);
            Tensor &&sub_tensor = (*self)[dims];
            ptr_new->tensor_meta_ = sub_tensor.tensor_meta_;
            ptr_new->tensor_buffer_ = sub_tensor.tensor_buffer_;
            ptr_new->tensor_buffer_->inc_ref();
            nb::inst_mark_ready(h_new);

            return h_new;
        })
        .def("reshape", [](Tensor &self, vector<int> &dims) { return self.reshape(dims); })
        .def_prop_ro("dtype", [](Tensor &t) { return t.dtype(); })
        .def_prop_ro("data_ptr", [](Tensor &t) { return t.data_ptr(); })
        .def_prop_ro("ref_count", [](Tensor &t) { return t.ref_count(); })
        .def_prop_ro("ndim", [](Tensor &t) { return t.ndim(); })
        .def_prop_ro("nbytes", [](Tensor &t) { return t.nbytes(); })
        .def_prop_ro("nelems", [](Tensor &t) { return t.nelems(); })
        .def_prop_ro("stride", [](Tensor &t) { return t.stride(); })
        .def_prop_ro("shape", [](Tensor &t) { return t.shape(); });
}