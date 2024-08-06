#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/tensor_indexing.h>
#include <nnops/tensor_meta.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdio.h>
#include <Python.h>

namespace nb = nanobind;

namespace pynnops {

void indexing(nnops::Tensor &tensor, nb::handle indices, int axis) {
    PyObject *ob_indices = indices.ptr();
    nnops::TensorMeta &meta = tensor.tensor_meta_;

    if (nb::isinstance<nb::tuple>(indices)) {
        // multi-dimensional indexing
        Py_ssize_t len = PyTuple_Size(ob_indices);
        if (len > tensor.ndim()) {
            std::string info = "too many indices for tensor: ";
            info += "tensor is " + std::to_string(tensor.ndim()) + "-dimensional, but "
                + std::to_string(len) + " were indexed";
            throw std::runtime_error(info);
        }

        // check ellipsis
        int idx = len, count = 0;
        for (int i=0; i<len; i++) {
            if (nb::isinstance<nb::ellipsis>(indices[i])) {
                count++;
                idx = i;
            }
        }

        if (count > 1)
            throw std::runtime_error("an index can only have a single ellipsis ('...')");

        for (int i=0; i<len; i++) {
            if (i == idx) {
                axis = tensor.ndim() + i - len + 1;
                continue;
            }

            indexing(tensor, indices[i], axis);
            if (nb::isinstance<nb::slice>(indices[i]))
                axis += 1;
        }
    } else if (nb::isinstance<nb::slice>(indices)) {
        Py_ssize_t start, stop, step;

        if (PySlice_GetIndices(ob_indices, tensor.shape()[axis], &start, &stop, &step) < 0)
            throw std::runtime_error("PySlice_GetIndices failed!");

        nnops::Slice slice(start, stop, step);
        nnops::slice_inplace(tensor, slice, axis);
    } else if (nb::isinstance<nb::int_>(indices)) {
        nnops::index_inplace(tensor, nb::cast<int>(indices), axis);
    } else if (nb::isinstance<nb::ellipsis>(indices)) {
        // do nothing
    } else {
        std::string info = "not supported indexing type: ";
        PyTypeObject *tp = ob_indices->ob_type;
        info += std::string(tp->tp_name);
        throw std::runtime_error(info);
    }
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<nnops::Tensor>(m, "Tensor")
        .def(nb::init<nnops::Tensor &>())
        .def(nb::init<nnops::DataType &, std::vector<int> &, std::string &>())
        .def("__str__", [](nnops::Tensor &self) { return self.to_string(); })
        .def("__repr__", &nnops::Tensor::to_repr)
        .def("__getitem__", [](nb::handle h, nb::handle indices) {
            PyObject *ob_self = h.ptr();
            nnops::Tensor *self = nb::inst_ptr<nnops::Tensor>(ob_self);
            PyTypeObject *tp_self = ob_self->ob_type;
            PyObject *ob_new = nb::detail::nb_inst_alloc(tp_self);
            nb::handle h_new(ob_new);
            nnops::Tensor *tensor_new = nb::inst_ptr<nnops::Tensor>(h_new);
            nnops::TensorMeta &meta = tensor_new->tensor_meta_;

            ob_new->ob_refcnt = 0;
            meta = self->tensor_meta_;
            indexing(*tensor_new, indices, 0);
            tensor_new->tensor_buffer_ = self->tensor_buffer_;
            tensor_new->tensor_buffer_->inc_ref();
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