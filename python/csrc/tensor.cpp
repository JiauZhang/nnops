#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/tensor_indexing.h>
#include <nnops/tensor_meta.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <stdio.h>
#include <Python.h>

namespace nb = nanobind;

namespace pynnops {

static PyTypeObject *pytensor_type = nullptr;

nb::handle pytensor_new() {
    if (pytensor_type == nullptr)
        throw std::runtime_error("PyTensorType is not initialized!");

    PyObject *ob_new = nb::detail::nb_inst_alloc(pytensor_type);
    ob_new->ob_refcnt = 0;
    nb::handle h(ob_new);
    nb::inst_mark_ready(h);
    return h;
}

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

nb::handle from_numpy(nb::ndarray<> array) {
    std::vector<int> shape;
    nnops::DataType dtype;
    auto array_dtype = array.dtype();
    nb::handle pytensor;

    for (int i=0; i<array.ndim(); i++)
        shape.push_back(array.shape(i));

    if (array_dtype == nb::dtype<char>()) {
        dtype = nnops::DataType::TYPE_INT8;
        char *array_data = (char *)array.data();
        pytensor = pytensor_new();
        nnops::Tensor *tensor = nb::inst_ptr<nnops::Tensor>(pytensor);
        new (tensor) nnops::Tensor(dtype, shape, nnops::DeviceType::CPU);
        char *tensor_data = (char *)(tensor->data_ptr());

        for (int i=0; i<tensor->nelems(); i++)
            tensor_data[i] = array_data[i];

        return pytensor;
    } else {
        throw std::runtime_error("invalid from_numpy dtype!");
    }
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    nb::class_<nnops::Tensor>(m, "Tensor")
        .def(nb::init<nnops::Tensor &>())
        .def(nb::init<nnops::DataType &, std::vector<int> &, std::string &>())
        .def("__init_pytensor_type", [](nb::handle h) {
            PyObject *ob_self = h.ptr();
            pytensor_type = ob_self->ob_type;
        })
        .def("__str__", [](nnops::Tensor &self) { return self.to_string(); })
        .def("__repr__", &nnops::Tensor::to_repr)
        .def("__getitem__", [](nb::handle h, nb::handle indices) {
            PyObject *ob_self = h.ptr();
            nnops::Tensor *self = nb::inst_ptr<nnops::Tensor>(ob_self);
            nb::handle h_new = pytensor_new();
            nnops::Tensor *tensor_new = nb::inst_ptr<nnops::Tensor>(h_new);
            nnops::TensorMeta &meta = tensor_new->tensor_meta_;

            meta = self->tensor_meta_;
            indexing(*tensor_new, indices, 0);
            tensor_new->tensor_buffer_ = self->tensor_buffer_;
            tensor_new->tensor_buffer_->inc_ref();

            return h_new;
        })
        .def("is_contiguous", &nnops::Tensor::is_contiguous)
        .def_static("from_numpy", from_numpy)
        .def("numpy", [](nnops::Tensor &self) {
            ;
        })
        .def("reshape", [](nb::handle h, nb::args args) {
            PyObject *ob_self = h.ptr();
            nnops::Tensor *self = nb::inst_ptr<nnops::Tensor>(ob_self);
            std::vector<int> indices;

            for (int i=0; i<args.size(); i++) {
                auto v = args[i];
                if (nb::isinstance<nb::int_>(v)) {
                    indices.push_back(nb::cast<int>(v));
                } else {
                    throw std::runtime_error("only int index supported!");
                }
            }

            nb::handle pytensor = pytensor_new();
            nnops::Tensor *tensor = nb::inst_ptr<nnops::Tensor>(pytensor);
            nnops::Tensor reshaped = self->reshape(indices);

            // new (tensor) nnops::Tensor(*self);
            // nnops::Tensor::reshape(tensor, indices);
            *tensor = reshaped;

            return pytensor;
        })
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