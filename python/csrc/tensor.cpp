#include <python/csrc/tensor.h>

namespace pynnops {

std::string tp_name(nb::handle &h) {
    PyObject *ob = h.ptr();
    PyTypeObject *tp = ob->ob_type;
    return std::string(tp->tp_name);
}

DataType parse_data_type(nb::handle h) {
    if (!nb::isinstance<DataType>(h))
        throw std::runtime_error("unsupported DataType: " + tp_name(h));
    return nb::cast<DataType>(h);
}

Device *parse_device_type(nb::handle h) {
    if (nb::isinstance<nb::str>(h)) {
        std::string name = nb::cast<std::string>(h);
        return Device::get_device(name);
    } else if (nb::isinstance<DeviceType>(h)) {
        DeviceType device = nb::cast<DeviceType>(h);
        return Device::get_device(device);
    } else {
        throw std::runtime_error("unsupported DeviceType: " + tp_name(h));
    }
}

TensorShape parse_tensor_shape(nb::handle h) {
    TensorShape shape;
    Py_ssize_t len;
    PyObject *ob = h.ptr();

    if (nb::isinstance<nb::tuple>(h)) {
        len = PyTuple_Size(ob);
    } else if (nb::isinstance<nb::list>(h)) {
        len = PyList_Size(ob);
    } else {
        throw std::runtime_error("Only list or tuple is supported for TensorShape!");
    }

    for (int i=0; i<len; i++) {
        if (!nb::isinstance<nb::int_>(h[i]))
            throw std::runtime_error("Only int data type is supported for shape dimensions!");
        shape.push_back(nb::cast<int>(h[i]));
    }

    return shape;
}

void indexing(Tensor &tensor, nb::handle indices, int axis) {
    PyObject *ob_indices = indices.ptr();
    TensorMeta &meta = tensor.tensor_meta_;

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

PyTensor::PyTensor(nb::kwargs &kwargs) {
    DataType dtype = DataType::TYPE_FLOAT32;
    Device *device = Device::get_device(DeviceType::CPU);
    TensorShape shape;
    std::array<std::string, 3> keys_ = {"dtype", "device", "shape"};

    if (kwargs.contains(keys_[0])) {
        auto val = kwargs[keys_[0].c_str()];
        dtype = parse_data_type(val);
        nb::del(val);
    }
    if (kwargs.contains(keys_[1])) {
        auto val = kwargs[keys_[1].c_str()];
        device = parse_device_type(val);
        nb::del(val);
    }
    if (kwargs.contains(keys_[2])) {
        auto val = kwargs[keys_[2].c_str()];
        shape = parse_tensor_shape(val);
        nb::del(val);
    }

    init_tensor(dtype, shape, device);
}

PyTensor PyTensor::__getitem__(nb::handle indices) {
    Tensor t = this->tensor();
    indexing(t, indices, 0);
    return PyTensor(t);
}

PyTensor PyTensor::py_reshape(nb::args args) {
    TensorShape indices;

    for (int i=0; i<args.size(); i++) {
        auto v = args[i];
        if (nb::isinstance<nb::int_>(v)) {
            indices.push_back(nb::cast<int>(v));
        } else {
            throw std::runtime_error("only int index supported!");
        }
    }

    Tensor &&tensor = this->reshape(indices);
    return PyTensor(tensor);
}

nb::ndarray<nb::numpy> PyTensor::numpy() {
    Tensor t = this->clone();
    PyTensor *tensor = new PyTensor(t);
    std::vector<size_t> shape;
    nb::dlpack::dtype dtype;

    nb::capsule deleter(tensor, [](void *p) noexcept {
        delete (PyTensor *)p;
    });
    for (auto s: tensor->shape())
        shape.push_back(s);

    if (tensor->dtype() == DataType::TYPE_FLOAT32)
        dtype = nb::dtype<float>();
    else if (tensor->dtype() == DataType::TYPE_INT32)
        dtype = nb::dtype<int>();
    else if (tensor->dtype() == DataType::TYPE_UINT32)
        dtype = nb::dtype<unsigned int>();
    else if (tensor->dtype() == DataType::TYPE_INT16)
        dtype = nb::dtype<short>();
    else if (tensor->dtype() == DataType::TYPE_UINT16)
        dtype = nb::dtype<unsigned short>();
    else if (tensor->dtype() == DataType::TYPE_INT8)
        dtype = nb::dtype<char>();
    else if (tensor->dtype() == DataType::TYPE_UINT8)
        dtype = nb::dtype<unsigned char>();
    else
        throw std::runtime_error("numpy() invalid DataType!");

    return nb::ndarray<nb::numpy>(
            tensor->data_ptr(), tensor->ndim(), shape.data(), deleter, nullptr, dtype);
}

void from_numpy_impl(nb::ndarray<> *src, int src_offset, PyTensor *dst, int dst_offset, int axis) {
    if (axis < src->ndim() - 1) {
        for (int i=0; i<src->shape(axis); i++)
            from_numpy_impl(
                src, src_offset + i * (src->stride(axis)),
                dst, dst_offset + i * (dst->stride()[axis]),
                axis + 1
            );
    }

    int itemsize = sizeof_dtype(dst->dtype());
    unsigned char *src_ptr = (unsigned char *)src->data() + src_offset * itemsize;
    unsigned char *dst_ptr = (unsigned char *)dst->data_ptr() + dst_offset * itemsize;
    auto &dst_stride = dst->stride();
    for (int i=0; i<src->shape(axis); i++) {
        for (int j=0; j<itemsize; j++)
            dst_ptr[j] = src_ptr[j];
        src_ptr += src->stride(axis) * itemsize;
        dst_ptr += dst_stride[axis] * itemsize;
    }
}

PyTensor from_numpy(nb::ndarray<> array) {
    TensorShape shape;
    DataType dtype;
    auto array_dtype = array.dtype();

    for (int i=0; i<array.ndim(); i++)
        shape.push_back(array.shape(i));

    if (array_dtype == nb::dtype<char>())
        dtype = DataType::TYPE_INT8;
    else if (array_dtype == nb::dtype<unsigned char>())
        dtype = DataType::TYPE_UINT8;
    else if (array_dtype == nb::dtype<unsigned short>())
        dtype = DataType::TYPE_UINT16;
    else if (array_dtype == nb::dtype<short>())
        dtype = DataType::TYPE_INT16;
    else if (array_dtype == nb::dtype<int>())
        dtype = DataType::TYPE_INT32;
    else if (array_dtype == nb::dtype<unsigned int>())
        dtype = DataType::TYPE_UINT32;
    else if (array_dtype == nb::dtype<float>())
        dtype = DataType::TYPE_FLOAT32;
    else
        throw std::runtime_error("invalid from_numpy dtype!");

    PyTensor tensor(dtype, shape, DeviceType::CPU);
    from_numpy_impl(&array, 0, &tensor, 0, 0);

    return tensor;
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    m.def("from_numpy", &from_numpy);
    m.def("is_broadcastable", [](PyTensor &t1, PyTensor &t2) {
        return PyTensor::is_broadcastable(t1.shape(), t2.shape()); });
    m.def("broadcast_shape", [](PyTensor &t1, PyTensor &t2) {
        return PyTensor::broadcast_shape(t1.shape(), t2.shape()); });

    nb::class_<PyTensor>(m, "PyTensor")
        .def(nb::init<nb::kwargs &>())
        .def("__str__", [](PyTensor &self) { return self.to_string(); })
        .def("__repr__", &PyTensor::to_repr)
        .def("__getitem__", &PyTensor::__getitem__)
        .def("is_contiguous", &PyTensor::is_contiguous)
        .def("contiguous", &PyTensor::py_contiguous)
        .def("clone", &PyTensor::py_clone)
        .def("numpy", &PyTensor::numpy)
        .def("reshape", &PyTensor::py_reshape)
        .def("broadcast_to", &PyTensor::py_broadcast_to)
        .def_prop_ro("dtype", [](PyTensor &t) { return t.dtype(); })
        .def_prop_ro("device", [](PyTensor &t) { return t.device()->get_device_type(); })
        .def_prop_ro("data_ptr", [](PyTensor &t) { return t.data_ptr(); })
        .def_prop_ro("ref_count", [](PyTensor &t) { return t.ref_count(); })
        .def_prop_ro("ndim", [](PyTensor &t) { return t.ndim(); })
        .def_prop_ro("nbytes", [](PyTensor &t) { return t.nbytes(); })
        .def_prop_ro("nelems", [](PyTensor &t) { return t.nelems(); })
        .def_prop_ro("stride", [](PyTensor &t) { return t.stride(); })
        .def_prop_ro("shape", [](PyTensor &t) { return t.shape(); });
}

} // namespace pynnops