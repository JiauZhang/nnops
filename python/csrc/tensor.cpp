#include <python/csrc/tensor.h>
#include <cstdint>
#include <array>

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

void indexing(TensorMeta &meta, nb::handle indices, int axis) {
    PyObject *ob_indices = indices.ptr();

    if (nb::isinstance<nb::tuple>(indices)) {
        // multi-dimensional indexing
        Py_ssize_t len = PyTuple_Size(ob_indices);
        if (len > meta.ndim()) {
            std::string info = "too many indices for tensor: ";
            info += "tensor is " + std::to_string(meta.ndim()) + "-dimensional, but "
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
                axis = meta.ndim() + i - len + 1;
                continue;
            }

            indexing(meta, indices[i], axis);
            if (nb::isinstance<nb::slice>(indices[i]))
                axis += 1;
        }
    } else if (nb::isinstance<nb::slice>(indices)) {
        Py_ssize_t start, stop, step;

        if (PySlice_GetIndices(ob_indices, meta.shape()[axis], &start, &stop, &step) < 0)
            throw std::runtime_error("PySlice_GetIndices failed!");

        nnops::Slice slice(start, stop, step);
        meta.slice_inplace(slice, axis);
    } else if (nb::isinstance<nb::int_>(indices)) {
        meta.index_inplace(nb::cast<int>(indices), axis);
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

    Tensor::init_tensor(dtype, shape, device);
}

PyTensor PyTensor::__getitem__(nb::handle indices) {
    TensorMeta meta = this->meta();
    indexing(meta, indices, 0);
    return PyTensor(meta, this->buffer());
}

void parse_int_args(nb::args &args, TensorShape &indices) {
    for (int i=0; i<args.size(); i++) {
        auto v = args[i];
        if (nb::isinstance<nb::int_>(v)) {
            indices.push_back(nb::cast<int>(v));
        } else {
            throw std::runtime_error("only int index supported!");
        }
    }
}

PyTensor PyTensor::py_reshape(nb::args args) {
    TensorShape indices;
    parse_int_args(args, indices);
    Tensor &&tensor = this->reshape(indices);
    return PyTensor(tensor);
}

PyTensor PyTensor::py_permute(nb::args args) {
    TensorShape indices;
    parse_int_args(args, indices);
    Tensor &&tensor = this->permute(indices);
    return PyTensor(tensor);
}

static constexpr std::array<
    nb::dlpack::dtype, DataType::COMPILE_TIME_MAX_DATA_TYPES> __nptypes__ = {
    nb::dtype<double>(), nb::dtype<float>(),
    nb::dtype<uint64_t>(), nb::dtype<int64_t>(),
    nb::dtype<uint32_t>(), nb::dtype<int32_t>(),
    nb::dtype<uint16_t>(), nb::dtype<int16_t>(),
    nb::dtype<uint8_t>(), nb::dtype<int8_t>(),
    nb::dtype<bool>(),
};
static constexpr std::array<DataType, DataType::COMPILE_TIME_MAX_DATA_TYPES> __types__ = {
    DataType::TYPE_FLOAT64, DataType::TYPE_FLOAT32,
    DataType::TYPE_UINT64, DataType::TYPE_INT64,
    DataType::TYPE_UINT32, DataType::TYPE_INT32,
    DataType::TYPE_UINT16, DataType::TYPE_INT16,
    DataType::TYPE_UINT8, DataType::TYPE_INT8,
    DataType::TYPE_BOOL,
};

template<typename T>
static int match_dtype(const std::array<T, __types__.size()> &types, T &type) {
    for (int i=0; i<types.size(); i++)
        if (types[i] == type)
            return i;
    throw std::runtime_error("match_dtype failed!");
}

nb::ndarray<nb::numpy> PyTensor::numpy() {
    Tensor t = this->clone();
    PyTensor *tensor = new PyTensor(t);
    std::vector<size_t> shape;

    nb::capsule deleter(tensor, [](void *p) noexcept {
        delete (PyTensor *)p;
    });
    for (auto s: tensor->shape())
        shape.push_back(s);

    DataType tensor_dtype = tensor->dtype();
    int idx = match_dtype<DataType>(__types__, tensor_dtype);
    nb::dlpack::dtype dtype = __nptypes__[idx];

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

    for (int i=0; i<array.ndim(); i++)
        shape.push_back(array.shape(i));

    nb::dlpack::dtype array_dtype = array.dtype();
    int idx = match_dtype<nb::dlpack::dtype>(__nptypes__, array_dtype);
    DataType dtype = __types__[idx];
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
        .def("astype", &PyTensor::astype)
        .def("to", &PyTensor::to)
        .def("numpy", &PyTensor::numpy)
        .def("reshape", &PyTensor::py_reshape)
        .def("permute", &PyTensor::py_permute)
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