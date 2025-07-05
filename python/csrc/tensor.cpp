#include <nnops/data_type.h>
#include <python/csrc/tensor.h>
#include <python/csrc/binary_ops.h>
#include <cstdint>
#include <array>

using namespace nb::literals;

namespace pynnops {

std::string tp_name(nb::handle &h) {
    PyObject *ob = h.ptr();
    PyTypeObject *tp = ob->ob_type;
    return std::string(tp->tp_name);
}

DataType parse_data_type(nb::handle h) {
    NNOPS_CHECK(nb::isinstance<DataType>(h), "unsupported DataType: %s", h.ptr()->ob_type->tp_name);
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
        NNOPS_CHECK(nb::isinstance<nb::int_>(h[i]), "Only int data type is supported for shape dimensions!");
        shape.push_back(nb::cast<int>(h[i]));
    }

    return shape;
}

void indexing(TensorMeta &meta, nb::handle indices, int axis) {
    PyObject *ob_indices = indices.ptr();

    if (nb::isinstance<nb::tuple>(indices)) {
        // multi-dimensional indexing
        Py_ssize_t len = PyTuple_Size(ob_indices);
        NNOPS_CHECK(len <= meta.ndim(), "too many indices for tensor: tensor is %d-dimensional, but %ld were indexed", meta.ndim(), len);

        // check ellipsis
        int idx = len, count = 0;
        for (int i=0; i<len; i++) {
            if (nb::isinstance<nb::ellipsis>(indices[i])) {
                count++;
                idx = i;
            }
        }

        NNOPS_CHECK(count <= 1, "an index can only have a single ellipsis ('...')");

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

        NNOPS_CHECK(PySlice_GetIndices(ob_indices, meta.shape()[axis], &start, &stop, &step) >= 0, "PySlice_GetIndices failed!");

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

Tensor create_pytensor(nb::kwargs &kwargs) {
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

    return Tensor(dtype, shape, device);
}

Tensor __getitem__(Tensor &self, nb::handle indices) {
    TensorMeta meta = self.meta();
    indexing(meta, indices, 0);
    return Tensor(meta, self.buffer());
}

void parse_int_args(const nb::args &args, TensorShape &indices) {
    for (int i=0; i<args.size(); i++) {
        auto v = args[i];
        if (nb::isinstance<nb::int_>(v)) {
            indices.push_back(nb::cast<int>(v));
        } else {
            throw std::runtime_error("only int index supported!");
        }
    }
}

Tensor reshape(const Tensor &self, const nb::args &args) {
    TensorShape indices;
    parse_int_args(args, indices);
    return self.reshape(indices);
}

Tensor permute(const Tensor &self, const nb::args &args) {
    TensorShape indices;
    parse_int_args(args, indices);
    return self.permute(indices);
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

nb::ndarray<nb::numpy> numpy(Tensor &self) {
    // numpy must independently own a tensor
    Tensor *tensor = new Tensor(self.clone());
    std::vector<size_t> shape;

    nb::capsule deleter(tensor, [](void *p) noexcept {
        delete (Tensor *)p;
    });
    for (auto s: tensor->shape())
        shape.push_back(s);

    DataType tensor_dtype = tensor->dtype();
    int idx = match_dtype<DataType>(__types__, tensor_dtype);
    nb::dlpack::dtype dtype = __nptypes__[idx];

    return nb::ndarray<nb::numpy>(
            tensor->data_ptr(), tensor->ndim(), shape.data(), deleter, nullptr, dtype);
}

void from_numpy_impl(nb::ndarray<> *src, int src_offset, Tensor *dst, int dst_offset, int axis) {
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

Tensor from_numpy(nb::ndarray<> array) {
    TensorShape shape;

    for (int i=0; i<array.ndim(); i++)
        shape.push_back(array.shape(i));

    nb::dlpack::dtype array_dtype = array.dtype();
    int idx = match_dtype<nb::dlpack::dtype>(__nptypes__, array_dtype);
    DataType dtype = __types__[idx];
    Tensor tensor(dtype, shape, DeviceType::CPU);

    from_numpy_impl(&array, 0, &tensor, 0, 0);

    return tensor;
}

void DEFINE_TENSOR_MODULE(nb::module_ & (m)) {
    m.def("from_numpy", &from_numpy);
    m.def("is_broadcastable", [](Tensor &t1, Tensor &t2) {
        return Tensor::is_broadcastable(t1.shape(), t2.shape(), 0); });
    m.def("broadcast_shape", [](Tensor &t1, Tensor &t2) {
        return Tensor::broadcast_shape(t1.shape(), t2.shape(), 0); });

    // https://nanobind.readthedocs.io/en/latest/classes.html#overloaded-methods
    auto &pytensor = nb::class_<Tensor>(m, "Tensor")
        .def(nb::new_([](nb::kwargs &kwargs) { return create_pytensor(kwargs); }))
        .def("__str__", nb::overload_cast<>(&Tensor::to_string, nb::const_))
        .def("__repr__", &Tensor::to_repr)
        .def("__getitem__", &__getitem__)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("astype", &Tensor::astype)
        .def("to", &Tensor::to)
        .def("numpy", &numpy)
        .def("reshape", &reshape)
        .def("permute", &permute)
        .def("transpose", [](Tensor &self, index_t dim0, index_t dim1) { return Tensor::transpose(self, dim0, dim1); })
        .def("broadcast_to", [](Tensor &self, TensorShape &shape) { return Tensor::broadcast_to(self, shape, 0); })
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("device", [](Tensor &self) { return self.device()->get_device_type(); })
        .def_prop_ro("data_ptr", [](Tensor &self) { return self.data_ptr(0); })
        .def_prop_ro("ref_count", &Tensor::ref_count)
        .def_prop_ro("ndim", &Tensor::ndim)
        .def_prop_ro("nbytes", &Tensor::nbytes)
        .def_prop_ro("nelems", &Tensor::nelems)
        .def_prop_ro("stride", nb::overload_cast<>(&Tensor::stride, nb::const_))
        .def_prop_ro("shape", nb::overload_cast<>(&Tensor::shape, nb::const_));

    pytensor.def("__matmul__", &matmul);

    // tensor-tensor binary ops
    #define MAKE_BINARY_OP_TENSOR_TENSOR_BINDING(op_type, op_name, op) pytensor.def("__"#op_name"__", &op_name##_tensor_tensor);
    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_BINDING)

    // tensor-scalar binary ops
    #define MAKE_BINARY_OP_TENSOR_SCALAR_BINDING(op_type, op_name, op, type) \
    pytensor.def("__"#op_name"__", &op_name##type##_tensor_scalar); \
    pytensor.def("__r"#op_name"__", &op_name##type##_tensor_scalar_reverse);
    #define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING(dtype, type) \
    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_BINDING, type)
    DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING)
}

} // namespace pynnops