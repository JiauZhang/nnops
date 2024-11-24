#ifndef __PYTENSOR_H__
#define __PYTENSOR_H__

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
#include <array>
#include <tuple>
#include <iostream>

namespace nb = nanobind;
using nnops::Tensor, nnops::TensorMeta, nnops::TensorShape;
using nnops::DataType, nnops::DeviceType, nnops::Device;

namespace pynnops {

class PyTensor : public Tensor {
public:
    PyTensor(): Tensor() {}
    PyTensor(const Tensor &other): Tensor(other) {}
    PyTensor(const PyTensor &other) {
        tensor_meta_ = other.tensor_meta_;
        tensor_buffer_ = other.tensor_buffer_;
        tensor_buffer_->inc_ref();
    }
    PyTensor(DataType dtype, TensorShape &dims, DeviceType device):
        Tensor(dtype, dims, device) {}
    PyTensor(nb::kwargs &kwargs);
    PyTensor py_broadcast_to(TensorShape &shape) {
        Tensor t = this->broadcast_to(shape);
        PyTensor tensor(t);
        return tensor;
    }
    PyTensor &operator=(const PyTensor &other) {
        tensor_meta_ = other.tensor_meta_;
        tensor_buffer_ = other.tensor_buffer_;
        tensor_buffer_->inc_ref();
        return *this;
    }
    PyTensor py_contiguous() {
        if (this->is_contiguous()) {
            return *this;
        } else {
            return this->py_clone();
        }
    }
    PyTensor py_clone() {
        Tensor tensor = this->clone();
        PyTensor pytensor(tensor);
        return pytensor;
    }
    nb::ndarray<nb::numpy> numpy();
    PyTensor py_reshape(nb::args args);
    PyTensor __getitem__(nb::handle indices);
    Tensor tensor() {
        Tensor t;
        t.tensor_meta_ = tensor_meta_;
        t.tensor_buffer_ = tensor_buffer_;
        t.tensor_buffer_->inc_ref();
        return t;
    }
};

} // namespace pynnops

#endif // __PYTENSOR_H__