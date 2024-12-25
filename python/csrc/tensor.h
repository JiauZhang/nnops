#ifndef __PYTENSOR_H__
#define __PYTENSOR_H__

#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor_buffer.h>
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
using nnops::Tensor, nnops::TensorMeta, nnops::TensorShape, nnops::TensorBuffer;
using nnops::DataType, nnops::DeviceType, nnops::Device;

namespace pynnops {

class PyTensor : public Tensor {
public:
    PyTensor(): Tensor() {}
    PyTensor(const Tensor &other): Tensor(other) {}
    PyTensor(const PyTensor &other) : Tensor(other.meta(), other.buffer()) {}
    PyTensor(const TensorMeta &meta, TensorBuffer *buffer) : Tensor(meta, buffer) {}
    PyTensor(DataType dtype, TensorShape &dims, DeviceType device):
        Tensor(dtype, dims, device) {}
    PyTensor(nb::kwargs &kwargs);
    PyTensor py_broadcast_to(TensorShape &shape) {
        Tensor t = this->broadcast_to(shape);
        PyTensor tensor(t);
        return tensor;
    }
    PyTensor &operator=(const PyTensor &other) {
        if (this != &other) {
            set_meta(other.meta());
            set_buffer(other.buffer());
        }
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
    PyTensor py_permute(nb::args args);
    PyTensor __getitem__(nb::handle indices);
    Tensor tensor() { return Tensor(this->meta(), this->buffer()); }
    PyTensor astype(DataType dtype) {
        Tensor t = Tensor::astype(dtype);
        return PyTensor(t);
    }
    PyTensor to(DeviceType device) {
        Tensor t = Tensor::to(device);
        return PyTensor(t);
    }
};

} // namespace pynnops

#endif // __PYTENSOR_H__