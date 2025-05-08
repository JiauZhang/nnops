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
using nnops::DataType, nnops::DeviceType, nnops::Device, nnops::index_t;

namespace pynnops {

void parse_int_args(const nb::args &args, TensorShape &indices);

} // namespace pynnops

#endif // __PYTENSOR_H__