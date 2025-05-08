#include <nnops/random.h>
#include <nanobind/nanobind.h>
#include <nnops/tensor.h>
#include <python/csrc/tensor.h>
#include <nnops/data_type.h>
#include <nnops/device.h>

namespace nb = nanobind;
using nnops::TensorShape, nnops::DeviceType, nnops::DataType;

namespace pynnops {

Tensor randn(nb::args &args) {
    TensorShape indices;
    parse_int_args(args, indices);
    Tensor tensor(DataType::TYPE_FLOAT32, indices, DeviceType::CPU);
    nnops::RandN randn(0, 1);
    randn.sample((float *)tensor.data_ptr(), tensor.nelems());
    return tensor;
}

void DEFINE_RANDOM_MODULE(nb::module_ & (m)) {
    m.def("randn", &randn);
}

} // namespace pynnops