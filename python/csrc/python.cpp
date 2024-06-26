#include <nanobind/nanobind.h>
#include <tensor_shape.h>
#include <tensor.h>
#include <device.h>

namespace nb = nanobind;

extern void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m));

NB_MODULE(_C, m) {
    DEFINE_TENSOR_SHAPE_MODULE(m);
    DEFINE_DATA_TYPE_MODULE(m);
    DEFINE_TENSOR_MODULE(m);
    DEFINE_DEVICE_TYPE_MODULE(m);
}
