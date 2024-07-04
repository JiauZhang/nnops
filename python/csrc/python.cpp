#include <nanobind/nanobind.h>

namespace nb = nanobind;

extern void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m));
extern void DEFINE_TENSOR_MODULE(nb::module_ & (m));
extern void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m));

NB_MODULE(_C, m) {
    DEFINE_DATA_TYPE_MODULE(m);
    DEFINE_TENSOR_MODULE(m);
    DEFINE_DEVICE_TYPE_MODULE(m);
}
