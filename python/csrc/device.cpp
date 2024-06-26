#include <nnops/device.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("NPU", DeviceType::NPU)
        .export_values();
}