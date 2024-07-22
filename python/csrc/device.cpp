#include <nnops/device.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynnops {

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<nnops::DeviceType>(m, "DeviceType")
        .value("CPU", nnops::DeviceType::CPU)
        .value("CUDA", nnops::DeviceType::CUDA)
        .value("NPU", nnops::DeviceType::NPU)
        .export_values();
}

} // namespace pynnops