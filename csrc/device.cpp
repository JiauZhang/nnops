#include <nanobind/nanobind.h>
#include <device.h>

namespace nb = nanobind;

void Device::register_device(DeviceType type, Device *device) {
    devices_[type] = device;
}

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("NPU", DeviceType::NPU)
        .export_values();
}