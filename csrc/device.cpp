#include <nanobind/nanobind.h>
#include <device.h>

namespace nb = nanobind;

std::map<DeviceType, Device *> Device::devices_;

void Device::register_device(DeviceType type, Device *device) {
    devices_[type] = device;
}

Device *Device::get_device(DeviceType type) {
    if (devices_.count(type))
        return devices_[type];
    else
        return nullptr;
}

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("NPU", DeviceType::NPU)
        .export_values();
}