#include <nanobind/nanobind.h>
#include <device.h>

namespace nb = nanobind;

std::map<DeviceType, Device *> Device::devices_;
std::map<std::string, Device *> Device::named_devices_;

void Device::register_device(std::string &name, DeviceType type, Device *device) {
    devices_[type] = device;
    named_devices_[name] = device;
}

Device *Device::get_device(DeviceType type) {
    if (devices_.count(type))
        return devices_[type];
    else
        return nullptr;
}

Device *Device::get_device(std::string &name) {
    if (named_devices_.count(name))
        return named_devices_[name];
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