#include <nnops/device.h>
#include <nnops/common.h>

namespace nnops {

Device *Device::devices_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES] = { nullptr };
std::map<std::string, Device *> Device::named_devices_;

void Device::register_device(std::string &name, DeviceType type, Device *device) {
    devices_[type] = device;
    named_devices_[name] = device;
}

Device *Device::get_device(DeviceType type) {
    return devices_[type];
}

Device *Device::get_device(std::string &name) {
    if (named_devices_.count(name))
        return named_devices_[name];
    else
        return nullptr;
}

void Device::set_device_name(std::string &name) {
    auto iter = named_devices_.find(name);
    NNOPS_CHECK(iter != named_devices_.end(), "set_device_name failed!");
    device_name_ = iter->first.c_str();
}

} // namespace nnops
