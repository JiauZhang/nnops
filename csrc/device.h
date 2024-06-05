#ifndef __DEVICE_TYPE_H__
#define __DEVICE_TYPE_H__

#include <nanobind/nanobind.h>
#include <map>

namespace nb = nanobind;

enum DeviceType {
    CPU,
    CUDA,
    NPU,
};

class Device {
public:
    static void register_device(DeviceType type, Device *device);

private:
    static std::map<DeviceType, Device *> devices_;
};

#define REGISTER_DEVICE(device_type, device_class)                      \
struct __##device_class_register {                                      \
    __##device_class_register(DeviceType type, Device *device) {        \
        Device::register_device(type, device);                          \
    }                                                                   \
};                                                                      \
static __##device_class_register *__registered_##device_class           \
    = new __##device_class_register(device_type, new device_class());

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m));

#endif // __DEVICE_TYPE_H__