#ifndef __DEVICE_TYPE_H__
#define __DEVICE_TYPE_H__

#include <map>
#include <string.h>

enum DeviceType {
    CPU,
    CUDA,
    NPU,
};

class Device {
public:
    static void register_device(std::string &name, DeviceType type, Device *device);
    static Device *get_device(DeviceType type);
    static Device *get_device(std::string &name);

    virtual void *malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;

private:
    static std::map<DeviceType, Device *> devices_;
    static std::map<std::string, Device *> named_devices_;
};

#define REGISTER_DEVICE(device_name, device_type, device_class)                                   \
struct __##device_class_register {                                                                \
    __##device_class_register(std::string dev_name, DeviceType dev_type, Device *device) {        \
        Device::register_device(dev_name, dev_type, device);                                      \
    }                                                                                             \
};                                                                                                \
static __##device_class_register *__registered_##device_class                                     \
    = new __##device_class_register(device_name, device_type, new device_class(device_name));

#endif // __DEVICE_TYPE_H__