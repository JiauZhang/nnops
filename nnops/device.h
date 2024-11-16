#ifndef __DEVICE_TYPE_H__
#define __DEVICE_TYPE_H__

#include <map>
#include <string.h>

namespace nnops {

enum DeviceType: unsigned char {
    CPU = 0,
    CUDA,
    NPU,
    COMPILE_TIME_MAX_DEVICE_TYPES,
};

class Device {
public:
    void set_device_name(std::string &name);
    inline std::string get_device_name() { return std::string(device_name_); }
    inline void set_device_type(DeviceType &dt) { device_type_ = dt; }
    inline DeviceType get_device_type() { return device_type_; }

    static void register_device(std::string &name, DeviceType type, Device *device);
    static Device *get_device(DeviceType type);
    static Device *get_device(std::string &name);

    virtual void *malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;

private:
    static Device *devices_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES];
    static std::map<std::string, Device *> named_devices_;
    const char *device_name_ = nullptr;
    DeviceType device_type_;
};

#define REGISTER_DEVICE(device_name, device_type, device_class)                                   \
struct __##device_class_register {                                                                \
    __##device_class_register(std::string dev_name, DeviceType dev_type, Device *device) {        \
        Device::register_device(dev_name, dev_type, device);                                      \
        device->set_device_name(dev_name);                                                        \
        device->set_device_type(dev_type);                                                        \
    }                                                                                             \
};                                                                                                \
static __##device_class_register *__registered_##device_class                                     \
    = new __##device_class_register(device_name, device_type, new device_class());

} // namespace nnops

#endif // __DEVICE_TYPE_H__