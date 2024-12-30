#ifndef __DEVICE_TYPE_H__
#define __DEVICE_TYPE_H__

#include <map>
#include <string>
#include <cstdint>

namespace nnops {

enum DeviceType : uint8_t {
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

    virtual void info() {}
    virtual void *malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;
    virtual void copy_to_cpu(void *src, void *dst, size_t size) = 0;
    virtual void copy_from_cpu(void *src, void *dst, size_t size) = 0;

private:
    static Device *devices_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES];
    static std::map<std::string, Device *> named_devices_;
    const char *device_name_ = nullptr;
    DeviceType device_type_;
};

} // namespace nnops

#define REGISTER_DEVICE(device_name, device_type, device_class)                                   \
struct __local_device_register__ {                                                                \
    __local_device_register__(                                                                    \
        std::string dev_name, nnops::DeviceType dev_type, nnops::Device *device) {                \
        nnops::Device::register_device(dev_name, dev_type, device);                               \
        device->set_device_name(dev_name);                                                        \
        device->set_device_type(dev_type);                                                        \
    }                                                                                             \
};                                                                                                \
static __local_device_register__ *__registered_local_device__                                     \
    = new __local_device_register__(device_name, device_type, new device_class())

#endif // __DEVICE_TYPE_H__