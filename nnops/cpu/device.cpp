#include <nnops/device.h>
#include <nnops/cpu/device.h>
#include <cstdlib>
#include <cstring>

namespace nnops::cpu {

void *Device::malloc(size_t size) {
    return std::malloc(size);
}

void Device::free(void *ptr) {
    std::free(ptr);
}

void Device::copy_to_cpu(void *src, void *dst, size_t size) {
    memcpy(dst, src, size);
}

void Device::copy_from_cpu(void *src, void *dst, size_t size) {
    memcpy(dst, src, size);
}

} // namespace nnops::cpu

REGISTER_DEVICE("cpu", nnops::DeviceType::CPU, nnops::cpu::Device);
