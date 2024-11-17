#include <nnops/device.h>
#include <nnops/cpu/device.h>
#include <cstdlib>

namespace nnops::cpu {

void *Device::malloc(size_t size) {
    return std::malloc(size);
}

void Device::free(void *ptr) {
    std::free(ptr);
}

} // namespace nnops::cpu

REGISTER_DEVICE("cpu", nnops::DeviceType::CPU, nnops::cpu::Device);
