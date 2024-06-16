#include <device.h>
#include <cpu/device.h>
#include <cstdlib>

void *CPUDevice::malloc(size_t size) {
    return std::malloc(size);
}

void CPUDevice::free(void *ptr) {
    std::free(ptr);
}

REGISTER_DEVICE("cpu", DeviceType::CPU, CPUDevice);