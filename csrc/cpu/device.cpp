#include <device.h>
#include <cpu/device.h>
#include <cstdlib>

void *CPUDevice::malloc(size_t size) {
    return std::malloc(size);
}

REGISTER_DEVICE(DeviceType::CPU, CPUDevice);