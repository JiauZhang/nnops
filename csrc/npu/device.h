#ifndef __NPU_DEVICE_H__
#define __NPU_DEVICE_H__

#include <device.h>

class NPUDevice final : public Device {
    void *malloc(size_t size) { return 0; }
};

#endif // __NPU_DEVICE_H__