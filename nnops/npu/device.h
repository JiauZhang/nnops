#ifndef __NPU_DEVICE_H__
#define __NPU_DEVICE_H__

#include <nnops/device.h>

class NPUDevice final : public Device {
public:
    void *malloc(size_t size) { return 0; }
    void free(void *ptr) {}
};

#endif // __NPU_DEVICE_H__