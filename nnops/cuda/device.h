#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__

#include <nnops/device.h>

class CUDADevice final : public Device {
public:
    CUDADevice(std::string name): name_(name) {}
    void *malloc(size_t size) { return 0; }
    void free(void *ptr) {}

    std::string name_;
};

#endif // __CUDA_DEVICE_H__