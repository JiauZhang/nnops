#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__

#include <device.h>

class CUDADevice final : public Device {
    void *malloc(size_t size) { return 0; }
};

#endif // __CUDA_DEVICE_H__