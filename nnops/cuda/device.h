#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__

#include <nnops/device.h>

namespace nnops::cuda {

class Device final : public nnops::Device {
public:
    void *malloc(size_t size) { return 0; }
    void free(void *ptr) {}
};

} // namespace nnops::cuda

#endif // __CUDA_DEVICE_H__