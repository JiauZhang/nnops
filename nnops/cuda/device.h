#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__

#include <nnops/device.h>

namespace nnops::cuda {

class Device final : public nnops::Device {
public:
    void *malloc(size_t size);
    void free(void *ptr);
    void copy_to_cpu(void *src, void *dst, size_t size);
    void copy_from_cpu(void *src, void *dst, size_t size);
};

} // namespace nnops::cuda

#endif // __CUDA_DEVICE_H__