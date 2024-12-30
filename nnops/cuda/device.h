#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__

#include <nnops/device.h>

namespace nnops::cuda {

class Device final : public nnops::Device {
public:
    Device();

    void info();
    void *malloc(size_t size);
    void free(void *ptr);
    void copy_to_cpu(void *src, void *dst, size_t size);
    void copy_from_cpu(void *src, void *dst, size_t size);

private:
    static int device_count_;
    static int multiprocessor_count_;
    static int max_threads_per_multiprocessor_;
    static int warp_size_;
    static int max_threads_per_block_;
    static int max_threads_dim_[3];
    static int max_grid_size_[3];
};

} // namespace nnops::cuda

#endif // __CUDA_DEVICE_H__