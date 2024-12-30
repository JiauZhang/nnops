#include <nnops/device.h>
#include <nnops/cuda/device.h>
#include <cuda_runtime.h>

namespace nnops::cuda {

int Device::device_count_ = -1;
int Device::warp_size_ = -1;
int Device::max_threads_per_block_ = -1;
int Device::max_threads_dim_[3];
int Device::max_grid_size_[3];

Device::Device() {
    if (device_count_ >= 0)
        return;
    cudaGetDeviceCount(&device_count_);
    if (device_count_ > 0) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, 0);
        warp_size_ = dp.warpSize;
        max_threads_per_block_ = dp.maxThreadsPerBlock;
        for (int i=0; i<3; i++) {
            max_threads_dim_[i] = dp.maxThreadsDim[i];
            max_grid_size_[i] = dp.maxGridSize[i];
        }
    } else {
        device_count_ = 0;
    }
}

void *Device::malloc(size_t size) {
    void *cuda_mem_ptr = nullptr;
    cudaMalloc(&cuda_mem_ptr, size);
    return cuda_mem_ptr;
}

void Device::free(void *ptr) {
    cudaFree(ptr);
}

void Device::copy_to_cpu(void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void Device::copy_from_cpu(void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

} // namespace nnops::cuda