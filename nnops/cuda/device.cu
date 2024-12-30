#include <nnops/device.h>
#include <nnops/cuda/device.h>
#include <cuda_runtime.h>

namespace nnops::cuda {

int Device::device_count_ = -1;

Device::Device() {
    if (device_count_ >= 0)
        return;
    cudaGetDeviceCount(&device_count_);
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