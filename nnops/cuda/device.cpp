#include <nnops/device.h>
#include <nnops/cuda/device.h>
#include <cuda_runtime.h>

namespace nnops::cuda {

void *Device::malloc(size_t size) {
    void *cuda_mem_ptr = nullptr;
    cudaMalloc(&cuda_mem_ptr, size);
    return cuda_mem_ptr;
}

void Device::free(void *ptr) {
    cudaFree(ptr);
}

} // namespace nnops::cuda

REGISTER_DEVICE("cuda", nnops::DeviceType::CUDA, nnops::cuda::Device);