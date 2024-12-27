#include <nnops/device.h>
#include <nnops/cuda/device.h>

REGISTER_DEVICE("cuda", nnops::DeviceType::CUDA, nnops::cuda::Device);