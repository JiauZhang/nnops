#include <nnops/device.h>
#include <nnops/cuda/device.h>

REGISTER_DEVICE("cuda", DeviceType::CUDA, CUDADevice);