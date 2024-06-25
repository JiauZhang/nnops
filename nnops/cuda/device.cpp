#include <device.h>
#include <cuda/device.h>

REGISTER_DEVICE("cuda", DeviceType::CUDA, CUDADevice);