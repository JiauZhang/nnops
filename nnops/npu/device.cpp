#include <nnops/device.h>
#include <nnops/npu/device.h>

REGISTER_DEVICE("npu", nnops::DeviceType::NPU, nnops::npu::Device);