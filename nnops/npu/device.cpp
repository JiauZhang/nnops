#include <nnops/device.h>
#include <nnops/npu/device.h>

REGISTER_DEVICE("npu", DeviceType::NPU, NPUDevice);