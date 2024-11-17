#ifndef __NPU_DEVICE_H__
#define __NPU_DEVICE_H__

#include <nnops/device.h>

namespace nnops::npu {

class Device final : public nnops::Device {
public:
    void *malloc(size_t size) { return 0; }
    void free(void *ptr) {}
};

} // namespace nnops::npu

#endif // __NPU_DEVICE_H__