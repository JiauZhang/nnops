#ifndef __CPU_DEVICE_H__
#define __CPU_DEVICE_H__

#include <nnops/device.h>
#include <string>

namespace nnops::cpu {

class Device final : public nnops::Device {
public:
    void *malloc(size_t size);
    void free(void *ptr);
};

} // namespace nnops::cpu

#endif // __CPU_DEVICE_H__