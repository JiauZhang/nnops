#ifndef __CPU_DEVICE_H__
#define __CPU_DEVICE_H__

#include <nnops/device.h>
#include <string>

namespace nnops {

class CPUDevice final : public Device {
public:
    CPUDevice(std::string name): name_(name) {}
    void *malloc(size_t size);
    void free(void *ptr);

    std::string name_;
};

} // namespace nnops

#endif // __CPU_DEVICE_H__