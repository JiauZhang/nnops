#ifndef __CPU_DEVICE_H__
#define __CPU_DEVICE_H__

#include <device.h>

class CPUDevice final : public Device {
    void *malloc(size_t size);
};

#endif // __CPU_DEVICE_H__