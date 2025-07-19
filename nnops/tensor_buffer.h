#ifndef __TENSOR_BUFFER_H__
#define __TENSOR_BUFFER_H__

#include <atomic>
#include <nnops/device.h>
#include <cstdlib>

namespace nnops {

class TensorBuffer {
public:
    TensorBuffer() = delete;
    TensorBuffer(Device *device, int size);

    void *data_ptr_;
    Device *device_;
    int size_;
};

} // namespace nnops

#endif // __TENSOR_BUFFER_H__