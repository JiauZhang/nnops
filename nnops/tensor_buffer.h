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

    inline void inc_ref() { ++ref_count_; }
    void dec_ref();
    inline int count() { return ref_count_; }

    void *data_ptr_;
    mutable std::atomic<int> ref_count_;
    Device *device_;
    int size_;
};

} // namespace nnops

#endif // __TENSOR_BUFFER_H__