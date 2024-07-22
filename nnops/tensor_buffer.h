#ifndef __TENSOR_BUFFER_H__
#define __TENSOR_BUFFER_H__

#include <atomic>
#include <nnops/device.h>
#include <cstdlib>

namespace nnops {

class TensorBuffer {
public:
    TensorBuffer() = delete;
    TensorBuffer(void *data_ptr, Device *device):
        data_ptr_(data_ptr), device_(device), ref_count_(1) {}

    void inc_ref() { ++ref_count_; }
    void dec_ref() {
        --ref_count_;
        if (ref_count_ == 0) {
            if (data_ptr_)
                device_->free(data_ptr_);
            std::free(this);
        }
    }
    int count() { return ref_count_; }

    void *data_ptr_;
    mutable std::atomic<int> ref_count_;
    Device *device_;
};

} // namespace nnops

#endif // __TENSOR_BUFFER_H__