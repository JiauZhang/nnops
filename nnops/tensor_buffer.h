#ifndef __TENSOR_BUFFER_H__
#define __TENSOR_BUFFER_H__

#include <atomic>
#include <nnops/device.h>
#include <cstdlib>

class TensorBuffer {
public:
    TensorBuffer(void *data_ptr, Device *device):
        data_ptr_(data_ptr), device_(device), ref_count_(1) {}

    void inc_ref() { ++ref_count_; }
    void dec_ref() { --ref_count_; }
    bool is_zero() { return ref_count_ == 0; }
    int count() { return ref_count_; }
    void free() {
        if (data_ptr_)
            device_->free(data_ptr_);
        std::free(this);
    }

    void *data_ptr_;
    mutable std::atomic<int> ref_count_;
    Device *device_;

private:
    TensorBuffer() {};
};

#endif // __TENSOR_BUFFER_H__