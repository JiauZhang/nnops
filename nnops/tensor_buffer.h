#ifndef __TENSOR_BUFFER_H__
#define __TENSOR_BUFFER_H__

#include <atomic>

class TensorBuffer {
public:
    TensorBuffer(void *data_ptr): data_ptr_(data_ptr), ref_count_(1) {}

    void inc_ref() { ++ref_count_; }
    void dec_ref() { --ref_count_; }
    bool is_zero() { return ref_count_ == 0; }
    int count() { return ref_count_; }

    void *data_ptr_;
    mutable std::atomic<int> ref_count_;

private:
    TensorBuffer() {};
};

#endif // __TENSOR_BUFFER_H__