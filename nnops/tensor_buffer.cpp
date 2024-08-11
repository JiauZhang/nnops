#include <nnops/tensor_buffer.h>

namespace nnops {

TensorBuffer::TensorBuffer(Device *device, int size):
    device_(device), ref_count_(1), size_(size) {
    data_ptr_ = device_->malloc(size_);
    if (data_ptr_ == nullptr)
        throw std::runtime_error("alloc TensorBuffer memory failed!");
}

void TensorBuffer::dec_ref() {
    --ref_count_;
    if (ref_count_ == 0) {
        if (data_ptr_)
            device_->free(data_ptr_);
        std::free(this);
    }
}

} // namespace nnops