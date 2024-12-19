#include <nnops/tensor_buffer.h>
#include <stdexcept>

namespace nnops {

TensorBuffer::TensorBuffer(Device *device, int size):
    device_(device), ref_count_(1), size_(size) {
    data_ptr_ = device_->malloc(size_);
    if (data_ptr_ == nullptr) {
        std::string info = "alloc TensorBuffer memory failed on " + device_->get_device_name() + " device!";
        throw std::runtime_error(info);
    }
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