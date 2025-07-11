#include <nnops/tensor_buffer.h>
#include <nnops/common.h>

namespace nnops {

TensorBuffer::TensorBuffer(Device *device, int size):
    device_(device), ref_count_(1), size_(size) {
    data_ptr_ = device_->malloc(size_);
    NNOPS_CHECK(data_ptr_ != nullptr, "alloc TensorBuffer memory failed on %s device!", device_->get_device_cname());
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