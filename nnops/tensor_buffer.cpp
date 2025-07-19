#include <nnops/tensor_buffer.h>
#include <nnops/common.h>

namespace nnops {

TensorBuffer::TensorBuffer(Device *device, int size): device_(device), size_(size) {
    data_ptr_ = device_->malloc(size_);
    NNOPS_CHECK(data_ptr_ != nullptr, "alloc TensorBuffer memory failed on %s device!", device_->get_device_cname());
}

} // namespace nnops