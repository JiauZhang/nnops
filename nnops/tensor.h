#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <nnops/device.h>
#include <string>

namespace nnops {

class Tensor {
public:
    Tensor();
    Tensor(DataType &dtype, std::vector<int> &dims, std::string &device);
    Tensor(DataType &dtype, std::vector<int> &dims, DeviceType device);
    Tensor(const Tensor &other);
    Tensor(const Tensor &other, const std::vector<int> &dims);
    ~Tensor();

    void init_tensor(DataType &dtype, std::vector<int> &dims, Device *device);
    inline void reshape_inplace(std::vector<int> &dims) { this->tensor_meta_.reshape_inplace(dims); }

    inline DataType dtype() { return this->tensor_meta_.dtype_; }
    inline const std::vector<int> &shape() { return this->tensor_meta_.dims_; }
    inline const std::vector<int> &stride() { return this->tensor_meta_.strides_; }
    inline void *data_ptr() { return this->tensor_buffer_->data_ptr_; }
    inline int ndim() { return this->shape().size(); }
    inline int ref_count() { return this->tensor_buffer_->count(); }
    inline size_t nelems() { return this->tensor_meta_.nelems_; }
    inline size_t nbytes() { return this->tensor_meta_.nbytes_; }

    void to_string(std::string *prefix, std::string *ret);
    std::string to_string();
    std::string to_repr();

    TensorMeta tensor_meta_;
    TensorBuffer *tensor_buffer_;
};

} // namespace nnops

#endif // __TENSOR_H__