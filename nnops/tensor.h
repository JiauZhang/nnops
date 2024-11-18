#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <nnops/device.h>
#include <string>
#include <optional>

namespace nnops {

class Tensor {
public:
    Tensor();
    Tensor(DataType dtype, std::vector<int> &dims, std::string &device);
    Tensor(DataType dtype, std::vector<int> &dims, DeviceType device);
    Tensor(DataType dtype, std::vector<int> &dims, Device *device);
    Tensor(const Tensor &other);
    ~Tensor();

    Tensor &operator=(Tensor &other);
    void init_tensor(DataType &dtype, std::vector<int> &dims, Device *device);
    inline void reshape_inplace(std::vector<int> &dims) { this->tensor_meta_.reshape_inplace(dims); }
    Tensor reshape(std::vector<int> &dims);
    inline static bool is_broadcastable(Tensor &t1, Tensor &t2) { return is_broadcastable(t1.shape(), t2.shape()); }
    static bool is_broadcastable(std::vector<int> &s1, std::vector<int> &s2);
    inline static std::vector<int> broadcast_shape(Tensor &t1, Tensor &t2) { return broadcast_shape(t1.shape(), t2.shape()); }
    static std::vector<int> broadcast_shape(std::vector<int> &s1, std::vector<int> &s2);
    bool is_broadcast();
    inline Tensor broadcast_to(std::vector<int> &shape) { return Tensor::broadcast_to(*this, shape); }
    static Tensor broadcast_to(Tensor &t, std::vector<int> &shape);

    inline DataType dtype() { return this->tensor_meta_.dtype_; }
    inline std::vector<int> &shape() { return this->tensor_meta_.dims_; }
    inline const std::vector<int> &stride() { return this->tensor_meta_.strides_; }
    inline void *data_ptr() { return this->tensor_buffer_->data_ptr_; }
    inline int ndim() { return this->shape().size(); }
    inline int ref_count() { return this->tensor_buffer_->count(); }
    inline Device *device() { return this->tensor_buffer_->device_; }
    inline size_t nelems() { return this->tensor_meta_.nelems_; }
    inline size_t nbytes() { return this->tensor_meta_.nbytes_; }
    inline int offset() { return this->tensor_meta_.offset_; }
    inline bool is_contiguous() {return this->tensor_meta_.is_contiguous(); }
    Tensor clone();
    Tensor contiguous();

    void to_string(std::string *prefix, std::string *ret);
    std::string to_string();
    std::string to_repr();

    TensorMeta tensor_meta_;
    TensorBuffer *tensor_buffer_;
};

} // namespace nnops

#endif // __TENSOR_H__