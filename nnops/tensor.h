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
    Tensor(DataType dtype, TensorShape &dims, std::string &device);
    Tensor(DataType dtype, TensorShape &dims, DeviceType device);
    Tensor(DataType dtype, TensorShape &dims, Device *device);
    Tensor(const Tensor &other);
    ~Tensor();

    Tensor &operator=(Tensor &other);
    void init_tensor(DataType &dtype, TensorShape &dims, Device *device);
    inline void reshape_inplace(TensorShape &dims) { this->tensor_meta_.reshape_inplace(dims); }
    Tensor reshape(TensorShape &dims);
    inline static bool is_broadcastable(Tensor &t1, Tensor &t2) { return is_broadcastable(t1.shape(), t2.shape()); }
    static bool is_broadcastable(TensorShape &s1, TensorShape &s2);
    inline static TensorShape broadcast_shape(Tensor &t1, Tensor &t2) { return broadcast_shape(t1.shape(), t2.shape()); }
    static TensorShape broadcast_shape(TensorShape &s1, TensorShape &s2);
    bool is_broadcast();
    inline Tensor broadcast_to(TensorShape &shape) { return Tensor::broadcast_to(*this, shape); }
    static Tensor broadcast_to(Tensor &t, TensorShape &shape);

    inline DataType dtype() { return this->tensor_meta_.dtype_; }
    inline TensorShape &shape() { return this->tensor_meta_.dims_; }
    inline TensorStride &stride() { return this->tensor_meta_.strides_; }
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

    const TensorMeta &meta() const;
    void set_meta(const TensorMeta &meta);
    TensorBuffer *buffer() const;
    void set_buffer(TensorBuffer *buf);

private:
    TensorMeta tensor_meta_;
    TensorBuffer *tensor_buffer_;
};

} // namespace nnops

#endif // __TENSOR_H__