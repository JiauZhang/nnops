#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <nnops/device.h>
#include <string>

namespace nnops {

class TensorIterator;
class TensorPartialIterator;
class TensorAccessor;

class Tensor {
public:
    Tensor();
    Tensor(DataType dtype, const TensorShape &dims, std::string &device);
    Tensor(DataType dtype, const TensorShape &dims, DeviceType device);
    Tensor(DataType dtype, const TensorShape &dims, Device *device);
    Tensor(const Tensor &other);
    Tensor(const TensorMeta &meta, TensorBuffer *buf);
    ~Tensor();

    inline const Tensor &operator*() { return *this; }
    Tensor &operator=(Tensor &other);
    void init_tensor(DataType &dtype, const TensorShape &dims, Device *device);

    inline void reshape_inplace(TensorShape &dims) { this->tensor_meta_.reshape_inplace(dims); }
    Tensor reshape(TensorShape &dims);
    inline static bool is_broadcastable(const Tensor &t1, const Tensor &t2) { return is_broadcastable(t1.shape(), t2.shape()); }
    static bool is_broadcastable(const TensorShape &s1, const TensorShape &s2);
    inline static TensorShape broadcast_shape(const Tensor &t1, const Tensor &t2) { return broadcast_shape(t1.shape(), t2.shape()); }
    static TensorShape broadcast_shape(const TensorShape &s1, const TensorShape &s2);
    bool is_broadcast();
    inline Tensor broadcast_to(const TensorShape &shape) { return Tensor::broadcast_to(*this, shape); }
    static Tensor broadcast_to(const Tensor &t, const TensorShape &shape);
    Tensor permute(TensorShape &index);

    inline DataType dtype() const { return this->tensor_meta_.dtype_; }
    inline void *data_ptr() const { return (void *)((char *)this->tensor_buffer_->data_ptr_ + this->offset() * this->itemsize()); }
    inline int ndim() const { return this->shape().size(); }
    inline int ref_count() { return this->tensor_buffer_->count(); }
    inline Device *device() { return this->tensor_buffer_->device_; }
    inline size_t nelems() { return this->tensor_meta_.nelems_; }
    inline size_t nbytes() { return this->tensor_meta_.nbytes_; }
    inline index_t itemsize() const { return sizeof_dtype(this->dtype()); }
    inline index_t offset() const { return this->tensor_meta_.offset_; }
    inline index_t &mutable_offset() { return this->tensor_meta_.offset_; }
    inline bool is_contiguous() {return this->tensor_meta_.is_contiguous(); }
    Tensor clone();
    Tensor contiguous();
    Tensor astype(DataType dtype);
    Tensor to(DeviceType device);

    void to_string(std::string *prefix, std::string *ret);
    std::string to_string();
    std::string to_repr();

    inline const TensorShape &shape() const { return this->tensor_meta_.dims_; }
    inline void set_shape(const TensorShape &shape) { this->tensor_meta_.dims_ = shape; }
    inline const TensorStride &stride() const { return this->tensor_meta_.strides_; }
    inline void set_stride(const TensorStride &stride) { this->tensor_meta_.strides_ = stride; }
    const TensorMeta &meta() const;
    void set_meta(const TensorMeta &meta);
    TensorBuffer *buffer() const;
    void set_buffer(TensorBuffer *buf);

    static TensorShape unravel_index(index_t idx, const TensorShape &shape);
    inline TensorShape unravel_index(index_t idx) const { return unravel_index(idx, this->shape()); }
    static index_t ravel_index(const TensorShape &dims, const TensorShape &shape);
    inline index_t ravel_index(const TensorShape &dims) const { return ravel_index(dims, this->shape()); }

private:
    TensorMeta tensor_meta_;
    TensorBuffer *tensor_buffer_;
};

} // namespace nnops

#endif // __TENSOR_H__