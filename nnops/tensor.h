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
    Tensor(DataType dtype, const TensorShape &dims, std::string &device);
    Tensor(DataType dtype, const TensorShape &dims, DeviceType device);
    Tensor(DataType dtype, const TensorShape &dims, Device *device);
    Tensor(const Tensor &other);
    Tensor(const TensorMeta &meta, TensorBuffer *buf);
    ~Tensor();

    inline const Tensor &operator*() { return *this; }
    Tensor &operator=(Tensor &other);
    void init_tensor(DataType &dtype, const TensorShape &dims, Device *device);

    static void fill(Tensor &self, const Tensor &value);
    inline void fill(const Tensor &value) { fill(*this, value); }

    inline void reshape_inplace(TensorShape &dims) { this->tensor_meta_.reshape_inplace(dims); }
    Tensor reshape(TensorShape &dims) const;
    inline static bool is_broadcastable(const Tensor &t1, const Tensor &t2) { return is_broadcastable(t1.shape(), t2.shape(), 0); }
    static bool is_broadcastable(const TensorShape &s1, const TensorShape &s2, int offset);
    inline static TensorShape broadcast_shape(const Tensor &t1, const Tensor &t2) { return broadcast_shape(t1.shape(), t2.shape(), 0); }
    static TensorShape broadcast_shape(const TensorShape &s1, const TensorShape &s2, int offset);
    bool is_broadcast();
    static bool is_broadcastable_to(const TensorShape &self, const TensorShape &other, int offset);
    inline bool is_broadcastable_to(const TensorShape &shape) { return is_broadcastable_to(this->shape(), shape, 0); }
    inline Tensor broadcast_to(const TensorShape &shape) const { return Tensor::broadcast_to(*this, shape, 0); }
    static Tensor broadcast_to(const Tensor &t, const TensorShape &shape, int offset);

    Tensor permute(TensorShape &index) const;
    static Tensor transpose(const Tensor &t, index_t dim0, index_t dim1);
    inline Tensor transpose(index_t dim0, index_t dim1) const { return transpose(*this, dim0, dim1); }

    inline DataType dtype() const { return this->tensor_meta_.dtype_; }
    inline void *data_ptr() const { return data_ptr(0); }
    inline void *data_ptr(index_t offset) const {
        return (void *)((char *)this->tensor_buffer_->data_ptr_ + (this->offset() + offset) * this->itemsize());
    }
    inline int ndim() const { return this->shape().size(); }
    inline int ref_count() { return this->tensor_buffer_->count(); }
    inline Device *device() const { return this->tensor_buffer_->device_; }
    inline size_t nelems() const { return this->tensor_meta_.nelems_; }
    inline size_t nbytes() { return this->tensor_meta_.nbytes(); }
    inline index_t itemsize() const { return sizeof_dtype(this->dtype()); }
    inline index_t offset() const { return this->tensor_meta_.offset_; }
    inline bool is_contiguous() const {return this->tensor_meta_.is_contiguous(); }
    Tensor clone() const;
    Tensor contiguous() const;
    Tensor astype(DataType dtype) const;
    Tensor to(DeviceType device) const;

    void to_string(std::string *prefix, std::string *ret) const;
    std::string to_string() const;
    std::string to_repr() const;

    inline std::string shape_as_string() const { return this->tensor_meta_.shape_as_string(); }
    inline const TensorShape &shape() const { return this->tensor_meta_.dims_; }
    index_t shape(int index) const;
    inline void set_shape(const TensorShape &shape) { this->tensor_meta_.dims_ = shape; }
    inline const TensorStride &stride() const { return this->tensor_meta_.strides_; }
    index_t stride(int index) const;
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

void tensor_clone_impl(const Tensor *src, int src_offset, Tensor *dst, int dst_offset, int axis);

} // namespace nnops

#endif // __TENSOR_H__