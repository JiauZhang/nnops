#ifndef __TENSOR_ITERATOR_H__
#define __TENSOR_ITERATOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/tensor_buffer.h>
#include <memory>

namespace nnops {

class TensorIterator {
public:
    TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer);
    TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer, index_t start, index_t stop);
    TensorIterator(const TensorMeta &tensor_meta, std::shared_ptr<TensorBuffer> buffer, index_t start, index_t stop, index_t offset);

    inline const TensorShape &shape() const { return this->tensor_meta_.dims_; }
    inline const TensorStride &stride() const { return this->tensor_meta_.strides_; }
    inline int ndim() const { return this->shape().size(); }
    inline void *data_ptr(index_t offset) const {
        return (void *)((char *)this->tensor_buffer_->data_ptr_ + (
            this->tensor_meta_.offset() + offset) * this->tensor_meta_.itemsize());
    }
    inline index_t offset() const { return this->offset_; }
    const TensorMeta &meta() const { return tensor_meta_; }
    std::shared_ptr<TensorBuffer> buffer() const { return this->tensor_buffer_; }
    inline DataType dtype() const { return this->tensor_meta_.dtype_; }

    TensorIterator &operator++();
    inline void *operator*() const { return (void *)((char *)this->data_ptr(offset_)); }

    inline void end() { offset_ = -1; }
    inline bool is_end() const { return offset_ == -1; }

protected:
    TensorMeta tensor_meta_;
    TensorShape index_;
    index_t offset_;
    std::shared_ptr<TensorBuffer> tensor_buffer_;
};

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__