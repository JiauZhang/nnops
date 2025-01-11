#ifndef __TENSOR_ITERATOR_H__
#define __TENSOR_ITERATOR_H__

#include <nnops/tensor.h>

namespace nnops {

class TensorIterator {
public:
    TensorIterator(const Tensor &tensor);

    TensorIterator &operator++();
    void *operator*() { return (void *)((char *)(tensor_->data_ptr()) + offset_ * tensor_->itemsize()); }

    inline void end() { offset_ = -1; }
    inline bool is_end() { return offset_ == -1; }

protected:
    const Tensor *tensor_;
    TensorShape index_;
    index_t offset_;
};

class TensorPartialIterator : public TensorIterator {
public:
    TensorPartialIterator(const Tensor &tensor, index_t start, index_t stop);

    TensorPartialIterator &operator++();
    Tensor tensor();

private:
    index_t start_, stop_;
};

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__