#ifndef __TENSOR_ITERATOR_H__
#define __TENSOR_ITERATOR_H__

#include <nnops/tensor.h>

namespace nnops {

class Tensor;

class TensorIterator {
public:
    TensorIterator(const Tensor &tensor);

    TensorIterator &operator++();
    void *operator*();

    inline void end() { offset_ = -1; }
    inline bool is_end() { return offset_ == -1; }

private:
    const Tensor *tensor_;
    TensorShape index_;
    index_t offset_;
};

class TensorPartialIterator {
public:
    TensorPartialIterator(const Tensor &tensor, index_t start, index_t stop);

    TensorPartialIterator &operator++();
    void *operator*();
    Tensor tensor();

    inline void end() { offset_ = -1; }
    inline bool is_end() { return offset_ == -1; }

private:
    const Tensor *tensor_;
    TensorShape index_;
    index_t offset_, start_, stop_;
};

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__