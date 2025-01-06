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

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__