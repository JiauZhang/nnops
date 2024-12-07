#ifndef __TENSOR_ITERATOR_H__
#define __TENSOR_ITERATOR_H__

#include <nnops/tensor.h>

namespace nnops {

class Tensor;

class TensorIterator {
public:
    TensorIterator(const Tensor &tensor);

    TensorIterator &operator++();
    bool operator!=(const TensorIterator &other);
    void *operator*();

    inline int offset() const { return offset_; }
    inline const TensorShape &index() const { return index_; }
    inline const Tensor &tensor() const { return tensor_; }
    inline void set_offset(int offset) { offset_ = offset; }
    inline void end() { offset_ = -1; }

private:
    const Tensor &tensor_;
    TensorShape index_;
    int offset_;
};

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__