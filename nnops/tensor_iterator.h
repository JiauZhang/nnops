#ifndef __TENSOR_ITERATOR_H__
#define __TENSOR_ITERATOR_H__

#include <nnops/tensor.h>

namespace nnops {

class TensorIterator {
public:
    TensorIterator(const Tensor &tensor);

private:
    TensorShape shape_, offset_;
    TensorStride stride_;
};

} // namespace nnops

#endif // __TENSOR_ITERATOR_H__