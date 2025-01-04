#ifndef __TENSOR_ACCESSOR_H__
#define __TENSOR_ACCESSOR_H__

#include <nnops/tensor.h>

namespace nnops {

class Tensor;

class TensorAccessor {
public:
    TensorAccessor(const Tensor &tensor);
    void *data_ptr_unsafe(const TensorShape &dims);
    void *data_ptr_unsafe(const TensorShape &anchor, index_t offset, index_t dim);
    void *data_ptr_unsafe(const void *anchor_ptr, index_t offset, index_t dim);

private:
    const Tensor *tensor_;
};

} // namespace nnops

#endif // __TENSOR_ACCESSOR_H__