#ifndef __TENSOR_UTIL_H__
#define __TENSOR_UTIL_H__

#include <nnops/tensor.h>
#include <nnops/tensor_iterator.h>

namespace nnops {

Tensor tensor_from(const TensorPartialIterator &iter);

} // namespace nnops

#endif // __TENSOR_UTIL_H__