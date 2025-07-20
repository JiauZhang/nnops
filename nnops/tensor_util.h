#ifndef __TENSOR_UTIL_H__
#define __TENSOR_UTIL_H__

#include <nnops/tensor.h>
#include <nnops/tensor_iterator.h>

namespace nnops {

inline Tensor tensor_from(const TensorIterator &iter) { return Tensor(iter.meta(), iter.buffer()); }

} // namespace nnops

#endif // __TENSOR_UTIL_H__