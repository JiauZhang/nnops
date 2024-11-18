#ifndef __OPS_MATH_H__
#define __OPS_MATH_H__

#include <nnops/tensor.h>

namespace nnops::cpu::ops {

nnops::Tensor add(nnops::Tensor &self, nnops::Tensor &other);

} // namespace nnops::cpu::ops

#endif // __OPS_MATH_H__