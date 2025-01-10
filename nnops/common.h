#ifndef __NNOPS_COMMON_H__
#define __NNOPS_COMMON_H__

#include <vector>

namespace nnops {

using index_t = int;
using TensorShape = std::vector<index_t>;
using TensorStride = std::vector<index_t>;

} // namespace nnops

#endif // __NNOPS_COMMON_H__