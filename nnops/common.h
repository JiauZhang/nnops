#ifndef __NNOPS_COMMON_H__
#define __NNOPS_COMMON_H__

#include <vector>
#include <stdexcept>
#include <string>

namespace nnops {

using index_t = int;
using TensorShape = std::vector<index_t>;
using TensorStride = std::vector<index_t>;

#define NNOPS_CHECK(cond, info)                               \
    if (!(cond)) {                                            \
        const std::string detailed_info =                     \
            + #cond " CHECK FAILED at "                       \
            + std::string(__FILE__)                           \
            + "::" + std::string(__func__) + "::#L"           \
            + std::to_string(__LINE__) + '\n' + info;         \
        throw std::runtime_error(detailed_info);              \
    }

} // namespace nnops

#endif // __NNOPS_COMMON_H__