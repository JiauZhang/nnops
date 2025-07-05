#ifndef __NNOPS_COMMON_H__
#define __NNOPS_COMMON_H__

#include <vector>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <memory>

namespace nnops {

using index_t = int;
using TensorShape = std::vector<index_t>;
using TensorStride = std::vector<index_t>;

#define NNOPS_CHECK(cond, fmt, ...)                                 \
do {                                                                \
    if (!(cond)) {                                                  \
        int size = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);   \
        auto buf = std::make_unique<char[]>(size + 1);              \
        std::snprintf(buf.get(), size + 1, fmt, ##__VA_ARGS__);     \
        throw std::runtime_error(                                   \
            std::string(#cond) + " CHECK FAILED at " +              \
            __FILE__ + "::" + __func__ + "::L" +                    \
            std::to_string(__LINE__) + "\n" + buf.get()             \
        );                                                          \
    }                                                               \
} while (0)

} // namespace nnops

#endif // __NNOPS_COMMON_H__