#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>
#include <nnops/data_type.h>
#include <string>

namespace nnops {

class TensorMeta {
public:
    TensorMeta(): nbytes_(0), nelems_(0), offset_(0) {}

    static std::string shape_as_string(const std::vector<int> &dims);
    void reshape_inplace(std::vector<int> &dims);
    bool is_contiguous();

    size_t nbytes_;
    size_t nelems_;
    int offset_;
    std::vector<int> dims_;
    std::vector<int> strides_;
    DataType dtype_;
};

} // namespace nnops

#endif // __TENSOR_SHAPE_H__