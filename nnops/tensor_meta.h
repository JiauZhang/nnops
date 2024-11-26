#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>
#include <nnops/data_type.h>
#include <string>

namespace nnops {

typedef std::vector<int> TensorShape;
typedef std::vector<int> TensorStride;

class TensorMeta {
public:
    TensorMeta(): nbytes_(0), nelems_(0), offset_(0) {}

    static std::string shape_as_string(const TensorShape &dims);
    inline std::string shape_as_string() { return TensorMeta::shape_as_string(this->dims_); };
    void reshape_inplace(TensorShape &dims);
    bool is_contiguous();

    size_t nbytes_;
    size_t nelems_;
    int offset_;
    TensorShape dims_;
    TensorStride strides_;
    DataType dtype_;
};

} // namespace nnops

#endif // __TENSOR_SHAPE_H__