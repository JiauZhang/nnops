#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>
#include <nnops/data_type.h>
#include <optional>
#include <string>

namespace nnops {

typedef std::vector<int> TensorShape;
typedef std::vector<int> TensorStride;

class Slice {
public:
    Slice() {}
    Slice(int start): start_(start) {}
    Slice(int start, int stop): start_(start), stop_(stop) {}
    Slice(int start, int stop, int step): start_(start), stop_(stop), step_(step) {}

    std::optional<int> start_, stop_, step_;
};

class TensorMeta {
public:
    TensorMeta(): nbytes_(0), nelems_(0), offset_(0) {}

    static std::string shape_as_string(const TensorShape &dims);
    inline std::string shape_as_string() { return TensorMeta::shape_as_string(this->dims_); };
    bool is_contiguous();

    void reshape_inplace(TensorShape &dims);
    void index_inplace(int index, int axis);
    void slice_inplace(Slice &slice, int axis);

    inline TensorShape &shape() { return this->dims_; }
    inline TensorStride &stride() { return this->strides_; }
    inline int ndim() { return this->shape().size(); }

    size_t nbytes_;
    size_t nelems_;
    int offset_;
    TensorShape dims_;
    TensorStride strides_;
    DataType dtype_;
};

} // namespace nnops

#endif // __TENSOR_SHAPE_H__