#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>
#include <nnops/data_type.h>
#include <string>

using namespace std;

class TensorMeta {
public:
    TensorMeta(): nbytes_(0), nelems_(0), offset_(0) {}

    size_t nbytes_;
    size_t nelems_;
    int offset_;
    vector<int> dims_;
    vector<int> strides_;
    DataType dtype_;
};

#endif // __TENSOR_SHAPE_H__