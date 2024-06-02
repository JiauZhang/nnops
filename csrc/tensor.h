#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "tensor_shape.h"
#include "data_type.h"

using namespace std;
namespace nb = nanobind;

class Tensor {
public:
    Tensor() {}
    Tensor(DataType &dtype, TensorShape &shape);
    Tensor(DataType &dtype, vector<int> &dims);

    void reshape(vector<int> &dims);
    void reshape(TensorShape &shape);

    TensorShape shape_;
    DataType dtype_;
};

void DEFINE_TENSOR_MODULE(nb::module_ & (m));

#endif // __TENSOR_H__