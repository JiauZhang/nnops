#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <tensor_shape.h>
#include <data_type.h>
#include <device.h>
#include <nanobind/stl/string.h>
#include <tensor_buffer.h>

using namespace std;
namespace nb = nanobind;

class Tensor {
public:
    Tensor(DataType &dtype, TensorShape &shape, std::string &device);
    Tensor(DataType &dtype, vector<int> &dims, std::string &device);
    Tensor(Tensor &other);
    ~Tensor();

    void reshape(vector<int> &dims);
    void reshape(TensorShape &shape);

    TensorShape shape_;
    DataType dtype_;
    Device *device_;
    TensorBuffer *tensor_buffer_;
    size_t nbytes_;
    size_t nelems_;

private:
    Tensor() {}
    void init_tensor(DataType &dtype, TensorShape &shape, std::string &device);
};

void DEFINE_TENSOR_MODULE(nb::module_ & (m));

#endif // __TENSOR_H__