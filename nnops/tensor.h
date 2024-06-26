#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_shape.h>
#include <nnops/data_type.h>
#include <nnops/device.h>
#include <nnops/tensor_buffer.h>

using namespace std;

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

#endif // __TENSOR_H__