#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>

using namespace std;

class Tensor {
public:
    Tensor(DataType &dtype, vector<int> &dims, std::string &device);
    Tensor(Tensor &other);
    ~Tensor();

    void reshape(vector<int> &dims);

    TensorMeta meta_;
    TensorBuffer *tensor_buffer_;

private:
    Tensor() {}
    void alloc_buffer(TensorMeta &meta);
};

#endif // __TENSOR_H__