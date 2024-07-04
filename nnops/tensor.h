#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <string>

class Tensor {
public:
    Tensor(DataType &dtype, std::vector<int> &dims, std::string &device);
    Tensor(const Tensor &other);
    Tensor(const Tensor &other, std::vector<int> &dims);
    ~Tensor();

    void reshape(std::vector<int> &dims);
    std::string to_string();

    Tensor operator[](std::vector<int> &dims);

    TensorMeta meta_;
    TensorBuffer *tensor_buffer_;

private:
    Tensor() {}
    void alloc_buffer(TensorMeta &meta);
};

#endif // __TENSOR_H__