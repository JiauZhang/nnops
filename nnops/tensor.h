#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <string>

class Tensor {
public:
    Tensor(DataType &dtype, std::vector<int> &dims, std::string &device);
    Tensor(Tensor &other);
    ~Tensor();

    void reshape(std::vector<int> &dims);
    template<DataType dtype>
    auto at(std::vector<int> &dims) {
        if (dims.size() != meta_.ndim())
            throw std::runtime_error("dims must be same!");

        auto *ptr = reinterpret_dtype<dtype>(tensor_buffer_->data_ptr_);
        return *ptr;
    }

    TensorMeta meta_;
    TensorBuffer *tensor_buffer_;

private:
    Tensor() {}
    void alloc_buffer(TensorMeta &meta);
};

#endif // __TENSOR_H__