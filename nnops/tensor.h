#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <nnops/tensor_meta.h>
#include <nnops/data_type.h>
#include <nnops/tensor_buffer.h>
#include <string>

class Tensor {
public:
    Tensor() {}
    Tensor(DataType &dtype, std::vector<int> &dims, std::string &device);
    Tensor(const Tensor &other);
    Tensor(const Tensor &other, const std::vector<int> &dims);
    ~Tensor();

    Tensor reshape(std::vector<int> &dims);
    string shape_as_string(const vector<int> &dims);

    DataType dtype() { return this->tensor_meta_.dtype_; }
    const std::vector<int> &shape() { return this->tensor_meta_.dims_; }
    const std::vector<int> &stride() { return this->tensor_meta_.strides_; }
    void *data_ptr() { return this->tensor_buffer_->data_ptr_; }
    int ndim() { return this->shape().size(); }
    int ref_count() { return this->tensor_buffer_->count(); }
    size_t nelems() { return this->tensor_meta_.nelems_; }
    size_t nbytes() { return this->tensor_meta_.nbytes_; }

    void to_string(std::string *prefix, std::string *ret);
    std::string to_string();
    std::string to_repr();

    Tensor operator[](std::vector<int> &dims);

    TensorMeta tensor_meta_;
    TensorBuffer *tensor_buffer_;

private:
    void alloc_buffer(TensorMeta &meta);
};

#endif // __TENSOR_H__