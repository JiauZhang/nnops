#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>
#include <nnops/data_type.h>
#include <nnops/device.h>
#include <string>

using namespace std;

class TensorMeta {
public:
    TensorMeta() {}

    int ndim();
    vector<int> &get_dims();
    void set_dims(vector<int> &dims);
    void reshape(vector<int> &dims);

    string shape_as_string(vector<int> &dims);

    size_t nbytes_;
    size_t nelems_;
    vector<int> dims_;
    vector<int> strides_;
    DataType dtype_;
    Device *device_;
};

#endif // __TENSOR_SHAPE_H__