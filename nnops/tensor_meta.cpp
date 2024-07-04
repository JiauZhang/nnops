#include <nnops/tensor_meta.h>
#include <string>

using namespace std;

void TensorMeta::reshape(vector<int> &dims) {
    int nelems = 1;
    for (auto dim: dims)
        nelems *= dim;

    if (nelems != nelems_) {
        string info = "Can't reshape tensor from shape "
            + shape_as_string(dims_) + " to " + shape_as_string(dims);
        throw std::runtime_error(info);
    }

    dims_ = dims;
}

string TensorMeta::shape_as_string(vector<int> &dims) {
    string shape_str;

    shape_str += '[';
    for (auto dim: dims)
        shape_str += to_string(dim) + ", ";
    auto len = shape_str.size();
    shape_str.resize(len-1);
    shape_str[len-2] = ']';

    return shape_str;
}