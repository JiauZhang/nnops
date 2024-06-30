#include <nnops/tensor_meta.h>

int TensorMeta::ndim() {
    return dims_.size();
}

vector<int> &TensorMeta::get_dims() {
    return dims_;
}

void TensorMeta::set_dims(vector<int> &dims) {
    dims_ = dims;
}

void TensorMeta::reshape(vector<int> &dims) {
    dims_ = dims;
}