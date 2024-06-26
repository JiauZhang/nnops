#include <nnops/tensor_shape.h>

TensorShape::TensorShape(const TensorShape &shape) { 
    dims_ = shape.dims_;
}

TensorShape::TensorShape(vector<int> &dims) {
    dims_ = dims;
}

int TensorShape::ndim() {
    return dims_.size();
}

vector<int> &TensorShape::get_dims() {
    return dims_;
}

void TensorShape::set_dims(TensorShape &shape) {
    dims_ = shape.dims_;
}

void TensorShape::set_dims(vector<int> &dims) {
    dims_ = dims;
}