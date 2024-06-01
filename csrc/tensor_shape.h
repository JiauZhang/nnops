#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

using namespace std;
namespace nb = nanobind;

class TensorShape {
public:
    TensorShape() {}
    TensorShape(TensorShape &shape) { dims_ = shape.dims_; }
    TensorShape(vector<int> &dims) { dims_ = dims; }

    int ndim() { return dims_.size(); }
    vector<int> get_dims() { return dims_; }
    void set_dims(TensorShape &shape) { dims_ = shape.dims_; }
    void set_dims(vector<int> &dims) { dims_ = dims; }

private:
    vector<int> dims_;
};

void DEFINE_TENSOR_SHAPE_MODULE(nb::module_ & (m)) {
    nb::class_<TensorShape>(m, "TensorShape")
        .def(nb::init<>())
        .def(nb::init<TensorShape &>())
        .def(nb::init<vector<int> &>())
        .def("get_dims", &TensorShape::get_dims)
        .def("set_dims", [](TensorShape &self, vector<int> &dims) { self.set_dims(dims); })
        .def("set_dims", [](TensorShape &self, TensorShape &shape) { self.set_dims(shape); })
        .def_prop_ro("ndim", [](TensorShape &shape) { return shape.ndim(); });
}

#endif // __TENSOR_SHAPE_H__