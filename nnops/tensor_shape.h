#ifndef __TENSOR_SHAPE_H__
#define __TENSOR_SHAPE_H__

#include <vector>

using namespace std;

class TensorShape {
public:
    TensorShape() {}
    TensorShape(const TensorShape &shape);
    TensorShape(vector<int> &dims);

    int ndim();
    vector<int> &get_dims();
    void set_dims(TensorShape &shape);
    void set_dims(vector<int> &dims);

private:
    vector<int> dims_;
};

#endif // __TENSOR_SHAPE_H__