#include <nnops/scalar.h>

namespace nnops {

#define GEN_SCALAR_CONSTRUCTOR(dtype, type) Scalar::Scalar(type data) { \
    data_.d_##type = data;                                  \
    dtype_ = dtype;                                                     \
}

DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_SCALAR_CONSTRUCTOR)

Scalar::Scalar(const Scalar &other) {
    dtype_ = other.dtype();
    data_ = other.data();
}

Scalar::Scalar() {}
Scalar::~Scalar() {}

Scalar Scalar::astype(DataType dtype) {
    Scalar scalar;
    scalar.set_dtype(dtype);
    auto cast_op = get_cast_op(dtype_, scalar.dtype());
    void *args[2] = {this->data_ptr(), scalar.data_ptr()};
    const index_t strides[2] = {this->itemsize(), scalar.itemsize()};
    cast_op(args, strides, 1);
    return scalar;
}

} // namespace nnops