#include <nnops/scalar.h>

namespace nnops {

#define GEN_SCALAR_CONSTRUCTOR(dtype, type) Scalar::Scalar(type data) { \
    data_ = std::malloc(sizeof(type));                                  \
    *reinterpret_cast<type *>(data_) = data;                            \
    dtype_ = dtype;                                                     \
}

DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_SCALAR_CONSTRUCTOR)

Scalar::Scalar(const Scalar &other) {
    dtype_ = other.dtype();
    data_ = std::malloc(sizeof_dtype(dtype_));
    auto copy_op = get_cast_op(dtype_, dtype_);
    copy_op(other.buffer(), data_);
}

Scalar::~Scalar() {
    if (data_)
        std::free(data_);
}

Scalar Scalar::astype(DataType dtype) {
    Scalar scalar;
    scalar.set_dtype(dtype);
    scalar.set_buffer(std::malloc(sizeof_dtype(dtype)));
    auto cast_op = get_cast_op(dtype_, scalar.dtype());
    cast_op(data_, scalar.buffer());
    return scalar;
}

} // namespace nnops