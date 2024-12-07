#ifndef __SCALAR_H__
#define __SCALAR_H__

#include <nnops/data_type.h>

namespace nnops {

class Scalar {
public:
    Scalar(): data_(nullptr) {}
#define GEN_SCALAR_CONSTRUCTOR(dtype, type) Scalar(type data);
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_SCALAR_CONSTRUCTOR)
#undef GEN_SCALAR_CONSTRUCTOR
    Scalar(const Scalar &other);
    ~Scalar();

    DataType dtype() const { return dtype_; }
    void set_dtype(DataType type) { dtype_ = type; }
    void *buffer() const { return data_; }
    void set_buffer(void *data) { data_ = data; }
    Scalar astype(DataType dtype);

private:
    DataType dtype_;
    void *data_;
};

} // namespace nnops

#endif // __SCALAR_H__