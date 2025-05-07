#ifndef __SCALAR_H__
#define __SCALAR_H__

#include <nnops/data_type.h>
#include <nnops/tensor.h>

namespace nnops {

#define GEN_SCALAR_DATA_ITEM(dtype, type) type d_##type;
union ScalarData {
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_SCALAR_DATA_ITEM)
};
#undef GEN_SCALAR_DATA_ITEM

class Scalar {
public:
    Scalar();
#define GEN_SCALAR_CONSTRUCTOR(dtype, type) Scalar(type data);
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_SCALAR_CONSTRUCTOR)
#undef GEN_SCALAR_CONSTRUCTOR
    Scalar(const Scalar &other);
    ~Scalar();

    inline DataType dtype() const { return dtype_; }
    inline ScalarData data() const { return data_; }
    inline void *data_ptr() const { return (void *)(&data_); }
    inline void set_dtype(DataType dtype) { dtype_ = dtype; }
    Scalar astype(DataType dtype);
    inline index_t itemsize() const { return sizeof_dtype(this->dtype_); }
    Tensor tensor() const;

private:
    DataType dtype_;
    ScalarData data_;
};

} // namespace nnops

#endif // __SCALAR_H__