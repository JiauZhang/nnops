#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>

enum DataType {
    TYPE_FLOAT32,
    TYPE_INT32,
    TYPE_UINT32,
    TYPE_INT16,
    TYPE_UINT16,
    TYPE_INT8,
    TYPE_UINT8,
};

#define DATATYPE_GEN_TEMPLATE(GEN)              \
    GEN(DataType::TYPE_FLOAT32, float)          \
    GEN(DataType::TYPE_INT32, int)              \
    GEN(DataType::TYPE_UINT32, unsigned int)    \
    GEN(DataType::TYPE_INT16, short)            \
    GEN(DataType::TYPE_UINT16, unsigned short)  \
    GEN(DataType::TYPE_INT8, char)              \
    GEN(DataType::TYPE_UINT8, unsigned char)

template<DataType dtype> struct datatype_to_type;

#define DATATYPE_TO_TYPE_ITEM(dtype, type) template<> \
    struct datatype_to_type<dtype> { using Type = type; };

DATATYPE_GEN_TEMPLATE(DATATYPE_TO_TYPE_ITEM)

template<DataType dtype>
auto *reinterpret_dtype(void *ptr) {
    using T = typename datatype_to_type<dtype>::Type;
    return (T *)ptr;
}

template<DataType dtype>
size_t sizeof_dtype() {
    using T = typename datatype_to_type<dtype>::Type;
    return sizeof(T);
}

size_t sizeof_dtype(DataType dtype);

#endif // __DATA_TYPE_H__