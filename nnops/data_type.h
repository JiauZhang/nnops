#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nnops {

enum DataType: uint8_t {
    TYPE_UINT8 = 0,
    TYPE_INT8,
    TYPE_UINT16,
    TYPE_INT16,
    TYPE_UINT32,
    TYPE_INT32,
    TYPE_UINT64,
    TYPE_INT64,
    TYPE_FLOAT32,
    TYPE_FLOAT64,
    COMPILE_TIME_MAX_DATA_TYPES,
};

#define DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, args...)      \
    GEN(DataType::TYPE_UINT8, uint8_t, ##args)          \
    GEN(DataType::TYPE_INT8, int8_t, ##args)            \
    GEN(DataType::TYPE_UINT16, uint16_t, ##args)        \
    GEN(DataType::TYPE_INT16, int16_t, ##args)          \
    GEN(DataType::TYPE_UINT32, uint32_t, ##args)        \
    GEN(DataType::TYPE_INT32, int32_t, ##args)          \
    GEN(DataType::TYPE_UINT64, uint64_t, ##args)        \
    GEN(DataType::TYPE_INT64, int64_t, ##args)          \
    GEN(DataType::TYPE_FLOAT32, float, ##args)          \
    GEN(DataType::TYPE_FLOAT64, double, ##args)

#define DATATYPE_GEN_TEMPLATE_LOOPx2(GEN)            \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, uint8_t)       \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, int8_t)        \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, uint16_t)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, int16_t)       \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, uint32_t)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, int32_t)       \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, uint64_t)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, int64_t)       \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, float)         \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, double)

size_t sizeof_dtype(DataType dtype);
std::function<void(void *, void *)> get_cast_op(DataType from, DataType to);
DataType get_promote_type(DataType ltype, DataType rtype);

} // namespace nnops

#endif // __DATA_TYPE_H__