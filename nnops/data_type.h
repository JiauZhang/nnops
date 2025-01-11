#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <nnops/common.h>

namespace nnops {

enum DataType : uint8_t {
    TYPE_BOOL = 0,
    TYPE_UINT8,
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

enum ScalarBinaryOpType : uint8_t {
    ADD = 0,
    SUB,
    MUL,
    DIV,
    COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES,
};

#define DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, args...)      \
    GEN(DataType::TYPE_BOOL, bool, ##args)              \
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

#define DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, args...)                                \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_BOOL, bool, ##args)          \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT8, uint8_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT8, int8_t, ##args)        \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT16, uint16_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT16, int16_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT32, uint32_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT32, int32_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT64, uint64_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT64, int64_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_FLOAT32, float, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_FLOAT64, double, ##args)

#define DATATYPE_GEN_TEMPLATE_LOOPx3(GEN, args...)                                \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_BOOL, bool, ##args)          \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT8, uint8_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT8, int8_t, ##args)        \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT16, uint16_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT16, int16_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT32, uint32_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT32, int32_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT64, uint64_t, ##args)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT64, int64_t, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_FLOAT32, float, ##args)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_FLOAT64, double, ##args)

#define SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(GEN, args...)   \
    GEN(ScalarBinaryOpType::ADD, add, +, ##args)             \
    GEN(ScalarBinaryOpType::SUB, sub, -, ##args)             \
    GEN(ScalarBinaryOpType::MUL, mul, *, ##args)             \
    GEN(ScalarBinaryOpType::DIV, div, /, ##args)

size_t sizeof_dtype(DataType dtype);

using dtype_cast_op_t = void (*)(void **args, const index_t *strides, const index_t size);
dtype_cast_op_t get_cast_op(DataType from, DataType to);

using scalar_binary_op_t = void (*)(void **args, const index_t *strides, const index_t size);
DataType get_promote_type(ScalarBinaryOpType op_type, DataType ltype, DataType rtype);
scalar_binary_op_t get_scalar_binary_op(ScalarBinaryOpType op_type, DataType ltype, DataType rtype);

} // namespace nnops

#endif // __DATA_TYPE_H__