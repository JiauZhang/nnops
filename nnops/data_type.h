#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <nnops/common.h>
#include <tuple>

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
using CppType = std::tuple<
    bool,
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    uint64_t,
    int64_t,
    float,
    double
>;
constexpr size_t kNumDataTypes = std::tuple_size<CppType>::value;

template <size_t index>
using CppTypeElement = typename std::tuple_element<index, CppType>::type;

template <size_t index>
struct DataTypeElementImpl : std::integral_constant<DataType, static_cast<DataType>(index)> {
    static_assert(index < kNumDataTypes, "Index out of range for DataType");
};

template <size_t index>
inline constexpr DataType DataTypeElement = DataTypeElementImpl<index>::value;

template <DataType T>
struct DataTypeToCppTypeImpl;

template <DataType T>
struct DataTypeToCppTypeImpl {
    static_assert(static_cast<uint8_t>(T) < std::tuple_size_v<CppType>, "Invalid DataType value");
    using type = CppTypeElement<static_cast<std::size_t>(T)>;
};

template <DataType T>
using DataTypeToCppType = typename DataTypeToCppTypeImpl<T>::type;

enum ScalarBinaryOpType : uint8_t {
    ADD = 0,
    SUB,
    MUL,
    DIV,
    COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES,
};

#define DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, ...)      \
    GEN(DataType::TYPE_BOOL, bool, ##__VA_ARGS__)              \
    GEN(DataType::TYPE_UINT8, uint8_t, ##__VA_ARGS__)          \
    GEN(DataType::TYPE_INT8, int8_t, ##__VA_ARGS__)            \
    GEN(DataType::TYPE_UINT16, uint16_t, ##__VA_ARGS__)        \
    GEN(DataType::TYPE_INT16, int16_t, ##__VA_ARGS__)          \
    GEN(DataType::TYPE_UINT32, uint32_t, ##__VA_ARGS__)        \
    GEN(DataType::TYPE_INT32, int32_t, ##__VA_ARGS__)          \
    GEN(DataType::TYPE_UINT64, uint64_t, ##__VA_ARGS__)        \
    GEN(DataType::TYPE_INT64, int64_t, ##__VA_ARGS__)          \
    GEN(DataType::TYPE_FLOAT32, float, ##__VA_ARGS__)          \
    GEN(DataType::TYPE_FLOAT64, double, ##__VA_ARGS__)

#define DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, ...)                                \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_BOOL, bool, ##__VA_ARGS__)          \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT8, uint8_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT8, int8_t, ##__VA_ARGS__)        \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT16, uint16_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT16, int16_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT32, uint32_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT32, int32_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_UINT64, uint64_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_INT64, int64_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_FLOAT32, float, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN, DataType::TYPE_FLOAT64, double, ##__VA_ARGS__)

#define DATATYPE_GEN_TEMPLATE_LOOPx3(GEN, ...)                                \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_BOOL, bool, ##__VA_ARGS__)          \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT8, uint8_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT8, int8_t, ##__VA_ARGS__)        \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT16, uint16_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT16, int16_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT32, uint32_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT32, int32_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_UINT64, uint64_t, ##__VA_ARGS__)    \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_INT64, int64_t, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_FLOAT32, float, ##__VA_ARGS__)      \
    DATATYPE_GEN_TEMPLATE_LOOPx2(GEN, DataType::TYPE_FLOAT64, double, ##__VA_ARGS__)

#define SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(GEN, ...)   \
    GEN(ScalarBinaryOpType::ADD, add, +, ##__VA_ARGS__)             \
    GEN(ScalarBinaryOpType::SUB, sub, -, ##__VA_ARGS__)             \
    GEN(ScalarBinaryOpType::MUL, mul, *, ##__VA_ARGS__)             \
    GEN(ScalarBinaryOpType::DIV, truediv, /, ##__VA_ARGS__)

size_t sizeof_dtype(DataType dtype);

using dtype_cast_op_t = void (*)(void **args, const index_t *strides, const index_t size);
dtype_cast_op_t get_cast_op(DataType from, DataType to);

using scalar_binary_op_t = void (*)(void **args, const index_t *strides, const index_t size);
DataType get_promote_type(ScalarBinaryOpType op_type, DataType ltype, DataType rtype);
scalar_binary_op_t get_scalar_binary_op(ScalarBinaryOpType op_type, DataType dtype);

} // namespace nnops

#endif // __DATA_TYPE_H__