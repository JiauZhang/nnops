#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nnops {

enum DataType: uint8_t {
    TYPE_FLOAT32 = 0,
    TYPE_INT32,
    TYPE_UINT32,
    TYPE_INT16,
    TYPE_UINT16,
    TYPE_INT8,
    TYPE_UINT8,
    COMPILE_TIME_MAX_DATA_TYPES,
};

#define DATATYPE_GEN_TEMPLATE(GEN)              \
    GEN(DataType::TYPE_FLOAT32, float)          \
    GEN(DataType::TYPE_INT32, int32_t)          \
    GEN(DataType::TYPE_UINT32, uint32_t)        \
    GEN(DataType::TYPE_INT16, int16_t)          \
    GEN(DataType::TYPE_UINT16, uint16_t)        \
    GEN(DataType::TYPE_INT8, int8_t)            \
    GEN(DataType::TYPE_UINT8, uint8_t)

size_t sizeof_dtype(DataType dtype);
std::function<void(void *, void *)> get_cast_op(DataType from, DataType to);

} // namespace nnops

#endif // __DATA_TYPE_H__