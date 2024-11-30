#include <nnops/data_type.h>
#include <unordered_map>
#include <array>

namespace nnops {

template<DataType dtype> struct datatype_to_type;

#define DATATYPE_TO_TYPE_ITEM(dtype, type) template<> \
    struct datatype_to_type<dtype> { using Type = type; };

DATATYPE_GEN_TEMPLATE(DATATYPE_TO_TYPE_ITEM)

template<DataType dtype>
constexpr size_t sizeof_dtype() {
    using T = typename datatype_to_type<dtype>::Type;
    return sizeof(T);
}

#define DTYPE_SIZE_ITEM(dtype) { dtype, sizeof_dtype<dtype>() }

static std::unordered_map<DataType, size_t> dtype_size = {
    DTYPE_SIZE_ITEM(DataType::TYPE_FLOAT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT16),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT16),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT8),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT8),
};

size_t sizeof_dtype(DataType dtype) {
    return dtype_size[dtype];
}

constexpr auto u1 = DataType::TYPE_UINT8;
constexpr auto i1 = DataType::TYPE_INT8;
constexpr auto u2 = DataType::TYPE_UINT16;
constexpr auto i2 = DataType::TYPE_INT16;
constexpr auto u4 = DataType::TYPE_UINT32;
constexpr auto i4 = DataType::TYPE_INT32;
constexpr auto f4 = DataType::TYPE_FLOAT32;

constexpr std::array<DataType, DataType::COMPILE_TIME_MAX_DATA_TYPES> index2dtype = {
    u1, i2, u2, i2, u4, i4, f4,
};

template<typename FromType, typename ToType>
void type_cast(void *src, void *dst) {
    *((ToType *)dst) = *((FromType *)src);
}

#define DATATYPE_GEN_LOOPx1(GEN, type1)     \
    GEN(type1, float)                       \
    GEN(type1, int32_t)                     \
    GEN(type1, uint32_t)                    \
    GEN(type1, int16_t)                     \
    GEN(type1, uint16_t)                    \
    GEN(type1, int8_t)                      \
    GEN(type1, uint8_t)

#define DATATYPE_GEN_LOOPx2(GEN)            \
    DATATYPE_GEN_LOOPx1(GEN, float)         \
    DATATYPE_GEN_LOOPx1(GEN, int32_t)       \
    DATATYPE_GEN_LOOPx1(GEN, uint32_t)      \
    DATATYPE_GEN_LOOPx1(GEN, int16_t)       \
    DATATYPE_GEN_LOOPx1(GEN, uint16_t)      \
    DATATYPE_GEN_LOOPx1(GEN, int8_t)        \
    DATATYPE_GEN_LOOPx1(GEN, uint8_t)

#define GEN_ITEM(type1, type2) template void type_cast<type1, type2>(void *src, void *dst);
DATATYPE_GEN_LOOPx2(GEN_ITEM)

#define GEN_ITEM_INST(type1, type2) &type_cast<type1, type2>,
static std::array<
    std::array<std::function<void(void *, void *)>, index2dtype.size()>,
    index2dtype.size()> __dtype_cast_ops__ = { DATATYPE_GEN_LOOPx2(GEN_ITEM_INST) };

constexpr std::array<int, index2dtype.size()> calculate_dtype2index() {
  std::array<int, index2dtype.size()> dtype2index = {};
  for (int i = 0; i < dtype2index.size(); i++) {
    dtype2index[i] = -1;
  }
  for (int i = 0; i < index2dtype.size(); i++) {
    dtype2index[index2dtype[i]] = i;
  }
  return dtype2index;
}

constexpr auto dtype2index = calculate_dtype2index();

std::function<void(void *, void *)> get_cast_op(DataType from, DataType to) {
    auto from_idx = dtype2index[from], to_idx = dtype2index[to];
    return __dtype_cast_ops__[from][to];
}

} // namespace nnops
