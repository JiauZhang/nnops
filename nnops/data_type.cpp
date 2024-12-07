#include <nnops/data_type.h>
#include <unordered_map>
#include <array>

namespace nnops {

#define GEN_DTYPE_SIZE(dtype, type) sizeof(type),
static constexpr std::array<size_t, DataType::COMPILE_TIME_MAX_DATA_TYPES> __dtype_size__ = {
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_DTYPE_SIZE)
};

size_t sizeof_dtype(DataType dtype) {
    return __dtype_size__[dtype];
}

#define u1 DataType::TYPE_UINT8
#define i1 DataType::TYPE_INT8
#define u2 DataType::TYPE_UINT16
#define i2 DataType::TYPE_INT16
#define u4 DataType::TYPE_UINT32
#define i4 DataType::TYPE_INT32
#define u8 DataType::TYPE_UINT64
#define i8 DataType::TYPE_INT64
#define f4 DataType::TYPE_FLOAT32
#define f8 DataType::TYPE_FLOAT64

#define U1 uint8_t
#define I1 int8_t
#define U2 uint16_t
#define I2 int16_t
#define U4 uint32_t
#define I4 int32_t
#define U8 uint64_t
#define I8 int64
#define F4 float
#define F8 double

constexpr std::array<DataType, DataType::COMPILE_TIME_MAX_DATA_TYPES> index2dtype = {
    u1, i2, u2, i2, u4, i4, u8, i8, f4, f8,
};

template<typename FromType, typename ToType>
void type_cast(void *src, void *dst) {
    *reinterpret_cast<ToType *>(dst) = *reinterpret_cast<FromType *>(src);
}

#define GEN_ITEM(dtype, type2, type1) template void type_cast<type1, type2>(void *src, void *dst);
DATATYPE_GEN_TEMPLATE_LOOPx2(GEN_ITEM)

#define GEN_ITEM_INST(dtype, type2, type1) type_cast<type1, type2>,
static std::array<
    std::array<std::function<void(void *, void *)>, index2dtype.size()>,
    index2dtype.size()> __dtype_cast_ops__ = { DATATYPE_GEN_TEMPLATE_LOOPx2(GEN_ITEM_INST) };

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

static constexpr std::array<std::array<DataType, index2dtype.size()>, index2dtype.size()>
    __promote_types__ = {
    /* align to numpy */
    /*       u1  i1  u2  i2  u4  i4  u8  i8  f4* f8*/
    /* u1 */ u1, i2, u2, i2, u4, i4, u8, i8, f4, f8,
    /* i1 */ i2, i1, i4, i2, i8, i4, f8, i8, f4, f8,
    /* u2 */ u2, i4, u2, i4, u4, i4, u8, i8, f4, f8,
    /* i2 */ i2, i2, i4, i2, i8, i4, f8, i8, f4, f8,
    /* u4 */ u4, i8, u4, i8, u4, i8, u8, i8, f8, f8,
    /* i4 */ i4, i4, i4, i4, i8, i4, f8, i8, f8, f8,
    /* u8 */ u8, f8, u8, f8, u8, f8, u8, f8, f8, f8,
    /* i8 */ i8, i8, i8, i8, i8, i8, f8, i8, f8, f8,
    /* f4 */ f4, f4, f4, f4, f8, f8, f8, f8, f4, f8,
    /* f8 */ f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,
};

inline DataType get_promote_type(DataType ltype, DataType rtype) {
    auto lindex = dtype2index[ltype], rindex = dtype2index[rtype];
    return __promote_types__[lindex][rindex];
}

template<typename ReturnType, typename LeftType, typename RightType>
void math_op(void *ret, void *lvalue, void *rvalue) {
    *reinterpret_cast<ReturnType *>(ret) =
        *reinterpret_cast<LeftType *>(lvalue) + *reinterpret_cast<RightType *>(rvalue);
}


} // namespace nnops
