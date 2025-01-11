#include <nnops/data_type.h>
#include <array>

namespace nnops {

#define GEN_DTYPE_SIZE(dtype, type) sizeof(type),
static constexpr std::array<size_t, DataType::COMPILE_TIME_MAX_DATA_TYPES> __dtype_size__ = {
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_DTYPE_SIZE)
};

size_t sizeof_dtype(DataType dtype) {
    return __dtype_size__[dtype];
}

#define b1 DataType::TYPE_BOOL
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

#define GEN_INDEX2DTYPE_ITEM(dtype, type) dtype,
constexpr std::array<DataType, DataType::COMPILE_TIME_MAX_DATA_TYPES> index2dtype = {
    DATATYPE_GEN_TEMPLATE_LOOPx1(GEN_INDEX2DTYPE_ITEM)
};

template<typename FromType, typename ToType>
void type_cast(void **args, const index_t *strides, const index_t size) {
    void *src = args[0], *dst = args[1];
    const index_t &fs = strides[0], &ts = strides[1];
    for (int i = 0; i < size; i++) {
        *reinterpret_cast<ToType *>(dst) = *reinterpret_cast<FromType *>(src);
        dst = (void *)((char *)dst + ts);
        src = (void *)((char *)src + fs);
    }
}

#define GEN_ITEM_INST(dtype2, type2, dtype1, type1) type_cast<type1, type2>,
static std::array<
    std::array<dtype_cast_op_t, index2dtype.size()>,
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

dtype_cast_op_t get_cast_op(DataType from, DataType to) {
    auto from_idx = dtype2index[from], to_idx = dtype2index[to];
    return __dtype_cast_ops__[from][to];
}

static constexpr std::array<std::array<DataType, index2dtype.size()>, index2dtype.size()>
    __promote_types__ = {
    /* align to numpy */
    /*       b1, u1  i1  u2  i2  u4  i4  u8  i8  f4* f8*/
    /* b1 */ b1, u1, i1, u2, i2, u4, i4, u8, i8, f4, f8,
    /* u1 */ u1, u1, i2, u2, i2, u4, i4, u8, i8, f4, f8,
    /* i1 */ i1, i2, i1, i4, i2, i8, i4, f8, i8, f4, f8,
    /* u2 */ u2, u2, i4, u2, i4, u4, i4, u8, i8, f4, f8,
    /* i2 */ i2, i2, i2, i4, i2, i8, i4, f8, i8, f4, f8,
    /* u4 */ u4, u4, i8, u4, i8, u4, i8, u8, i8, f8, f8,
    /* i4 */ i4, i4, i4, i4, i4, i8, i4, f8, i8, f8, f8,
    /* u8 */ u8, u8, f8, u8, f8, u8, f8, u8, f8, f8, f8,
    /* i8 */ i8, i8, i8, i8, i8, i8, i8, f8, i8, f8, f8,
    /* f4 */ f4, f4, f4, f4, f4, f8, f8, f8, f8, f4, f8,
    /* f8 */ f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,
};

DataType get_promote_type(ScalarBinaryOpType op_type, DataType ltype, DataType rtype) {
    if (
        op_type == ScalarBinaryOpType::DIV
        && ltype != DataType::TYPE_FLOAT32 && ltype != DataType::TYPE_FLOAT64
        && rtype != DataType::TYPE_FLOAT32 && rtype != DataType::TYPE_FLOAT64
    )
        return DataType::TYPE_FLOAT64;
    return __promote_types__[ltype][rtype];
}

#define GEN_SCALAR_BINARY_OP_FUNCTOR(dtype3, type3, dtype2, type2, dtype1, type1, op_functor) op_functor<type3, type2, type1>,
#define MAKE_SCALAR_BINARY_OP_TEMPLATE(op_type, op_name, op)                     \
template<typename ReturnType, typename LeftType, typename RightType>             \
void scalar_binary_op_##op_name(void **args, const index_t *strides, const index_t size) { \
    void *ret = args[0], *lvalue = args[1], *rvalue = args[2];                   \
    const index_t ret_s = strides[0], left_s = strides[1], right_s = strides[2]; \
    for (int i = 0; i < size; i++) {                                             \
        *reinterpret_cast<ReturnType *>(ret) =                                   \
            static_cast<ReturnType>(*reinterpret_cast<LeftType *>(lvalue)) op    \
            static_cast<ReturnType>(*reinterpret_cast<RightType *>(rvalue));     \
        ret = (void *)((char *)ret + ret_s);                                     \
        lvalue = (void *)((char *)lvalue + left_s);                              \
        rvalue = (void *)((char *)rvalue + right_s);                             \
    }                                                                            \
}                                                                                \
constexpr std::array<std::array<std::array<scalar_binary_op_t,                   \
    index2dtype.size()>, index2dtype.size()>, index2dtype.size()> __functors_##op_name = { \
    DATATYPE_GEN_TEMPLATE_LOOPx3(GEN_SCALAR_BINARY_OP_FUNCTOR, scalar_binary_op_##op_name) \
};

SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_SCALAR_BINARY_OP_TEMPLATE)

#define SELECT_SCALAR_BINARY_OP_FUNCTOR(op_type, op_name, op) scalar_binary_ops[i][j][op_type] = __functors_##op_name[i][j][k];
constexpr std::array<std::array<std::array<
    scalar_binary_op_t, ScalarBinaryOpType::COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES>,
    index2dtype.size()>, index2dtype.size()> initialize_scalar_binary_op() {
    std::array<std::array<std::array<
        scalar_binary_op_t, ScalarBinaryOpType::COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES>,
        index2dtype.size()>, index2dtype.size()> scalar_binary_ops = {};

    for (int i = 0; i < index2dtype.size(); i++)
        for (int j = 0; j < index2dtype.size(); j++)
            for (int k = 0; k < index2dtype.size(); k++)
                if (index2dtype[k] == __promote_types__[index2dtype[i]][index2dtype[j]]) {
                    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(SELECT_SCALAR_BINARY_OP_FUNCTOR)
                    if (
                        index2dtype[i] != DataType::TYPE_FLOAT32 && index2dtype[i] != DataType::TYPE_FLOAT64
                        && index2dtype[j] != DataType::TYPE_FLOAT32 && index2dtype[j] != DataType::TYPE_FLOAT64
                    )
                        scalar_binary_ops[i][j][ScalarBinaryOpType::DIV] = __functors_div[i][j][DataType::TYPE_FLOAT64];
                }

    return scalar_binary_ops;
}

constexpr std::array<std::array<std::array<
    scalar_binary_op_t, ScalarBinaryOpType::COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES>,
    index2dtype.size()>, index2dtype.size()> __scalar_binary_ops__ = initialize_scalar_binary_op();

scalar_binary_op_t get_scalar_binary_op(ScalarBinaryOpType op_type, DataType ltype, DataType rtype) {
    auto left_idx = dtype2index[ltype], right_idx = dtype2index[rtype];
    return __scalar_binary_ops__[right_idx][left_idx][op_type];
}

} // namespace nnops
