#include <nnops/data_type.h>
#include <array>
#include <utility> // for std::index_sequence, std::make_index_sequence

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

template<DataType FromDataType, DataType ToDataType>
void type_cast(void **args, const index_t *strides, const index_t size) {
    using FromCppType = DataTypeToCppType<FromDataType>;
    using ToCppType = DataTypeToCppType<ToDataType>;
    void *src = args[0], *dst = args[1];
    const index_t &fs = strides[0], &ts = strides[1];
    for (int i = 0; i < size; i++) {
        *reinterpret_cast<ToCppType *>(dst) = *reinterpret_cast<FromCppType *>(src);
        dst = (void *)((char *)dst + ts);
        src = (void *)((char *)src + fs);
    }
}

template <size_t I, size_t J>
constexpr dtype_cast_op_t get_cast_op_by_index() {
    constexpr DataType FromDataType = DataTypeElement<I>;
    constexpr DataType ToDataType = DataTypeElement<J>;
    return type_cast<FromDataType, ToDataType>;
}

template <size_t I, size_t... J>
constexpr auto make_inner_array(std::index_sequence<J...>) {
    return std::array<dtype_cast_op_t, sizeof...(J)> {
        get_cast_op_by_index<I, J>()...
    };
}

template <size_t... I>
constexpr auto make_outer_array(std::index_sequence<I...>) {
    return std::array<std::array<dtype_cast_op_t, sizeof...(I)>, sizeof...(I)> {
        make_inner_array<I>(std::make_index_sequence<sizeof...(I)>{})...
    };
}

constexpr auto make_dtype_cast_ops() {
    return make_outer_array(std::make_index_sequence<kNumDataTypes>{});
}

static constexpr auto __dtype_cast_ops__ = make_dtype_cast_ops();
static_assert(__dtype_cast_ops__.size() == kNumDataTypes, "dtype cast table row size error!");
static_assert(__dtype_cast_ops__[0].size() == kNumDataTypes, "dtype cast column size error!");

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

template <typename T, ScalarBinaryOpType OP>
struct ScalarBinaryOpImpl;

#define DEFINE_SCALAR_BINARY_OP_IMPL(op_type, op_name, op_symbol)              \
template <typename T> \
struct ScalarBinaryOpImpl<T, op_type> { \
    static void apply(void **args, const index_t *strides, const index_t size) { \
        void *ret = args[0], *lvalue = args[1], *rvalue = args[2];                   \
        const index_t ret_s = strides[0], left_s = strides[1], right_s = strides[2]; \
        for (int i = 0; i < size; i++) {                                             \
            *reinterpret_cast<T *>(ret) = (*reinterpret_cast<T *>(lvalue)) op_symbol (*reinterpret_cast<T *>(rvalue));     \
            ret = (void *)((char *)ret + ret_s);                                     \
            lvalue = (void *)((char *)lvalue + left_s);                              \
            rvalue = (void *)((char *)rvalue + right_s);                             \
        }                                                                            \
    } \
};
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(DEFINE_SCALAR_BINARY_OP_IMPL)

template <ScalarBinaryOpType OP, DataType DT>
constexpr scalar_binary_op_t get_scalar_binary_op_func() {
    using T = std::tuple_element_t<static_cast<size_t>(DT), CppType>;
    return &ScalarBinaryOpImpl<T, OP>::apply;
}

template <ScalarBinaryOpType OP, size_t... Idxs>
constexpr auto make_op_array(std::index_sequence<Idxs...>) {
    return std::array<scalar_binary_op_t, sizeof...(Idxs)>{
        get_scalar_binary_op_func<OP, static_cast<DataType>(Idxs)>()...
    };
}

template <size_t... Ops>
constexpr auto make_ops_array(std::index_sequence<Ops...>) {
    return std::array{
        make_op_array<static_cast<ScalarBinaryOpType>(Ops)>(
            std::make_index_sequence<DataType::COMPILE_TIME_MAX_DATA_TYPES>{}
        )...
    };
}

static constexpr auto __scalar_binary_ops__ = []() constexpr {
    return make_ops_array(
        std::make_index_sequence<ScalarBinaryOpType::COMPILE_TIME_MAX_SCALAR_BINARY_OP_TYPES>{}
    );
}();

scalar_binary_op_t get_scalar_binary_op(ScalarBinaryOpType op_type, DataType dtype) {
    return __scalar_binary_ops__[op_type][dtype];
}

} // namespace nnops
