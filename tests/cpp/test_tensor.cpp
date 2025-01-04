#include <gtest/gtest.h>
#include <nnops/device.h>
#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/tensor_iterator.h>
#include <nnops/tensor_accessor.h>
#include <array>

using nnops::DataType, nnops::DeviceType;
using nnops::Tensor, nnops::TensorShape, nnops::index_t;
using nnops::TensorIterator, nnops::TensorAccessor;

class TensorTest : public testing::Test {
protected:
    const TensorShape shape = {3, 4, 5, 6};
    const std::array<index_t, 10> indices = {0, 1, 6, 30, 120, 359, 8, 33, 128, 233};
    const std::array<index_t, 10 * 4> outputs = {
        // 0, 1, 6, 30, 120
        0, 0, 0, 0,    0, 0, 0, 1,    0, 0, 1, 0,    0, 1, 0, 0,    1, 0, 0, 0,
        // 359, 8, 33, 128, 233
        2, 3, 4, 5,    0, 0, 1, 2,    0, 1, 0, 3,    1, 0, 1, 2,    1, 3, 3, 5
    };
};

TEST_F(TensorTest, TensorInit) {
    Tensor t_cpu_u8(DataType::TYPE_UINT8, shape, DeviceType::CPU);
    EXPECT_EQ(t_cpu_u8.nelems(), 360);
    EXPECT_EQ(t_cpu_u8.nbytes(), 360);
    Tensor t_cpu_f32(DataType::TYPE_FLOAT32, shape, DeviceType::CPU);
    EXPECT_EQ(t_cpu_f32.nelems(), 360);
    EXPECT_EQ(t_cpu_f32.nbytes(), 360 * 4);
}

TEST_F(TensorTest, UnravelIndex) {
    EXPECT_EQ(shape.size() * indices.size(), outputs.size());
    for (int i = 0; i < indices.size(); i++) {
        TensorShape out = Tensor::unravel_index(indices[i], shape);
        EXPECT_EQ(out.size(), shape.size());
        for (int j = 0; j < shape.size(); j++)
            EXPECT_EQ(out[j], outputs[i * shape.size() + j]);
    }
}

TEST_F(TensorTest, RavelIndex) {
    EXPECT_EQ(shape.size() * indices.size(), outputs.size());
    for (int i = 0; i < indices.size(); i++) {
        TensorShape dims(shape.size());
        for (int j = 0; j < shape.size(); j++)
            dims[j] = outputs[i * shape.size() + j];
        index_t out = Tensor::ravel_index(dims, shape);
        EXPECT_EQ(out, indices[i]);
    }
}

TEST_F(TensorTest, TensorIteratorAndAccessor) {
    TensorShape s = {2, 2, 3};
    Tensor t(DataType::TYPE_FLOAT32, s, DeviceType::CPU);
    float *ptr = (float *)t.data_ptr();
    int i;
    for (i = 0; i < t.nelems(); i++)
        ptr[i] = i + 1.12345;

    TensorIterator iter = t.begin(), iter_end = t.end();
    i = 0;
    for (; iter != iter_end; ++iter, ++i)
        EXPECT_EQ(*iter, (void *)(ptr + i));

    TensorAccessor acc = t.accessor();
    TensorShape anchor_0 = {0, 0, 0}, anchor_1 = {1, 1, 2}, anchor_2 = {0, 1, 1};
    float *ptr_0 = (float *)acc.data_ptr_unsafe(anchor_0);
    EXPECT_EQ(ptr_0, ptr);
    float *ptr_1 = (float *)acc.data_ptr_unsafe(anchor_1);
    EXPECT_EQ(ptr_1, ptr + 11);
    float *ptr_2 = (float *)acc.data_ptr_unsafe(anchor_2);
    EXPECT_EQ(ptr_2, ptr + 4);

    for (i = 0; i < s[1]; i++) {
        for (int j = 0; j < s[2]; j++) {
            TensorShape anchor_ij = {0, i, j};
            float *ptr_ij = (float *)acc.data_ptr_unsafe(anchor_ij);
            EXPECT_EQ(*ptr_ij, *(ptr + i * 3 + j));
            float *ptr_ij_next = (float *)acc.data_ptr_unsafe((void *)ptr_ij, 1, 0);
            EXPECT_EQ(*ptr_ij_next, *(ptr + 6 + i * 3 + j));
        }
    }
}
