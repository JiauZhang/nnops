#include <gtest/gtest.h>
#include <nnops/device.h>
#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <nnops/tensor_iterator.h>
#include <nnops/tensor_accessor.h>
#include <nnops/cpu/ops/functional.h>
#include <array>
#include <vector>

using nnops::DataType, nnops::DeviceType;
using nnops::Tensor, nnops::TensorShape, nnops::index_t;
using nnops::TensorIterator, nnops::TensorAccessor, nnops::TensorPartialIterator;

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
    ASSERT_EQ(t_cpu_u8.nelems(), 360);
    ASSERT_EQ(t_cpu_u8.nbytes(), 360);
    Tensor t_cpu_f32(DataType::TYPE_FLOAT32, shape, DeviceType::CPU);
    ASSERT_EQ(t_cpu_f32.nelems(), 360);
    ASSERT_EQ(t_cpu_f32.nbytes(), 360 * 4);
}

TEST_F(TensorTest, IsBroadcast) {
    const TensorShape s1 = {2, 3, 4, 5, 7, 8}, s2 = {4, 1, 8, 3}, s3 = {4, 2, 8, 3}, s4 = {2, 1, 8, 3};
    ASSERT_TRUE(Tensor::is_broadcastable(s1, s2, 2));
    ASSERT_FALSE(Tensor::is_broadcastable(s1, s3, 2));
    ASSERT_TRUE(Tensor::is_broadcastable(s1, s3, 3));
    ASSERT_FALSE(Tensor::is_broadcastable(s1, s4, 2));
    ASSERT_TRUE(Tensor::is_broadcastable(s2, s3, 2));
    ASSERT_FALSE(Tensor::is_broadcastable(s2, s4, 2));
    ASSERT_FALSE(Tensor::is_broadcastable(s3, s4, 2));
    const TensorShape s5 = {3, 4}, s6 = {6, 7};
    ASSERT_TRUE(Tensor::is_broadcastable(s5, s6, 2));
}

TEST_F(TensorTest, BroadcastShape) {
    const TensorShape s1 = {2, 3, 4, 5, 7, 8}, s2 = {4, 1, 8, 3}, s3 = {4, 2, 8, 3};
    ASSERT_EQ(Tensor::broadcast_shape(s1, s2, 2), std::vector<index_t>({2, 3, 4, 5}));
    ASSERT_EQ(Tensor::broadcast_shape(s1, s2, 3), std::vector<index_t>({2, 3, 4}));
    const TensorShape s5 = {3, 4}, s6 = {6, 7};
    ASSERT_EQ(Tensor::broadcast_shape(s5, s6, 2), std::vector<index_t>());
}

TEST_F(TensorTest, BroadcastTo) {
    const TensorShape s1 = {2, 1, 4, 1, 7, 8}, s2 = {4, 1, 8, 3};
    const Tensor t1(DataType::TYPE_UINT8, s1, DeviceType::CPU), t2(DataType::TYPE_UINT8, s2, DeviceType::CPU);
    const TensorShape s3 = {2, 3, 4, 5, 6, 6}, s4 = {4, 3, 3, 3};
    ASSERT_EQ(Tensor::broadcast_to(t1, s3, 2).shape(), std::vector<index_t>({2, 3, 4, 5, 7, 8}));
    ASSERT_EQ(Tensor::broadcast_to(t1, s3, 5).shape(), std::vector<index_t>({2, 1, 4, 1, 7, 8}));
    ASSERT_EQ(Tensor::broadcast_to(t2, s4, 2).shape(), std::vector<index_t>({4, 3, 8, 3}));
    ASSERT_EQ(Tensor::broadcast_to(t2, s4, 3).shape(), std::vector<index_t>({4, 1, 8, 3}));
    ASSERT_EQ(Tensor::broadcast_to(t2, s3, 3).shape(), std::vector<index_t>({2, 3, 4, 1, 8, 3}));
    ASSERT_EQ(Tensor::broadcast_to(t2, s1, 3).shape(), std::vector<index_t>({2, 1, 4, 1, 8, 3}));
    ASSERT_EQ(Tensor::broadcast_to(t2, s1, 2).shape(), std::vector<index_t>({2, 1, 4, 1, 8, 3}));
    const TensorShape s5 = {3, 4}, s6 = {6, 7};
    const Tensor t3(DataType::TYPE_UINT8, s5, DeviceType::CPU);
    ASSERT_EQ(Tensor::broadcast_to(t3, s6, 2).shape(), s5);
    ASSERT_EQ(Tensor::broadcast_to(t3, s1, 2).shape(), std::vector<index_t>({2, 1, 4, 1, 3, 4}));
}

TEST_F(TensorTest, TensorMatMul) {
    const TensorShape s1 = {2, 3}, s2 = {3, 3};
    const Tensor t1(DataType::TYPE_FLOAT32, s1, DeviceType::CPU), t2(DataType::TYPE_FLOAT32, s2, DeviceType::CPU);
    // t1: [[0, 1, 2], [3, 4, 5]]
    for (int i = 0; i < t1.nelems(); i++)
        *((float *)t1.data_ptr(i)) = i;
    // t2: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for (int i = 0; i < t2.nelems(); i++)
        *((float *)t2.data_ptr(i)) = i;
    Tensor ret = nnops::cpu::ops::matmul(t1, t2);
    float out[6] = {15, 18, 21, 42, 54, 66};
    ASSERT_EQ(ret.nelems(), 6);
    for (int i = 0; i < 6; i++)
        ASSERT_EQ(*((float *)ret.data_ptr(i)), out[i]);
}

TEST_F(TensorTest, UnravelIndex) {
    ASSERT_EQ(shape.size() * indices.size(), outputs.size());
    for (int i = 0; i < indices.size(); i++) {
        TensorShape out = Tensor::unravel_index(indices[i], shape);
        ASSERT_EQ(out.size(), shape.size());
        for (int j = 0; j < shape.size(); j++)
            ASSERT_EQ(out[j], outputs[i * shape.size() + j]);
    }
}

TEST_F(TensorTest, RavelIndex) {
    ASSERT_EQ(shape.size() * indices.size(), outputs.size());
    for (int i = 0; i < indices.size(); i++) {
        TensorShape dims(shape.size());
        for (int j = 0; j < shape.size(); j++)
            dims[j] = outputs[i * shape.size() + j];
        index_t out = Tensor::ravel_index(dims, shape);
        ASSERT_EQ(out, indices[i]);
    }
}

TEST_F(TensorTest, TensorIteratorAndAccessor) {
    TensorShape s = {2, 2, 3};
    Tensor t(DataType::TYPE_FLOAT32, s, DeviceType::CPU);
    float *ptr = (float *)t.data_ptr();
    int i;
    for (i = 0; i < t.nelems(); i++)
        ptr[i] = i + 1.12345;

    TensorIterator iter(t);
    i = 0;
    for (; !iter.is_end(); ++iter, ++i)
        ASSERT_EQ(*iter, (void *)(ptr + i));

    TensorAccessor acc(t);
    TensorShape anchor_0 = {0, 0, 0}, anchor_1 = {1, 1, 2}, anchor_2 = {0, 1, 1};
    float *ptr_0 = (float *)acc.data_ptr_unsafe(anchor_0);
    ASSERT_EQ(ptr_0, ptr);
    float *ptr_1 = (float *)acc.data_ptr_unsafe(anchor_1);
    ASSERT_EQ(ptr_1, ptr + 11);
    float *ptr_2 = (float *)acc.data_ptr_unsafe(anchor_2);
    ASSERT_EQ(ptr_2, ptr + 4);

    for (i = 0; i < s[1]; i++) {
        for (int j = 0; j < s[2]; j++) {
            TensorShape anchor_ij = {0, i, j};
            float *ptr_ij = (float *)acc.data_ptr_unsafe(anchor_ij);
            ASSERT_EQ(*ptr_ij, *(ptr + i * 3 + j));
            float *ptr_ij_next = (float *)acc.data_ptr_unsafe((void *)ptr_ij, 1, 0);
            ASSERT_EQ(*ptr_ij_next, *(ptr + 6 + i * 3 + j));
        }
    }
}

TEST_F(TensorTest, TensorPartialIteratorAndAccessor) {
    TensorShape s = {2, 2, 3, 2, 3};
    Tensor t(DataType::TYPE_FLOAT32, s, DeviceType::CPU);
    float *data_ptr = (float *)t.data_ptr();
    for (int i = 0; i < t.nelems(); i++)
        data_ptr[i] = (float)i;
    // make a 3-d sub tensor iterator
    TensorPartialIterator iter(t, 0, s.size() - 3);
    int iter_count = 0;
    for (; !iter.is_end(); ++iter, ++iter_count) {
        int sub_iter_count = 0;
        Tensor sub_t = iter.tensor();
        ASSERT_EQ(sub_t.ndim(), 3);
        ASSERT_EQ(sub_t.nelems(), 18);
        ASSERT_EQ(sub_t.ref_count(), 2);
        TensorIterator sub_iter(sub_t);
        for (; !sub_iter.is_end(); ++sub_iter, ++sub_iter_count) {
            float *sub_ptr = (float *)(*sub_iter);
            ASSERT_EQ(*sub_ptr, (float)(iter_count * 18 + sub_iter_count));
        }

        TensorShape ijk(3, 0);
        TensorAccessor acc_ijk(sub_t);
        void *anchor_ptr = acc_ijk.data_ptr_unsafe(ijk);
        for (int i = 0; i < sub_t.shape()[0]; i++) {
            ijk[0] = i;
            void *anchor_ptr_i = acc_ijk.data_ptr_unsafe(anchor_ptr, i, 0);
            for (int j = 0; j < sub_t.shape()[1]; j++) {
                ijk[1] = j;
                void *anchor_ptr_ij = acc_ijk.data_ptr_unsafe(anchor_ptr_i, j, 1);
                for (int k = 0; k < sub_t.shape()[2]; k++) {
                    ijk[2] = k;
                    void *anchor_ptr_ijk = acc_ijk.data_ptr_unsafe(anchor_ptr_ij, k, 2);
                    float *sub_ptr = (float *)acc_ijk.data_ptr_unsafe(ijk);
                    ASSERT_EQ(*sub_ptr, (float)(iter_count * 18 + i * 6 + j * 3 + k));
                    ASSERT_EQ((void *)sub_ptr, anchor_ptr_ijk);
                }
            }
        }
    }
}
