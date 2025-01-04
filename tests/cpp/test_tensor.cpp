#include <gtest/gtest.h>
#include <nnops/device.h>
#include <nnops/data_type.h>
#include <nnops/tensor_meta.h>
#include <nnops/tensor.h>
#include <array>

using nnops::DataType, nnops::DeviceType;
using nnops::Tensor, nnops::TensorShape, nnops::index_t;

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
