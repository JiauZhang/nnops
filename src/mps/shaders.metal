#include <metal_stdlib>
using namespace metal;

#define BINARY_OP_F32(name, op)                                         \
  kernel void name(device const float* a [[buffer(0)]],                 \
                   device const float* b [[buffer(1)]],                 \
                   device float* out [[buffer(2)]],                     \
                   uint index [[thread_position_in_grid]]) {            \
    out[index] = a[index] op b[index];                                  \
  }

BINARY_OP_F32(add_f32, +)
BINARY_OP_F32(sub_f32, -)
BINARY_OP_F32(mul_f32, *)
BINARY_OP_F32(div_f32, /)

#define MATMUL_KERNEL(name, T)                                          \
  kernel void name(device const T* a [[buffer(0)]],                     \
                   device const T* b [[buffer(1)]],                     \
                   device T* out [[buffer(2)]],                         \
                   constant uint& M [[buffer(3)]],                      \
                   constant uint& N [[buffer(4)]],                      \
                   constant uint& K [[buffer(5)]],                      \
                   uint2 gid [[thread_position_in_grid]]) {             \
    if (gid.x >= M || gid.y >= N) return;                              \
    T sum = 0;                                                          \
    for (uint i = 0; i < K; i++) {                                      \
      sum += a[gid.x * K + i] * b[i * N + gid.y];                      \
    }                                                                   \
    out[gid.x * N + gid.y] = sum;                                       \
  }

MATMUL_KERNEL(matmul_f32, float)
MATMUL_KERNEL(matmul_i32, int)
MATMUL_KERNEL(matmul_u32, uint)
MATMUL_KERNEL(matmul_i16, short)
MATMUL_KERNEL(matmul_u16, ushort)
MATMUL_KERNEL(matmul_i8, char)
MATMUL_KERNEL(matmul_u8, uchar)