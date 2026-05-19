use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nnops::mps::ops::{binary_ops, matmul};
use nnops::{DeviceType, Scalar, Tensor};

mod common;
use common::*;

// ----------------------------------------------------------------
// Binary ops: tensor-tensor on MPS
// ----------------------------------------------------------------

fn bench_mps_binary_ops(c: &mut Criterion) {
    let ops: &[(&str, fn(&Tensor, &Tensor) -> Tensor)] = &[
        ("add", binary_ops::add),
        ("sub", binary_ops::sub),
        ("mul", binary_ops::mul),
        ("truediv", binary_ops::truediv),
    ];

    for &(op_name, op_fn) in ops {
        let mut group = c.benchmark_group(format!("mps_binary_{op_name}"));
        group.measurement_time(BINARY_MEASUREMENT);
        group.warm_up_time(BINARY_WARMUP);
        for &shape in BINARY_SIZES {
            let a = tensor_f32(shape, DeviceType::Mps);
            let b = tensor_f32(shape, DeviceType::Mps);
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
                |bench| bench.iter(|| op_fn(black_box(&a), black_box(&b))),
            );
        }
        group.finish();
    }
}

// ----------------------------------------------------------------
// Binary ops: tensor-scalar on MPS
// ----------------------------------------------------------------

fn bench_mps_binary_scalar_ops(c: &mut Criterion) {
    let ops: &[(&str, fn(&Tensor, &Scalar) -> Tensor)] = &[
        ("add_scalar", binary_ops::add_scalar),
        ("sub_scalar", binary_ops::sub_scalar),
        ("mul_scalar", binary_ops::mul_scalar),
        ("truediv_scalar", binary_ops::truediv_scalar),
    ];

    for &(op_name, op_fn) in ops {
        let mut group = c.benchmark_group(format!("mps_{op_name}"));
        group.measurement_time(BINARY_MEASUREMENT);
        group.warm_up_time(BINARY_WARMUP);
        for &shape in SCALAR_SIZES {
            let a = tensor_f32(shape, DeviceType::Mps);
            let s = Scalar::Float32(3.14);
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
                |bench| bench.iter(|| op_fn(black_box(&a), black_box(&s))),
            );
        }
        group.finish();
    }
}

// ----------------------------------------------------------------
// MPS matmul
// ----------------------------------------------------------------

fn bench_mps_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_matmul");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    for &n in MATMUL_SIZES {
        let a = tensor_f32(&[n, n], DeviceType::Mps);
        let b = tensor_f32(&[n, n], DeviceType::Mps);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n}x{n}")),
            |bench| bench.iter(|| matmul::matmul(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

fn bench_mps_matmul_rect(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_matmul_rect");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    for &(s1, s2) in MATMUL_RECT_SHAPES {
        let a = tensor_f32(s1, DeviceType::Mps);
        let b = tensor_f32(s2, DeviceType::Mps);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{:?}x{:?}", s1, s2)),
            |bench| bench.iter(|| matmul::matmul(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

criterion_group! {
    name = mps_binary;
    config = Criterion::default().sample_size(BINARY_SAMPLE_SIZE);
    targets = bench_mps_binary_ops, bench_mps_binary_scalar_ops
}

criterion_group! {
    name = mps_matmul_grp;
    config = Criterion::default().sample_size(MATMUL_SAMPLE_SIZE);
    targets = bench_mps_matmul, bench_mps_matmul_rect
}

criterion_main!(mps_binary, mps_matmul_grp);