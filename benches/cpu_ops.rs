use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nnops::cpu::ops::{binary_ops, matmul, unary_ops};
use nnops::{DataType, DeviceType, Scalar, Tensor};

mod common;
use common::*;

fn tensor_f64(shape: &[i64]) -> Tensor {
    let mut t = Tensor::with_device_type(DataType::Float64, &shape.to_vec(), DeviceType::Cpu);
    t.fill_with(|i| (i % 997) as f64);
    t
}

fn tensor_i32(shape: &[i64]) -> Tensor {
    let mut t = Tensor::with_device_type(DataType::Int32, &shape.to_vec(), DeviceType::Cpu);
    t.fill_with(|i| (i % 997) as f64);
    t
}

// ----------------------------------------------------------------
// Binary ops: tensor-tensor
// ----------------------------------------------------------------

fn bench_binary_ops(c: &mut Criterion) {
    let ops: &[(&str, fn(&Tensor, &Tensor) -> Tensor)] = &[
        ("add", binary_ops::add),
        ("sub", binary_ops::sub),
        ("mul", binary_ops::mul),
        ("truediv", binary_ops::truediv),
    ];

    for &(op_name, op_fn) in ops {
        let mut group = c.benchmark_group(format!("cpu_binary_{op_name}"));
        group.measurement_time(BINARY_MEASUREMENT);
        group.warm_up_time(BINARY_WARMUP);
        for &shape in BINARY_SIZES {
            let a = tensor_f32(shape, DeviceType::Cpu);
            let b = tensor_f32(shape, DeviceType::Cpu);
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
                |bench| bench.iter(|| op_fn(black_box(&a), black_box(&b))),
            );
        }
        group.finish();
    }
}

fn bench_binary_ops_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_binary_f64_add");
    group.measurement_time(BINARY_MEASUREMENT);
    group.warm_up_time(BINARY_WARMUP);
    for &shape in SCALAR_SIZES {
        let a = tensor_f64(shape);
        let b = tensor_f64(shape);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
            |bench| bench.iter(|| binary_ops::add(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

fn bench_binary_ops_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_binary_i32_add");
    group.measurement_time(BINARY_MEASUREMENT);
    group.warm_up_time(BINARY_WARMUP);
    for &shape in SCALAR_SIZES {
        let a = tensor_i32(shape);
        let b = tensor_i32(shape);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
            |bench| bench.iter(|| binary_ops::add(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

// ----------------------------------------------------------------
// Binary ops: tensor-scalar
// ----------------------------------------------------------------

fn bench_binary_scalar_ops(c: &mut Criterion) {
    let ops: &[(&str, fn(&Tensor, &Scalar) -> Tensor)] = &[
        ("add_scalar", binary_ops::add_scalar),
        ("sub_scalar", binary_ops::sub_scalar),
        ("mul_scalar", binary_ops::mul_scalar),
        ("truediv_scalar", binary_ops::truediv_scalar),
    ];

    for &(op_name, op_fn) in ops {
        let mut group = c.benchmark_group(format!("cpu_{op_name}"));
        group.measurement_time(BINARY_MEASUREMENT);
        group.warm_up_time(BINARY_WARMUP);
        for &shape in SCALAR_SIZES {
            let a = tensor_f32(shape, DeviceType::Cpu);
            let s = Scalar::Float32(3.14);
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
                |bench| bench.iter(|| op_fn(black_box(&a), black_box(&s))),
            );
        }
        group.finish();
    }
}

fn bench_binary_scalar_reverse_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_add_scalar_reverse");
    group.measurement_time(BINARY_MEASUREMENT);
    group.warm_up_time(BINARY_WARMUP);
    for &shape in SCALAR_SIZES {
        let a = tensor_f32(shape, DeviceType::Cpu);
        let s = Scalar::Float32(3.14);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}x{}", shape[0], shape[1])),
            |bench| bench.iter(|| binary_ops::add_scalar_reverse(black_box(&s), black_box(&a))),
        );
    }
    group.finish();
}

// ----------------------------------------------------------------
// Matmul
// ----------------------------------------------------------------

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_matmul");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    for &n in MATMUL_SIZES {
        let a = tensor_f32(&[n, n], DeviceType::Cpu);
        let b = tensor_f32(&[n, n], DeviceType::Cpu);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n}x{n}")),
            |bench| bench.iter(|| matmul::matmul(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

fn bench_matmul_rect(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_matmul_rect");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    for &(s1, s2) in MATMUL_RECT_SHAPES {
        let a = tensor_f32(s1, DeviceType::Cpu);
        let b = tensor_f32(s2, DeviceType::Cpu);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{:?}x{:?}", s1, s2)),
            |bench| bench.iter(|| matmul::matmul(black_box(&a), black_box(&b))),
        );
    }
    group.finish();
}

// ----------------------------------------------------------------
// Linear (matmul + bias add)
// ----------------------------------------------------------------

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_linear");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    let configs: &[(&str, i64, i64, i64)] = &[
        ("small", 1, 128, 64),
        ("medium", 32, 256, 128),
        ("large", 64, 512, 256),
    ];
    for &(label, batch, in_features, out_features) in configs {
        let input = tensor_f32(&[batch, in_features], DeviceType::Cpu);
        let weight = tensor_f32(&[out_features, in_features], DeviceType::Cpu);
        let bias = tensor_f32(&[out_features], DeviceType::Cpu);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{label}_b{batch}_i{in_features}_o{out_features}")),
            |bench| {
                bench.iter(|| {
                    unary_ops::linear(
                        black_box(&input),
                        black_box(&weight),
                        Some(black_box(&bias)),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_linear_nobias(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_linear_nobias");
    group.measurement_time(MATMUL_MEASUREMENT);
    group.warm_up_time(MATMUL_WARMUP);

    let input = tensor_f32(&[32, 256], DeviceType::Cpu);
    let weight = tensor_f32(&[128, 256], DeviceType::Cpu);
    group.bench_function(
        BenchmarkId::from_parameter("b32_i256_o128"),
        |bench| bench.iter(|| unary_ops::linear(black_box(&input), black_box(&weight), None)),
    );
    group.finish();
}

criterion_group! {
    name = cpu_binary;
    config = Criterion::default().sample_size(BINARY_SAMPLE_SIZE);
    targets = bench_binary_ops, bench_binary_ops_f64, bench_binary_ops_i32,
              bench_binary_scalar_ops, bench_binary_scalar_reverse_ops
}

criterion_group! {
    name = cpu_matmul_grp;
    config = Criterion::default().sample_size(MATMUL_SAMPLE_SIZE);
    targets = bench_matmul, bench_matmul_rect
}

criterion_group! {
    name = cpu_linear_grp;
    config = Criterion::default().sample_size(MATMUL_SAMPLE_SIZE);
    targets = bench_linear, bench_linear_nobias
}

criterion_main!(cpu_binary, cpu_matmul_grp, cpu_linear_grp);