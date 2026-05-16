use crate::tensor::Tensor;

pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    crate::cpu::ops::unary_ops::linear(input, weight, bias)
}