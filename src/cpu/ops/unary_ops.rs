use crate::tensor::Tensor;
use crate::cpu::ops::matmul::matmul;
use crate::cpu::ops::binary_ops::add;

pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    nnops_check!(
        input.shape_at(-1) == weight.shape_at(1),
        "linear input and weight are incompatible."
    );
    if let Some(b) = bias {
        nnops_check!(
            b.shape_at(0) == weight.shape_at(0),
            "linear bias and weight are incompatible."
        );
    }
    let w_t = weight.transpose(-1, -2);
    let mut ret = matmul(input, &w_t);
    if let Some(b) = bias {
        ret = add(&ret, b);
    }
    ret
}