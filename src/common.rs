pub type Index = i64;
pub type TensorShape = Vec<Index>;
pub type TensorStride = Vec<Index>;

#[macro_export]
macro_rules! nnops_check {
    ($cond:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {
        if !($cond) {
            let msg = format!($fmt $(, $arg)*);
            panic!(
                "{} CHECK FAILED at {}::{}::L{}\n{}",
                stringify!($cond),
                file!(),
                line!(),
                line!(),
                msg
            );
        }
    };
}

pub fn shape_as_string(dims: &TensorShape) -> String {
    if dims.is_empty() {
        return "[]".to_string();
    }
    let mut s = String::from("[");
    for (i, dim) in dims.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&dim.to_string());
    }
    s.push(']');
    s
}

pub fn unravel_index(mut idx: Index, shape: &TensorShape) -> TensorShape {
    let mut indices = vec![0i64; shape.len()];
    let mut strides_contig = vec![0i64; shape.len()];
    strides_contig[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides_contig[i] = strides_contig[i + 1] * shape[i + 1];
    }
    let nelems = strides_contig[0] * shape[0];
    nnops_check!(idx <= nelems, "index {} is out of bounds for TensorShape with size {}", idx, nelems);
    for i in 0..indices.len() {
        indices[i] = idx / strides_contig[i];
        idx -= indices[i] * strides_contig[i];
    }
    indices
}

pub fn ravel_index(indices: &TensorShape, shape: &TensorShape) -> Index {
    nnops_check!(
        indices.len() == shape.len(),
        "parameter indices must be a sequence of length {}",
        shape.len()
    );
    let mut strides_contig = vec![0i64; shape.len()];
    strides_contig[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides_contig[i] = strides_contig[i + 1] * shape[i + 1];
    }
    for i in 0..shape.len() {
        nnops_check!(
            indices[i] < shape[i],
            "indices[{}]: {} is out of bounds for shape[{}]: {}",
            i, indices[i], i, shape[i]
        );
    }
    let mut idx = 0;
    for i in 0..shape.len() {
        idx += indices[i] * strides_contig[i];
    }
    idx
}

pub fn is_contiguous(shape: &TensorShape, strides: &TensorStride) -> bool {
    let mut nelems = 1;
    for i in (0..shape.len()).rev() {
        if nelems != strides[i] {
            return false;
        }
        nelems *= shape[i];
    }
    true
}

pub fn is_broadcastable(s1: &TensorShape, s2: &TensorShape, offset: usize) -> bool {
    let dims = std::cmp::min(s1.len(), s2.len());
    let s1_size = s1.len() - 1;
    let s2_size = s2.len() - 1;
    for i in offset..dims {
        if s1[s1_size - i] != s2[s2_size - i] && s1[s1_size - i] != 1 && s2[s2_size - i] != 1 {
            return false;
        }
    }
    true
}

pub fn broadcast_shape(s1: &TensorShape, s2: &TensorShape, offset: usize) -> TensorShape {
    let dims = std::cmp::min(s1.len(), s2.len());
    let (shape_long, shape_short): (&TensorShape, &TensorShape) = if dims == s1.len() {
        (s2, s1)
    } else {
        (s1, s2)
    };
    let dims = shape_long.len() - offset;
    let mut shape = vec![0i64; dims];
    shape[..dims].copy_from_slice(&shape_long[..dims]);
    let mut dims = dims as isize - 1;
    for i in (0..shape_short.len() - offset).rev() {
        if shape_short[i] != shape[dims as usize] {
            shape[dims as usize] *= shape_short[i];
        }
        dims -= 1;
    }
    shape
}

pub fn is_broadcastable_to(self_shape: &TensorShape, other: &TensorShape, offset: usize) -> bool {
    let dims = std::cmp::min(self_shape.len(), other.len());
    if self_shape.len() > other.len() || dims < offset {
        return false;
    }
    let self_i = self_shape.len() - 1;
    let other_i = other.len() - 1;
    for i in offset..dims {
        if self_shape[self_i - i] != other[other_i - i] && self_shape[self_i - i] != 1 {
            return false;
        }
    }
    true
}

pub fn broadcast_to(
    shape: &TensorShape,
    strides: &TensorStride,
    target_shape: &TensorShape,
    offset: usize,
) -> (TensorShape, TensorStride) {
    nnops_check!(
        is_broadcastable_to(shape, target_shape, offset),
        "could not broadcast tensor from shape {} into shape {}",
        shape_as_string(shape),
        shape_as_string(target_shape)
    );

    let mut new_strides = strides.clone();
    if shape.is_empty() {
        return (target_shape.clone(), new_strides);
    }
    let ts_size = shape.len() - 1;
    let s_size = target_shape.len() - 1;
    let diff = target_shape.len() - shape.len();
    let mut shape_cp = target_shape.clone();

    if ts_size >= offset {
        for i in 0..=ts_size - offset {
            if shape[i] == 1 {
                new_strides[i] = 0;
            }
        }
    }
    for i in 0..offset {
        shape_cp[s_size - i] = shape[ts_size - i];
    }
    new_strides.resize(target_shape.len(), 0);
    for i in (diff..=s_size).rev() {
        new_strides[i] = new_strides[i - diff];
    }
    for item in new_strides.iter_mut().take(diff) {
        *item = 0;
    }

    (shape_cp, new_strides)
}