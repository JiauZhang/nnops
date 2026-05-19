use crate::common::{Index, TensorShape, TensorStride, is_contiguous, shape_as_string};
use crate::data_type::{DataType, sizeof_dtype};

#[derive(Debug, Clone)]
pub struct Slice {
    pub start: Option<Index>,
    pub stop: Option<Index>,
    pub step: Option<Index>,
}

impl Default for Slice {
    fn default() -> Self {
        Self::new()
    }
}

impl Slice {
    pub fn new() -> Self {
        Slice { start: None, stop: None, step: None }
    }

    pub fn with_start(start: Index) -> Self {
        Slice { start: Some(start), stop: None, step: None }
    }

    pub fn with_start_stop(start: Index, stop: Index) -> Self {
        Slice { start: Some(start), stop: Some(stop), step: None }
    }

    pub fn new_full(start: Index, stop: Index, step: Index) -> Self {
        Slice { start: Some(start), stop: Some(stop), step: Some(step) }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub nelems: usize,
    pub offset: Index,
    pub dims: TensorShape,
    pub strides: TensorStride,
    pub dtype: DataType,
}

impl Default for TensorMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorMeta {
    pub fn new() -> Self {
        TensorMeta {
            nelems: 0,
            offset: 0,
            dims: Vec::new(),
            strides: Vec::new(),
            dtype: DataType::Float32,
        }
    }

    pub fn with_dtype_shape(dtype: DataType, dims: &TensorShape) -> Self {
        let mut nelems: usize = 1;
        let mut strides = vec![0i64; dims.len()];
        for i in (0..dims.len()).rev() {
            strides[i] = nelems as Index;
            nelems *= dims[i] as usize;
        }
        TensorMeta {
            nelems,
            offset: 0,
            dims: dims.clone(),
            strides,
            dtype,
        }
    }

    pub fn shape_as_string(&self) -> String {
        shape_as_string(&self.dims)
    }

    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.dims, &self.strides)
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn itemsize(&self) -> Index {
        sizeof_dtype(self.dtype)
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.nelems * self.itemsize() as usize
    }

    pub fn permute(&self, index: &TensorShape) -> Self {
        let len = index.len();
        nnops_check!(len == self.dims.len(), "axes size don't match!");
        let mut count = vec![0i64; len];
        let mut indices = index.clone();
        for idx in indices.iter_mut() {
            nnops_check!(
                !(*idx < -(len as Index) || *idx >= len as Index),
                "axis {} is out of bounds for tensor of dimension {}",
                *idx, len
            );
            if *idx < 0 {
                *idx += len as Index;
            }
            count[*idx as usize] += 1;
            nnops_check!(count[*idx as usize] <= 1, "repeated axis in permute");
        }

        let mut meta = self.clone();
        for (i, &idx) in indices.iter().enumerate() {
            meta.dims[i] = self.dims[idx as usize];
            meta.strides[i] = self.strides[idx as usize];
        }
        meta
    }

    pub fn transpose(&self, dim0: Index, dim1: Index) -> Self {
        let size = self.ndim() as Index;
        let mut dims = [dim0, dim1];
        for d in &mut dims {
            nnops_check!(
                !(*d < -size || *d >= size),
                "axis {} is out of bounds for tensor of dimension {}",
                *d, size
            );
            if *d < 0 {
                *d += size;
            }
        }
        nnops_check!(dims[0] != dims[1], "repeated axis in transpose");

        let mut meta = self.clone();
        meta.dims[dims[0] as usize] = self.dims[dims[1] as usize];
        meta.strides[dims[0] as usize] = self.strides[dims[1] as usize];
        meta.dims[dims[1] as usize] = self.dims[dims[0] as usize];
        meta.strides[dims[1] as usize] = self.strides[dims[0] as usize];
        meta
    }

    pub fn reshape_inplace(&mut self, indices: &mut TensorShape) -> Result<(), String> {
        let mut idx = indices.len();
        let mut count = 0;
        let mut nelems: usize = 1;

        for (i, &value) in indices.iter().enumerate() {
            if value < 0 {
                idx = i;
                count += 1;
            } else {
                nelems *= value as usize;
            }
        }

        if count > 1 {
            return Err("can only specify one unknown dimension!".to_string());
        } else if count == 1 {
            if nelems == 0 || !self.nelems.is_multiple_of(nelems) {
                let info = format!(
                    "cannot reshape tensor of shape {} into shape {}",
                    self.shape_as_string(),
                    shape_as_string(indices)
                );
                return Err(info);
            }
            indices[idx] = (self.nelems / nelems) as Index;
            nelems *= indices[idx] as usize;
        }

        if nelems != self.nelems {
            let info = format!(
                "cannot reshape tensor of shape {} into shape {}",
                self.shape_as_string(),
                shape_as_string(indices)
            );
            return Err(info);
        }

        nelems = 1;
        self.dims = indices.clone();
        self.strides.resize(self.dims.len(), 0);
        for i in (0..self.dims.len()).rev() {
            self.strides[i] = nelems as Index;
            nelems *= self.dims[i] as usize;
        }
        Ok(())
    }

    pub fn index_inplace(&mut self, mut index: Index, axis: usize) {
        nnops_check!(
            !(index >= self.dims[axis] || index < -self.dims[axis]),
            "index {} is out of bounds for axis {} with size {}",
            index, axis, self.dims[axis]
        );

        if index < 0 {
            index += self.dims[axis];
        }

        self.offset += index * self.strides[axis];
        self.nelems /= self.dims[axis] as usize;

        for i in axis..self.dims.len() - 1 {
            self.dims[i] = self.dims[i + 1];
            self.strides[i] = self.strides[i + 1];
        }
        self.dims.pop();
        self.strides.pop();
    }

    pub fn slice_inplace(&mut self, slice: &Slice, axis: usize) {
        let start = slice.start.unwrap_or(0);
        let step = slice.step.unwrap_or(1);
        let stop = slice.stop.unwrap_or_else(|| {
            if step > 0 { self.dims[axis] } else { -1 }
        });

        self.offset += start * self.strides[axis];
        self.nelems /= self.dims[axis] as usize;
        self.strides[axis] *= step;

        if (start < stop) && (step > 0) {
            self.dims[axis] = (stop - start - 1) / step + 1;
        } else if (start > stop) && (step < 0) {
            self.dims[axis] = (start - stop - 1) / (-step) + 1;
        } else {
            self.dims[axis] = 0;
        }

        self.nelems *= self.dims[axis] as usize;
    }
}