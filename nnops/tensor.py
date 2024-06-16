from nnops._C import (
    TensorShape as __TensorShape,
    Tensor as __Tensor,
)
import nnops.dtype as DT_

class TensorShape(__TensorShape):
    def __init__(self, *, shape=[]):
        super().__init__(shape)

del __TensorShape

class Tensor(__Tensor):
    def __init__(self, *, dtype=DT_.float32, shape=[], device='cpu'):
        super().__init__(dtype, shape, device)

    def __getitem__(self, key):
        ...

del __Tensor