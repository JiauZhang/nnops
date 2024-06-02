from nnops._C import (
    TensorShape as __TensorShape,
    DataType as __DataType,
    Tensor as __Tensor,
)
import nnops.dtype as DT_

class TensorShape(__TensorShape):
    def __init__(self, shape=[]):
        super().__init__()
        self.set_dims(shape)

del __TensorShape

class Tensor(__Tensor):
    def __init__(self, dtype=DT_.float32, shape=[], device=None):
        super().__init__(dtype, shape)

del __Tensor