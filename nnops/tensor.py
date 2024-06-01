from nnops._C import (
    TensorShape as __TensorShape,
    DataType as __DataType,
    Tensor as __Tensor,
)

class TensorShape(__TensorShape):
    def __init__(self, shape=[]):
        super().__init__()
        self.set_dims(shape)

del __TensorShape

class DataType(__DataType):
    def __init__(self, dtype=None):
        super().__init__()

del __DataType

class Tensor(__Tensor):
    def __init__(self, dtype=None):
        super().__init__()

del __Tensor