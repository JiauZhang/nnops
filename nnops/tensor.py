from nnops._C import (
    TensorShape as __TensorShape
)

class TensorShape(__TensorShape):
    def __init__(self, shape=[]):
        super().__init__()
        self.set_dims(shape)

del __TensorShape