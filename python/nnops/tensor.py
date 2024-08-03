from nnops._C import (
    Tensor as __Tensor,
)
import nnops.dtype as DT_

class Tensor(__Tensor):
    def __init__(self, *, dtype=DT_.float32, shape=[], device='cpu'):
        super().__init__(dtype, shape, device)

    def reshape(self, *dims):
        if not all([isinstance(i, int) for i in dims]):
            raise IndexError('only integers supported!')

        return super().reshape(dims)

del __Tensor