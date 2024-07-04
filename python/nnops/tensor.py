from nnops._C import (
    Tensor as __Tensor,
)
import nnops.dtype as DT_

class Tensor(__Tensor):
    def __init__(self, *, dtype=DT_.float32, shape=[], device='cpu'):
        super().__init__(dtype, shape, device)

    def __getitem__(self, index):
        return self.getitem(index)

del __Tensor