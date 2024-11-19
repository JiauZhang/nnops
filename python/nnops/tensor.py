from nnops._C import (
    Tensor as __Tensor,
    from_numpy,
    is_broadcastable, broadcast_shape,
)
import nnops.dtype as DT_

class Tensor(__Tensor):
    def __init__(self, *, dtype=DT_.float32, shape=[], device='cpu'):
        super().__init__(dtype, shape, device)

del __Tensor

__tensor = Tensor(shape=[1])
__tensor.__init_pytensor_type()
del __tensor
