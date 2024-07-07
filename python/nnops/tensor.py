from nnops._C import (
    Tensor as __Tensor,
)
import nnops.dtype as DT_

class Tensor(__Tensor):
    def __init__(self, *, dtype=DT_.float32, shape=[], device='cpu'):
        super().__init__(dtype, shape, device)

    def __getitem__(self, index):
        index_is_valid = True
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, tuple):
            if not all([isinstance(i, int) for i in index]):
                index_is_valid = False
        else:
            index_is_valid = False

        if not index_is_valid:
            raise IndexError('only integers supported!')

        return self.getitem(index)

    def reshape(self, *dims):
        if not all([isinstance(i, int) for i in dims]):
            raise IndexError('only integers supported!')

        return super().reshape(dims)

del __Tensor