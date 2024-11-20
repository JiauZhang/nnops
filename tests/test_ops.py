import nnops.ops as ops
import nnops.tensor
import numpy as np

class TestDeviceType():
    def test_add(self):
        np_a = np.random.randn(3, 1, 4).astype(np.float32)
        np_b = np.random.randn(2, 1).astype(np.float32)
        np_c = np.random.randn(5, 4).astype(np.float32)
        t_a = nnops.tensor.from_numpy(np_a)
        t_b = nnops.tensor.from_numpy(np_b)
        t_c = nnops.tensor.from_numpy(np_c)
        assert (ops.add(t_a, t_b).numpy() == np_a + np_b).all()
        assert (ops.add(t_a, t_c).numpy() == np_a + np_c).all()
