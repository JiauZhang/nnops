from nnops.tensor import Tensor, from_numpy
from nnops import device, dtype
import numpy as np

class TestDeviceType():
    def test_device_type(self):
        dev_types = {
            'cpu': device.CPU,
            'cuda': device.CUDA,
        }
        for device_name, device_type in dev_types.items():
            t_a = Tensor(shape=[1], device=device_name)
            t_b = Tensor(shape=[1], device=device_type)
            assert t_a.device == device_type and t_b.device == device_type

class TestDeviceTo():
    def test_cpu_and_cpu(self):
        a_cpu = from_numpy(np.random.randn(3, 6, 9).astype(np.float32))
        b_cpu = a_cpu.to(device.CPU)
        assert a_cpu.dtype == dtype.float32 and a_cpu.device == device.CPU
        assert b_cpu.dtype == a_cpu.dtype and b_cpu.device == a_cpu.device
        assert (a_cpu.numpy() == b_cpu.numpy()).all()

    def test_cpu_and_cuda(self):
        a_cpu = from_numpy(np.random.randn(2, 5, 8).astype(np.float32))
        a_cuda = a_cpu.to(device.CUDA)
        assert a_cuda.device == device.CUDA

        b_cpu = a_cuda.to(device.CPU)
        assert b_cpu.device == a_cpu.device and b_cpu.dtype == a_cpu.dtype
        assert (a_cpu.numpy() == b_cpu.numpy()).all()