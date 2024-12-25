from nnops.tensor import Tensor
from nnops import device

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
    def test_cpu_to_cuda(self):
        t_cpu = Tensor(shape=[3, 5, 7], device=device.CPU)
        t_cuda = t_cpu.to(device.CUDA)
        assert t_cuda.device == device.CUDA
