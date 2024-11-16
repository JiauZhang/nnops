from nnops.tensor import Tensor
from nnops import device

class TestDeviceType():
    def test_device_type(self):
        dev_types = {
            'cpu': device.CPU,
        }
        for device_name, device_type in dev_types.items():
            t_a = Tensor(shape=[1], device=device_name)
            t_b = Tensor(shape=[1], device=device_type)
            assert t_a.device == device_type and t_b.device == device_type
