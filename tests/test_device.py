import pytest
from nnops.tensor import Tensor, from_numpy
from nnops import device, dtype
from conftest import devices as _devices
import numpy as np

class TestDeviceType():
    @pytest.mark.parametrize('dev', _devices)
    def test_device_type(self, dev):
        t = Tensor(shape=[1], device=dev)
        assert t.device == dev

class TestDeviceTo():
    @pytest.mark.parametrize('dev', _devices)
    def test_to(self, dev):
        a_cpu = from_numpy(np.random.randn(2, 5, 8).astype(np.float32))

        if dev != device.CPU and not dev.is_available():
            with pytest.raises(RuntimeError):
                a_cpu.to(dev)

        a_dev = a_cpu.to(dev)
        assert a_dev.device == dev
        assert a_dev.dtype == a_cpu.dtype
        assert (a_cpu.numpy() == a_dev.numpy()).all()

        b_cpu = a_dev.to(device.CPU)
        assert b_cpu.device == device.CPU
        assert b_cpu.dtype == a_cpu.dtype
        assert (a_cpu.numpy() == b_cpu.numpy()).all()

        c_dev = a_dev.to(dev)
        assert c_dev.device == dev
        assert c_dev.dtype == a_dev.dtype
        assert (a_dev.numpy() == c_dev.numpy()).all()

        a_cpu_int = from_numpy(np.random.randint(0, 100, (3, 4), dtype=np.int32))
        a_dev_int = a_cpu_int.to(dev)
        assert a_dev_int.device == dev
        assert a_dev_int.dtype == a_cpu_int.dtype
        b_cpu_int = a_dev_int.to(device.CPU)
        assert (a_cpu_int.numpy() == b_cpu_int.numpy()).all()

        a_cpu_strided = from_numpy(np.random.randn(4, 6, 8).astype(np.float32))[::2, ::2, ::2]
        a_dev_strided = a_cpu_strided.to(dev)
        assert a_dev_strided.device == dev
        b_cpu_strided = a_dev_strided.to(device.CPU)
        assert (a_cpu_strided.numpy() == b_cpu_strided.numpy()).all()
