#include <nnops/device.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using nnops::Device, nnops::DeviceType;

namespace pynnops {

void show_device_info(DeviceType device) {
    Device *dev = Device::get_device(device);
    if (dev == nullptr)
        throw std::runtime_error("invalid device type!");
    dev->info();
}

void DEFINE_DEVICE_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("NPU", DeviceType::NPU)
        .export_values();

    m.def("show_device_info", &show_device_info);
}

} // namespace pynnops