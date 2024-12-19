#include <nnops/operator.h>
#include <stdexcept>

namespace nnops {

std::map<std::string, Operator *> Operator::operators_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES];

void Operator::register_operator(std::string &op_name, DeviceType type, Operator *op) {
    auto &ops_ = Operator::operators_[type];
    if (ops_.count(op_name)) {
        Device *device = Device::get_device(type);
        std::string info = "Operator <" + op_name + "> has been registered in "
            + device->get_device_name() + " device!";
        throw std::runtime_error(info);
    }
    ops_[op_name] = op;
}

Operator *Operator::get_operator(std::string &op_name, DeviceType type) {
    auto &ops_ = Operator::operators_[type];
    if (ops_.count(op_name))
        return ops_[op_name];
    else
        return nullptr;
}

void Operator::set_operator_name(std::string &op_name) {
    DeviceType type = get_device_type();
    auto &ops_ = Operator::operators_[type];
    auto iter = ops_.find(op_name);
    if (iter != ops_.end())
        operator_name_ = iter->first.c_str();
    else
        throw std::runtime_error("set_operator_name failed!");
}

} // namespace nnops
