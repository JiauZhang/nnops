#include <nnops/operator.h>
#include <nnops/common.h>

namespace nnops {

std::map<std::string, Operator *> Operator::operators_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES];

void Operator::register_operator(std::string &op_name, DeviceType type, Operator *op) {
    auto &ops_ = Operator::operators_[type];
    Device *device = Device::get_device(type);
    NNOPS_CHECK(!ops_.count(op_name), "Operator `%s` has been registered in %s device!", op_name.c_str(), device->get_device_cname());
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
    NNOPS_CHECK(iter != ops_.end(), "set_operator_name failed!");
    operator_name_ = iter->first.c_str();
}

} // namespace nnops
