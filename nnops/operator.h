#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include <map>
#include <string.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

namespace nnops {

class Operator {
public:
    void set_operator_name(std::string &op_name);
    inline std::string get_operator_name() { return std::string(operator_name_); }
    inline void set_device_type(DeviceType &dt) { device_type_ = dt; }
    inline DeviceType get_device_type() { return device_type_; }

    static void register_operator(std::string &op_name, DeviceType type, Operator *op);
    static Operator *get_operator(std::string &op_name, DeviceType type);

private:
    static std::map<std::string, Operator *> operators_[DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES];
    const char *operator_name_ = nullptr;
    DeviceType device_type_;
};

} // namespace nnops

#endif // __OPERATOR_H__