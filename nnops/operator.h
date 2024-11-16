#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include <map>
#include <string.h>
#include <nnops/tensor.h>
#include <nnops/device.h>

namespace nnops {

class Operator {
public:
    static void register_operator(std::string &op_name, DeviceType type, Operator *op);
    static Operator *get_operator(std::string &op_name, DeviceType type);

    virtual void operator() = 0;
};

} // namespace nnops

#endif // __OPERATOR_H__