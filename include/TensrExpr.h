#pragma once 
#include "Tensr.h"


template<typename T, DeviceType Device>
class TensrExpr {
public:
    virtual ~TensrExpr() = default;

    virtual Tensr<T, Device> eval() = 0;
    virtual std::vector<size_t> shape() = 0;
    virtual std::shared_ptr<TensrExpr<T, Device>> clone() = 0;  

};


template<typename T, DeviceType Device>
class ConstExpr : public TensrExpr<T, Device> {
private:
    Tensr<T, Device> value_;
public:

    ConstExpr(Tensr<T, Device>& t) : value_(t) {}

    Tensr<T, Device> eval() override { return value_; }
    std::vector<size_t> shape() override { return value_.shape(); }
    std::shared_ptr<TensrExpr<T, Device>> clone() override { 
        return std::make_shared<ConstExpr<T, Device>>(*this);
    }
};  