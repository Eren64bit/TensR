#pragma once 
#include "Tensr.h"
#include <functional>


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

template<typename T, DeviceType Device>
class BinaryExpr : public TensrExpr<T, Device> {
private:
    std::shared_ptr<TensrExpr<T, Device>> lhs_;
    std::shared_ptr<TensrExpr<T, Device>> rhs_;
    std::function<T(T, T)> op_;
public:
    BinaryExpr(std::shared_ptr<TensrExpr<T, Device>> lhs, std::shared_ptr<TensrExpr<T, Device>> rhs, std::function<T(T, T)> op)
        : lhs_(lhs), rhs_(rhs), op_(op) {}

    Tensr<T, Device> eval() override {
        auto lval = lhs_.eval();
        auto rval = rhs_->eval();
        //return 
    }
};