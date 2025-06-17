#include <stdexcept>

class TensrException : public std::runtime_error {
public:
    explicit TensrException(const std::string& msg)
        : std::runtime_error("Tensor error: " + msg)  {}
};

// Specific exceptions
class TensorShapeError : public TensrException {
public:
    explicit TensorShapeError(const std::string& msg) 
        : TensrException("Shape Error: " + msg) {}
};

class TensorIndexError : public TensrException {
public:
    explicit TensorIndexError(const std::string& msg) 
        : TensrException("Index Error: " + msg) {}
};

class TensorViewError : public TensrException {
public:
    explicit TensorViewError(const std::string& msg) 
        : TensrException("View Error: " + msg) {}
};

class TensorDataTypeError : public TensrException {
public:
    explicit TensorDataTypeError(const std::string& msg) 
        : TensrException("DataType Error: " + msg) {}
};