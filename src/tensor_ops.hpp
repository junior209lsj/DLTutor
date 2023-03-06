#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H
#include <tensor.hpp>

namespace dltu {
namespace ops {

template <typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T> &b) {
    if (a.Size() != b.Size()) {
        throw std::invalid_argument("Tensors' size do not match");
    }
    Tensor<T> result(a);

    for(std::size_t i = 0; i < a.Size(); i++) {
        result[i] += b[i];
    }

    return result;
}

}
} // namespace dltu

#endif