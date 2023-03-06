#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.hpp"

namespace dltu {

template <typename T> class Tensor;

namespace ops {

std::size_t ComputeSize(const std::vector<std::size_t>& shape) {
  std::size_t result = 1;
  for(std::size_t i = 0; i < shape.size(); i++) {
    result *= shape[i];
  }
  return result;
}

std::size_t ComputeIndex(const std::vector<std::size_t>& shape,
                         const std::initializer_list<std::size_t>& indices) {
  auto ndim = shape.size();
  if (indices.size() != ndim) {
    throw std::out_of_range("Invalid number of dimensions");
  }

  std::size_t idx = 0;
  std::size_t multiplier = 1;
  std::size_t dim_index = ndim - 1;

  for (auto it = std::rbegin(indices); it != std::rend(indices); it++) {
    if (*it >= shape[dim_index]) {
      throw std::out_of_range("Index out of range");
    }
    idx += multiplier * (*it);
    multiplier *= shape[dim_index--];
  }

  return idx;
}

template <typename T>
Tensor<T> Transpose(Tensor<T>& x) {
  if (x.Ndim() == 2) {
    auto shape = x.Shape();
    Tensor<T> result({shape[1], shape[0]});
    for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
        result({j, i}) = x({i, j});
      }
    }
    return std::move_if_noexcept(result);
  } else {
    return std::move_if_noexcept(x);
  }
}

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