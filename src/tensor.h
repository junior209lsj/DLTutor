#pragma once
#include <cstddef>
#include <memory>

namespace dltu {

template<typename T>
class Tensor {
 public:
  // Constructors
  Tensor();
  Tensor(const std::initializer_list<std::size_t>& shape);
  Tensor(const Tensor<T>& other);
  Tensor(Tensor<T>&& other) noexcept;

  // Destructor
  ~Tensor();

  // Assignment operator
  Tensor<T>& operator=(const Tensor<T>& other);
  Tensor<T>& operator=(Tensor<T>&& other) noexcept;

  // Accessors
  T& operator[](std::size_t idx);
  const T& operator[](std::size_t idx) const;
  T& operator()(const std::initializer_list<std::size_t>& indices);
  const T& operator()(const std::initializer_list<std::size_t>& indices) const;

  // Methods
  Tensor<T> Transpose() const;

 private:
  // Private member variables
  std::shared_ptr<T> data_;
  std::size_t size_;
  std::size_t ndim_;
  std::unique_ptr<std::size_t[]> shape_;

  // Private helper functions
  std::size_t ComputeIndex(const std::initializer_list<std::size_t>& indices) const;
};

}  // namespace dltu
