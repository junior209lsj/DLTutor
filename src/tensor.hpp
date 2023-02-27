#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <cstddef>
#include <stdexcept>
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

  // template<typename U>
  //   friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

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

template<typename T>
Tensor<T>::Tensor() : size_(0), ndim_(0) {}

template<typename T>
Tensor<T>::Tensor(const std::initializer_list<std::size_t>& shape)
    : size_(0), ndim_(shape.size()), shape_(new std::size_t[ndim_]) {
  size_ = 1;
  std::size_t i = 0;
  for (auto dim : shape) {
    size_ *= dim;
    shape_[i++] = dim;
  }
  data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& other)
    : size_(other.size_), ndim_(other.ndim_), shape_(new std::size_t[ndim_]) {
  std::copy(other.shape_.get(), other.shape_.get() + ndim_, shape_.get());
  data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
  std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
}

template<typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
    : size_(std::exchange(other.size_, 0)), ndim_(std::exchange(other.ndim_, 0)),
      data_(std::move(other.data_)), shape_(std::move(other.shape_)) {}

template<typename T>
Tensor<T>::~Tensor() {}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
  if (this != &other) {
    ndim_ = other.ndim_;
    size_ = other.size_;
    shape_.reset(new std::size_t[ndim_]);
    std::copy(other.shape_.get(), other.shape_.get() + ndim_, shape_.get());
    data_.reset(new T[size_]);
    std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
  }
  return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
  if (this != &other) {
    ndim_ = std::exchange(other.ndim_, 0);
    size_ = std::exchange(other.size_, 0);
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
  }
  return *this;
}

// Accessors

template<typename T>
T& Tensor<T>::operator[](std::size_t idx) {
  return data_.get()[idx];
}

template<typename T>
const T& Tensor<T>::operator[](std::size_t idx) const {
  return data_.get()[idx];
}

template<typename T>
T& Tensor<T>::operator()(const std::initializer_list<std::size_t>& indices) {
  std::size_t idx = ComputeIndex(indices);
  return data_.get()[idx];
}

template<typename T>
const T& Tensor<T>::operator()(const std::initializer_list<std::size_t>& indices) const {
  std::size_t idx = ComputeIndex(indices);
  return data_.get()[idx];
}

template<typename T>
std::size_t Tensor<T>::ComputeIndex(const std::initializer_list<std::size_t>& indices) const
{
    std::size_t index = 0;
    std::size_t offset = 1;
    auto indices_it = indices.begin();

    for (std::size_t i = 0; i < ndim_; ++i)
    {
        if (indices_it == indices.end())
        {
            throw std::out_of_range("Index out of range");
        }
        std::size_t dim = *indices_it++;
        if (dim >= shape_[i])
        {
            throw std::out_of_range("Index out of range");
        }
        index += offset * dim;
        offset *= shape_[i];
    }

    return index;
}

}  // namespace dltu
#endif