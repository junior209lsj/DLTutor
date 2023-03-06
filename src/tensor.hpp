#ifndef TENSOR_H
#define TENSOR_H
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace dltu {

template <typename T>
class Tensor {
 public:
  // Constructors
  Tensor();
  Tensor(const std::initializer_list<std::size_t> &shape)
      : Tensor(std::vector<std::size_t>(shape)) {}
  Tensor(const std::vector<std::size_t> &shape);

  Tensor(const std::initializer_list<std::size_t> &shape,
         const std::initializer_list<T> &data)
      : Tensor(std::vector<std::size_t>(shape), std::vector<T>(data)) {}
  Tensor(const std::vector<std::size_t> &shape,
         const std::initializer_list<T> &data)
      : Tensor(shape, std::vector<T>(data)) {}
  Tensor(const std::initializer_list<std::size_t> &shape,
         const std::vector<T> &data)
      : Tensor(std::vector<std::size_t>(shape), data) {}
  Tensor(const std::vector<std::size_t> &shape, const std::vector<T> &data);

  Tensor(const Tensor<T> &other);
  Tensor(Tensor<T> &&other) noexcept;

  // Destructor
  ~Tensor();

  // Assignment operator
  Tensor<T> &operator=(const Tensor<T> &other);
  Tensor<T> &operator=(Tensor<T> &&other) noexcept;

  // Accessors
  T &operator[](std::size_t idx);
  const T &operator[](std::size_t idx) const;
  T &operator()(const std::initializer_list<std::size_t> &indices);
  const T &operator()(const std::initializer_list<std::size_t> &indices) const;

  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

  // Methods
  Tensor<T> Transpose();

  std::size_t Size() const;
  std::size_t Ndim() const;

 private:
  // Private member variables
  std::shared_ptr<T[]> data_;
  std::size_t size_;
  std::size_t ndim_;
  // std::unique_ptr<std::size_t[]> shape_;
  std::vector<std::size_t> shape_;

  // Private helper functions
  std::size_t ComputeIndex(
      const std::initializer_list<std::size_t> &indices) const;
};

template <typename T>
Tensor<T>::Tensor() : size_(0), ndim_(0) {}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::size_t> &shape)
    : size_(std::accumulate(shape.begin(), shape.end(), 1,
                            std::multiplies<std::size_t>())),
      ndim_(shape.size()),
      shape_(shape) {
  data_ = std::shared_ptr<T[]>(new T[size_], std::default_delete<T[]>());
}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::size_t> &shape,
                  const std::vector<T> &data)
    : size_(std::accumulate(shape.begin(), shape.end(), 1,
                            std::multiplies<std::size_t>())),
      ndim_(shape.size()),
      shape_(shape) {
  if (size_ != data.size()) {
    throw std::invalid_argument("Tensor size does not match data size");
  }

  data_ = std::shared_ptr<T[]>(new T[size_], std::default_delete<T[]>());
  std::copy(data.begin(), data.end(), data_.get());
}

// 다른 텐서를 인자로 받아서 복사 생성
template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    : size_(other.size_),
      ndim_(other.ndim_),
      shape_(std::vector<std::size_t>(other.shape_)) {
  data_ = std::shared_ptr<T[]>(new T[size_], std::default_delete<T[]>());
  std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
}

// std::move(다른 텐서)를 인자로 받아 원래있던 텐서는 비우고, 그 내용을 새로운
// 텐서에 옮김
template <typename T>
Tensor<T>::Tensor(Tensor<T> &&other) noexcept
    : data_(std::move(other.data_)),
      size_(std::exchange(other.size_, 0)),
      ndim_(std::exchange(other.ndim_, 0)),
      shape_(std::move(other.shape_)) {}

template <typename T>
Tensor<T>::~Tensor() {}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
  if (this != &other) {
    ndim_ = other.ndim_;
    size_ = other.size_;
    shape_ = other.shape_;
    data_.reset(new T[size_]);
    std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
  }
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) noexcept {
  if (this != &other) {
    ndim_ = std::exchange(other.ndim_, 0);
    size_ = std::exchange(other.size_, 0);
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
  }
  return *this;
}

// Accessors

template <typename T>
T &Tensor<T>::operator[](std::size_t idx) {
  return data_.get()[idx];
}

template <typename T>
const T &Tensor<T>::operator[](std::size_t idx) const {
  return data_.get()[idx];
}

template <typename T>
T &Tensor<T>::operator()(const std::initializer_list<std::size_t> &indices) {
  std::size_t idx = ComputeIndex(indices);
  return data_.get()[idx];
}

template <typename T>
const T &Tensor<T>::operator()(
    const std::initializer_list<std::size_t> &indices) const {
  std::size_t idx = ComputeIndex(indices);
  return data_.get()[idx];
}

template <typename T>
std::size_t Tensor<T>::ComputeIndex(
    const std::initializer_list<std::size_t> &indices) const {
  if (indices.size() != ndim_) {
    throw std::out_of_range("Invalid number of dimensions");
  }

  std::size_t idx = 0;
  std::size_t multiplier = 1;
  std::size_t dim_index = ndim_ - 1;

  for (auto it = std::rbegin(indices); it != std::rend(indices); it++) {
    if (*it >= shape_[dim_index]) {
      throw std::out_of_range("Index out of range");
    }
    idx += multiplier * (*it);
    multiplier *= shape_[dim_index--];
  }

  return idx;
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor) {
  os << "[";
  for (std::size_t i = 0; i < tensor.size_; i++) {
    os << tensor.data_[i];
    if (i != tensor.size_ - 1) {
      os << ", ";
    }
  }
  os << "]";

  return os;
}

template <typename T>
Tensor<T> Tensor<T>::Transpose() {
  if (ndim_ <= 1) {
    return *this;
  } else if (ndim_ == 2) {
    Tensor<T> result({shape_[1], shape_[0]});
    for (std::size_t i = 0; i < shape_[0]; i++) {
      for (std::size_t j = 0; j < shape_[1]; j++) {
        result({j, i}) = data_.get()[ComputeIndex({i, j})];
      }
    }

    return result;
  } else {  // 3차원 이상 텐서는 맨 마지막 2개 차원 transpose
    // TODO: 개발하기
    // return result;
    return *this;
  }
}

template <typename T>
std::size_t Tensor<T>::Size() const {
  return size_;
}

template <typename T>
std::size_t Tensor<T>::Ndim() const {
  return ndim_;
}

}  // namespace dltu
#endif