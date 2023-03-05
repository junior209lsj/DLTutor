#include <iostream>
#include <tensor.hpp>

#include <gtest/gtest.h>

TEST(Tensor, index_computation) {
  dltu::Tensor<float> t({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_EQ(1, t({0, 0, 0}));
  EXPECT_EQ(2, t({0, 0, 1}));
  EXPECT_EQ(3, t({0, 1, 0}));
  EXPECT_EQ(4, t({0, 1, 1}));
  EXPECT_EQ(5, t({1, 0, 0}));
  EXPECT_EQ(6, t({1, 0, 1}));
  EXPECT_EQ(7, t({1, 1, 0}));
  EXPECT_EQ(8, t({1, 1, 1}));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}