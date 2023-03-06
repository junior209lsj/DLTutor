#include <iostream>
#include <random>
#include <tensor.hpp>
#include <tensor_ops.hpp>

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

TEST(Tensor, transpose_1d) {
    dltu::Tensor<float> t({4}, {2, 4, 6, 8});
    dltu::Tensor<float> tt = t.Transpose();
    EXPECT_EQ(2, tt({0}));
    EXPECT_EQ(4, tt({1}));
    EXPECT_EQ(6, tt({2}));
    EXPECT_EQ(8, tt({3}));
}

TEST(Tensor, transpose_2d) {
    dltu::Tensor<float> t({17, 21});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (std::size_t i = 0; i < 17; i++) {
        for (std::size_t j = 0; j < 21; j++) {
            t({i, j}) = dis(gen);
        }
    }

    dltu::Tensor<float> tt = t.Transpose();

    for (std::size_t i = 0; i < 17; i++) {
        for (std::size_t j = 0; j < 21; j++) {
            EXPECT_FLOAT_EQ(t({i, j}), tt({j, i}));
        }
    }
}

TEST(Tensor, add) {
    dltu::Tensor<float> ta({17, 21});
    dltu::Tensor<float> tb({17, 21});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (std::size_t i = 0; i < 17; i++) {
        for (std::size_t j = 0; j < 21; j++) {
            ta({i, j}) = dis(gen);
            tb({i, j}) = dis(gen);
        }
    }

    dltu::Tensor<float> tc = dltu::ops::add(ta, tb);

    for (std::size_t i = 0; i < 17; i++) {
        for (std::size_t j = 0; j < 21; j++) {
            EXPECT_FLOAT_EQ(tc({i, j}), ta({i, j}) + tb({i, j}));
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}