#include <iostream>
#include <tensor.hpp>

int main(void) {

    dltu::Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});

    for(size_t i = 0; i < 2; i++) {
        for(size_t j = 0; j < 3; j++) {
            std::cout << t({i, j}) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}