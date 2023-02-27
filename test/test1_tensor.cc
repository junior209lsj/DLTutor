#include <iostream>
#include <tensor.hpp>

int main(void) {

    dltu::Tensor<float> t({10, 10});

    for(size_t i = 0; i < 10; i++) {
        for(size_t j = 0; j < 10; j++) {
            std::cout << t({i, j}) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}