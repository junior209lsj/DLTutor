#include <iostream>
#include <tensor.h>

int main(void) {

    dltu::Tensor<float> t({10, 10});

    std::cout << t << std::endl;

    std::cout << t[0] << std::endl;

    return 0;
}