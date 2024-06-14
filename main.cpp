//
// ~thwmakos~
//
// 13/6/2024
//

#include <iostream>
#include <array>
#include <cmath>

#include "matrix.hpp"

using namespace thwmakos;

// neural network layers determined at compile time
// input and output layers are first and last elements
// of the array, so they are included here
constexpr std::array<FloatType, 3> network_sizes = {28 * 28, 30, 10};

struct Network
{
	std::array<std::vector<FloatType>, network_sizes.size()> m_biases;
};

void test_matrix()
{
	Matrix mat1 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
	Matrix mat2 = mat1;
	mat2.at(1, 1) = 10.0f;

	Matrix mat3(-4, 3);
	
	std::cout << mat1 << '\n';
	std::cout << mat2 << '\n';
	std::cout << mat3 << '\n';
}

int main()
{
    std::cout << "test \n";
	std::cout << network_sizes[0] << "\n";
	test_matrix();
    return 0;
}
