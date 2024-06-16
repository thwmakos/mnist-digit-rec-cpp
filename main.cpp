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

struct network
{
	std::array<std::vector<FloatType>, network_sizes.size()> m_biases;
};

// try various tests on the Matrix class
void test_matrix()
{
	matrix mat1 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
	matrix mat2 = mat1;
	mat2.at(1, 1) = 10.0f;

	matrix mat3(4, 3);
		
	assert(mat1 == mat2);	
	assert(mat1 != mat3);

	// test multiplication with two random matrices
	matrix left ={{1.67,6.41,1.26,6.1},{2.31,3.75,10.1,1.7},{-3.3,-4.7,0.3,9.1},{0.7,-1.5,-3.8,1.5}};
	matrix right ={{6.1,-1.31,4.6,-3.1},{9.01,1.4,-6.4,3.1},{2.7,1.8,1.8,-2.6},{-0.5,1.2,6.1,-4.1}};
	matrix expected_product = {{68.29310000000001,16.3743,6.135999999999996,-13.592},
		{74.2985,22.4439,15.176,-28.766},
		{-66.217,9.203,70.95000000000001,-42.42999999999999},
		{-20.255,-8.057,15.13,-3.090000000000001}};


	auto product = multiply(left, right);	
	//std::cout << product << '\n';

	assert(product == expected_product);	

	try
	{
		multiply(mat1, mat3);
	}
	catch(std::invalid_argument& e)
	{
		std::cout << e.what() << '\n';
	}

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
