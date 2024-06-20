//
// ~thwmakos~
//
// tests.cpp
//
// Wed 19 Jun 21:01:28 BST 2024
//

// all tests for the neural network implementation
// are in this file

#include <doctest/doctest.h>

#include <iostream>

#include "matrix.hpp"
#include "network.hpp"

using thwmakos::FloatType;
using thwmakos::matrix;
using thwmakos::network;

// try various tests on the Matrix class
TEST_CASE("testing matrix class")
{
	matrix mat1 {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
	matrix mat2 = mat1;

	CHECK(mat1 == mat2);

	mat2(1, 1) = 10.0f;

	CHECK(mat2(1, 1) == 10.0f);
		
	matrix mat3(4, 3);
		
	CHECK(mat1 != mat3);

	REQUIRE_THROWS(mat1 = matrix {{2.0f, 3.0f}, {1.0f}});

	// test multiplication with two random matrices
	matrix left {{1.67,6.41,1.26,6.1},{2.31,3.75,10.1,1.7},{-3.3,-4.7,0.3,9.1},{0.7,-1.5,-3.8,1.5}};
	matrix right {{6.1,-1.31,4.6,-3.1},{9.01,1.4,-6.4,3.1},{2.7,1.8,1.8,-2.6},{-0.5,1.2,6.1,-4.1}};
	matrix expected_product {{68.29310000000001,16.3743,6.135999999999996,-13.592},
		{74.2985,22.4439,15.176,-28.766},
		{-66.217,9.203,70.95000000000001,-42.42999999999999},
		{-20.255,-8.057,15.13,-3.090000000000001}};

	
	REQUIRE_NOTHROW(multiply(left, right));
	auto product = multiply(left, right);	

	CHECK(product == expected_product);	
	CHECK(left * right == product);

	REQUIRE_THROWS_AS(multiply(mat1, mat3), const std::invalid_argument&);

	mat2.set_size(10, 16);
	CHECK(mat2.size() == std::make_pair(10, 16));

	matrix mat4;
	CHECK(mat4.num_cols() == 0);
}

TEST_CASE("testing network class")
{
	network nwk;

	std::cout << "biases[1] \n" << nwk.m_biases[1].size().first << ' ' << nwk.m_biases[1].size().second << '\n';
		
	CHECK(nwk.m_weights[0].size() == std::make_pair(30, 28 * 28));
	CHECK(nwk.m_weights[1].size() == std::make_pair(10, 30));
	CHECK(nwk.m_biases[0].size() == std::make_pair(30, 1));
	
	matrix input_vector(28 * 28, 1);

	REQUIRE_NOTHROW(nwk.evaluate(input_vector));
	
	auto result = nwk.evaluate(input_vector);

	CHECK(result.size() == std::make_pair(10, 1));
}
