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

#include <print>

#include "../source/matrix.hpp"
#include "../source/network.hpp"
#include "../source/data_loader.hpp"

using thwmakos::FloatType;

using thwmakos::matrix;
using thwmakos::row_vector;
using thwmakos::column_vector;

using thwmakos::network;
using thwmakos::data_loader;

constexpr std::array network_layer_size = { 28 * 28, 30, 10 };

auto multiply_helper(const matrix &left, const matrix &right)
{
	if(left.num_cols() != right.num_rows())
	{
		throw std::invalid_argument("");
	}

	matrix prod(left.num_rows(), right.num_cols());
	
	thwmakos::multiply(prod, left, right);

	return prod;
}

// try various tests on the Matrix class
TEST_CASE("testing matrix class")
{
	matrix mat1 {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
	matrix mat2 = mat1;

	CHECK(mat1 == mat2);

	mat2[1, 1] = 10.0f;

	CHECK(mat2(1, 1) == 10.0f);
		
	matrix mat3(4, 3);
		
	CHECK(mat1 != mat3);
	REQUIRE_THROWS(mat1 = matrix {{2.0f, 3.0f}, {1.0f}});
	// test transpose function
	CHECK(transpose(mat1) == matrix {{1.0f, 4.0f, 7.0f}, {2.0f, 5.0f, 8.0f}, {3.0f, 6.0f, 9.0f}});

	// test multiplication with two random matrices
	matrix left {{1.67,6.41,1.26,6.1},{2.31,3.75,10.1,1.7},{-3.3,-4.7,0.3,9.1},{0.7,-1.5,-3.8,1.5}};
	matrix right {{6.1,-1.31,4.6,-3.1},{9.01,1.4,-6.4,3.1},{2.7,1.8,1.8,-2.6},{-0.5,1.2,6.1,-4.1}};
	matrix expected_product {{68.29310000000001,16.3743,6.135999999999996,-13.592},
		{74.2985,22.4439,15.176,-28.766},
		{-66.217,9.203,70.95000000000001,-42.42999999999999},
		{-20.255,-8.057,15.13,-3.090000000000001}};

		
	REQUIRE_NOTHROW(left * right);
	
	auto product = multiply_helper(left, right);	

	CHECK(product == expected_product);	
	CHECK(left * right == product);
	CHECK(3.0f * left * right == product + product + product);

	REQUIRE_THROWS_AS(multiply_helper(mat1, mat3), const std::invalid_argument &);

	mat2.set_size(10, 16);
	CHECK(mat2.size() == std::tuple { 10, 16 });

	matrix mat4;
	CHECK(mat4.num_cols() == 0);

	std::vector<FloatType> data5 { 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f }; 
	REQUIRE_THROWS_AS(matrix(10, 10, data5), const std::invalid_argument&);

	matrix mat5(3, 2, std::move(data5));

	CHECK(mat5[0, 0] == 5.0f);
	CHECK(mat5[2, 1] == 30.0f);

	CHECK(mat5 + mat5 == 2.0f * mat5);
	CHECK(((2.0f * mat5) - mat5) == mat5);

	matrix mat6 = mat5;
	mat5 += 2.0f * mat5;
	mat5 -= 2.0f * mat6;
	CHECK(mat6 == mat5);

	matrix mat7 {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}};
	column_vector col1 {1.0f, 2.0f, 3.0f};

	CHECK(add_column(mat7, col1) == matrix {{2.0f, 3.0f, 4.0f, 5.0f}, {7.0f, 8.0f, 9.0f, 10.0f}, {12.0f, 13.0f, 14.0f, 15.0f}});

	CHECK(get_column(mat7, 1) == column_vector {2.0f, 6.0f, 10.0f});
	REQUIRE_THROWS(get_column(mat7, 4));
}

TEST_CASE("testing network class")
{
	network nwk(std::span { network_layer_size });

	//std::cout << "biases[1] \n" << nwk.m_biases[1].size().first << ' ' << nwk.m_biases[1].size().second << '\n';
		
	CHECK(nwk.m_weights[0].size() == std::tuple { 30, 28 * 28 });
	CHECK(nwk.m_weights[1].size() == std::tuple { 10, 30 });
	CHECK(nwk.m_biases[0].size() == std::tuple { 30, 1 });
	
	//std::cout << nwk.m_biases[0] << '\n';

	column_vector input_vector(28 * 28);

	REQUIRE_NOTHROW(nwk.evaluate(input_vector));
	
	auto result = nwk.evaluate(input_vector);

	CHECK(result.size() == std::tuple { 10, 1 });

	//std::cout << result << '\n';
}

TEST_CASE("testing matrix functions")
{
	matrix mat1 {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
	
	auto mat2 = elementwise_apply(mat1, [](FloatType x) { return 2.0 * x; });
	
	CHECK(mat2 == matrix {{2.0f, 4.0f},{6.0f, 8.0f},{10.0f, 12.0f}});
	
	matrix mat3 {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};

	CHECK(elementwise_multiply(mat3, mat3) == matrix {{1.0f, 4.0f}, {9.0f, 16.0f}, {25.0f, 36.0f}});
}

TEST_CASE("testing data_loader")
{
	data_loader loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");

	CHECK(loader.num_samples() == 60000);

	std::println("training image data size: {}KiB\n", static_cast<float> (loader.image_data().size() / 1024.0f));
	std::println("training label data size: {}KiB\n", static_cast<float> (loader.label_data().size() / 1024.0f));

	int label_index = 2001;
	std::println("loader.m_label_data[{}] = {}", label_index, loader.label_data()[label_index]);

	auto sample = loader.get_sample(label_index);
	CHECK(sample.label[loader.label_data()[label_index], 0] == 1.0f);
	CHECK(sample.label_val == loader.label_data()[label_index]);

}

TEST_CASE("testing network training")
{
	network nwk { network_layer_size };
	
	//std::cout << "testing backpropagation:\n";
	//std::cout << "random biases of last layer:\n";
	//std::cout << nwk.m_biases[1] << '\n';
	
	// test if dimensions check in backpropagation()
	// using a dummy sample	
	network::gradient grad { network_layer_size };
	auto input = matrix(28 * 28, 1);
	auto expected = matrix(10, 1);

	REQUIRE_NOTHROW(grad = nwk.backpropagation(input, expected));
	
	CHECK(grad.weights[0].num_rows() == nwk.m_weights[0].num_rows());
	CHECK(grad.weights[0].num_rows() == nwk.m_weights[0].num_rows());
	CHECK(grad.weights[1].num_cols() == nwk.m_weights[1].num_cols());
	CHECK(grad.weights[1].num_cols() == nwk.m_weights[1].num_cols());
	CHECK(grad.biases[0].num_rows() == nwk.m_biases[0].num_rows());
	CHECK(grad.biases[0].num_rows() == nwk.m_biases[0].num_rows());
	CHECK(grad.biases[1].num_cols() == nwk.m_biases[1].num_cols());
	CHECK(grad.biases[1].num_cols() == nwk.m_biases[1].num_cols());

	nwk.train(1, 10, 3.0f, 5.0f);
}
