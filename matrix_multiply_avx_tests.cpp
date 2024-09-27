// ~thwmakos~
//
// matrix_multiply_avx_tests.cpp
//
// 27/9/2024
//

#include <doctest/doctest.h>

#include <random>
#include <print> // we have access to <print> in gcc 14!

#include "matrix.hpp"
#include "matrix_multiply_avx.hpp"

using thwmakos::matrix;
using thwmakos::FloatType;

// do not stringify matrix in doctest assertions,
// provide dummy toString
namespace thwmakos {
	doctest::String toString(const matrix &value)
	{
		return "internationally empty";
	}
}

TEST_CASE("test AVX512 matrix multiplication")
{
	// borrowed code from network.cpp to initialise 
	// a matrix with random values
	std::random_device rd {}; 
	std::default_random_engine eng { rd() };
	std::normal_distribution<FloatType> normal(0.0f, 1.0f);
	
	// lambda to randomise an sequence of matrices using 
	// the random distribution constructed above
	auto randomise = [&normal, &eng] (auto& mat)
	{
		const auto [num_rows, num_cols] = mat.size();

		for(auto i = 0; i < num_rows; ++i)
		{
			for(auto j = 0; j < num_cols; ++j)	
			{
				mat[i, j] = normal(eng);
			}
		}
	};
	
	// controls the size of matrices we multiply
	constexpr int n = 4;
	constexpr int m = 32;
	constexpr int scale = 3;

	matrix A((n * m) * (scale - 2), n * m * scale);
	matrix B(n * m * scale, (n * m) * (scale - 1));

	randomise(A);
	randomise(B);
	
	matrix C1 = multiply(A, B);
	matrix C2 = multiply_avx512(A, B);

	std::println("A dimensions: ({}, {}), B dimensions: ({}, {})", A.num_rows(), A.num_cols(), B.num_rows(), B.num_cols());

	CHECK(C1 == C2);
}
