// ~thwmakos~
//
// matrix_multiply_avx_tests.cpp
//
// 27/9/2024
//

#include <doctest/doctest.h>

#include <random>
#include <print> // we have access to <print> in gcc 14!
#include <chrono>

#include "matrix.hpp"
#include "matrix_multiply_avx.hpp"

using thwmakos::matrix;
using thwmakos::FloatType;

// do not stringify matrix in d
// provide dummy toString
//namespace thwmakos {
//	doctest::String toString(const matrix &value)
//	{
//	return "intentionally empty";
//	}
//}

#ifdef THWMAKOS_NDEBUG
constexpr bool release_build = true;
#else
constexpr bool release_build = false;
#endif

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
	constexpr int n = 8;
	constexpr int m = 4 * 16;
	constexpr int scale = 10;
	constexpr int middle = 300;

	matrix A(n * scale, middle);
	matrix B(middle, m * scale);

	randomise(A);
	randomise(B);
	
	CHECK(multiply(A, B) == multiply_avx512(A, B));

	// test speedup in optimised build
	if constexpr(release_build)
	{
		matrix A_large(n * 50 * scale, 2 * middle); 
		matrix B_large(2 * middle, m * 10 * scale); 
		auto t1 = std::chrono::high_resolution_clock::now();	
		matrix C1 = multiply(A_large, B_large);
		auto t2 = std::chrono::high_resolution_clock::now();
		matrix C2 = multiply_avx512(A_large, B_large);
		auto t3 = std::chrono::high_resolution_clock::now();

		std::println("A dimensions: ({}, {}), B dimensions: ({}, {})", A_large.num_rows(), A_large.num_cols(), B_large.num_rows(), B_large.num_cols());
		std::println("Naive multiply with transpose took {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));
		std::println("AVX512 multiply took {}", std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2));
		std::println("AVX512 was {} times faster", 
				std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 
				std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());
	}
}
