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
	std::uniform_real_distribution<FloatType> uni(-10.0f, 10.0f);
	
	// lambda to randomise an sequence of matrices using 
	// the random distribution constructed above
	auto randomise = [&uni, &eng] (auto& mat)
	{
		const auto [num_rows, num_cols] = mat.size();

		for(auto i = 0; i < num_rows; ++i)
		{
			for(auto j = 0; j < num_cols; ++j)	
			{
				mat[i, j] = uni(eng);
			}
		}
	};

	// controls the size of matrices we multiply
	constexpr int n = 8;
	constexpr int m = 4 * 16;
	
	// test various sizes
	for(int scale : {3, 4, 5, 6, 7, 8, 9, 10})
	{
		for(int middle : {31, 42, 53, 57, 64, 65, 69})
		{
			matrix A(n * scale, middle);
			matrix B(middle, m * (scale - 2));

			randomise(A);
			randomise(B);
			
			CHECK(multiply(A, B) == multiply_avx512(A, B));
		}
	}


	// test speedup in optimised build
	if constexpr(release_build)
	{
		matrix A_large(n * 250, 2000); 
		matrix B_large(2000, m * 32); 
		randomise(A_large);
		randomise(B_large);
		auto t1 = std::chrono::high_resolution_clock::now();	
		matrix C1 = multiply(A_large, B_large);
		auto t2 = std::chrono::high_resolution_clock::now();
		matrix C2 = multiply_avx512(A_large, B_large);
		auto t3 = std::chrono::high_resolution_clock::now();
		
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();


		std::println("A dimensions: ({}, {}), B dimensions: ({}, {})", A_large.num_rows(), A_large.num_cols(), B_large.num_rows(), B_large.num_cols());
		std::println("Naive multiply with transpose took {}", duration1);
		std::println("AVX512 multiply took {}", duration2);
		std::println("AVX512 was {} times faster", duration1 / (duration2 != 0 ? duration2 : 1)); 
		// add one above to avoid division by zero
																							   
	}
}
