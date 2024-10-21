// ~thwmakos~
//
// matrix_avx_tests.cpp
//
// 27/9/2024
//

#include <doctest/doctest.h>

#include <random>
#include <print> // we have access to <print> in gcc 14!
#include <chrono>

#include "matrix.hpp"
#include "matrix_avx.hpp"

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

#ifdef NDEBUG
constexpr bool release_build = true;
#else
constexpr bool release_build = false;
#endif

TEST_CASE("test AVX512/AVX2 matrix multiplication")
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
	
	// test various sizes aligned to submatrix size
	for(int scale : {3, 4, 5, 6, 7, 8, 9, 10})
	{
		for(int middle : {31, 42, 53, 57, 64, 65, 69})
		{
			matrix A(n * scale, middle);
			matrix B(middle, m * (scale - 2));

			randomise(A);
			randomise(B);
			
			auto expected = multiply_naive(A, B);

			CHECK(expected == multiply_avx512(A, B));
			CHECK(expected == multiply_avx2(A, B));
		}
	}
	
	// test unaligned sizes that use masked version of submatrix
	for(int n : {2, 3, 11, 17, 23, 31, 39})
	{
		for(int m : {2, 9, 17, 33, 49, 63, 64, 77})
		{
			for(int middle : {2, 11, 16, 21})
			{
				matrix A(n, middle);
				matrix B(middle, m);

				randomise(A);
				randomise(B);

				auto expected = multiply_naive(A, B);

				CHECK_MESSAGE(expected == multiply_avx512(A, B), std::format("avx512: dimensions: {}, {}, {}", n, middle, m));
				CHECK_MESSAGE(expected == multiply_avx2(A, B), std::format("avx2: dimensions: {}, {}, {}", n, middle, m));
			}
		}
	}


	// test speedup in optimised build
	if constexpr(release_build)
	{
		matrix A_large(2011, 1994); 
		matrix B_large(1994, 777); 
		randomise(A_large);
		randomise(B_large);
		auto t1 = std::chrono::high_resolution_clock::now();	
		matrix C1 = multiply_naive(A_large, B_large);
		auto t2 = std::chrono::high_resolution_clock::now();
		matrix C2 = multiply_avx512(A_large, B_large);
		auto t3 = std::chrono::high_resolution_clock::now();
		matrix C3 = multiply_avx2(A_large, B_large);
		auto t4 = std::chrono::high_resolution_clock::now();
		
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
		auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();


		std::println("A dimensions: ({}, {}), B dimensions: ({}, {})", A_large.num_rows(), A_large.num_cols(), B_large.num_rows(), B_large.num_cols());
		std::println("Naive multiply with transpose took {}ms", duration1);
		std::println("AVX512 multiply took {}ms", duration2);
		std::println("AVX2 multiply took {}ms", duration3);
		std::println("AVX512 was {} times faster than naive", duration1 / (duration2 != 0 ? duration2 : 1)); 
		// add one above to avoid division by zero
	}
}
