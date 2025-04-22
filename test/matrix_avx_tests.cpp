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

#include "../source/matrix.hpp"
#include "../source/matrix_avx.hpp"

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

auto multiply_helper(const matrix &left, const matrix &right, auto &&func)
{
	if(left.num_cols() != right.num_rows())
	{
		throw std::invalid_argument("");
	}

	matrix prod(left.num_rows(), right.num_cols());
	
	func(prod, left, right);

	return prod;
}

// return a tuple of the result of the expression func and the duration,
// in milliseconds the execution took
auto time_function_call(auto &&func)
{
	auto start    = std::chrono::high_resolution_clock::now();
	auto result   = (func)();
	auto end      = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

	return std::tuple { result, duration };
}

// fill a matrix with random values
void randomise(matrix &mat)
{
	// borrowed code from network.cpp to initialise
	// a matrix with random values
	static std::random_device rd {};
	static std::default_random_engine eng { rd() };
	static std::uniform_real_distribution<FloatType> uni(-10.0f, 10.0f);

	// lambda to randomise an sequence of matrices using
	// the random distribution constructed above
	const auto [num_rows, num_cols] = mat.size();

	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			mat[i, j] = uni(eng);
		}
	}
}

TEST_CASE("test AVX512 add_to")
{
	matrix m1(63, 41);
	matrix m2(63, 41);

	randomise(m1);
	randomise(m2);

	matrix m1_copy = m1;

	matrix m3 { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };
	matrix m4 { { 2.0, 2.0, 2.0 }, { 2.0, 2.0, 2.0 } };
	matrix m5 = m4;

	add_to(m4, m3);
#ifdef __AVX512F__
	add_to_avx512(m5, m3);
#endif

	m1 += m2;
	add_to(m1_copy, m2);
	CHECK(m1 == m1_copy);
#ifdef __AVX512F__
	CHECK(m4 == m5);
#endif
}

TEST_CASE("test AVX512 matrix multiplication")
{
	std::println("Release build: {}", release_build);

	auto multiply_naive_helper = [] (const matrix &A, const matrix &B) 
		{ 
			return multiply_helper(A, B, thwmakos::multiply_naive);
		};
	auto multiply_avx512_helper = [] (const matrix &A, const matrix &B) 
		{ 
			return multiply_helper(A, B, thwmakos::multiply_avx512);
		};
	auto multiply_avx2_helper = [] (const matrix &A, const matrix &B) 
		{ 
			return multiply_helper(A, B, thwmakos::multiply_avx2);
		};
	auto multiply_tiled_helper = [] (const matrix &A, const matrix &B)
		{
			return multiply_helper(A, B, thwmakos::multiply_avx512_blocked_tiled);
		};

	// test unaligned sizes that use masked version of submatrix
	for(int n : {1, 2, 3, 11, 17, 23, 31, 39})
	{
		for(int m : {1, 2, 9, 17, 33, 49, 63, 64, 77})
		{
			for(int middle : {2, 11, 16, 21, 33, 64})
			{
				matrix A(n, middle);
				matrix B(middle, m);

				randomise(A);
				randomise(B);

				auto expected = multiply_naive_helper(A, B);
#ifdef __AVX512F__
				CHECK_MESSAGE(expected == multiply_avx512_helper(A, B), std::format("avx512: dimensions: {}, {}, {}", n, middle, m));
				CHECK_MESSAGE(expected == multiply_tiled_helper(A, B), std::format("avx512: dimensions: {}, {}, {}", n, middle, m));
#endif
#ifdef __AVX2__
				CHECK_MESSAGE(expected == multiply_avx2_helper(A, B), std::format("avx2: dimensions: {}, {}, {}", n, middle, m));
#endif
			}
		}
	}
	
	// test speedup in optimised build
	if constexpr(release_build)
	{
		matrix A_large(2011, 2993);
		matrix B_large(2993, 1777);
		randomise(A_large);
		randomise(B_large);

		auto [C1, duration1] = time_function_call([&]() { return multiply_naive_helper(A_large, B_large); } );

#ifdef __AVX512F__
		auto [C2, duration2] = time_function_call([&]() { return multiply_avx512_helper(A_large, B_large); } );
		auto [Cbt, duration_bt] = time_function_call([&] () { return multiply_tiled_helper(A_large, B_large); } );
#endif
#ifdef __AVX2__
		auto [C3, duration3] = time_function_call([&]() { return multiply_avx2_helper(A_large, B_large); } );
#endif
		std::println("A dimensions: ({}, {}), B dimensions: ({}, {})", A_large.num_rows(), A_large.num_cols(), B_large.num_rows(), B_large.num_cols());
		std::println("Naive multiply with transpose took {}", duration1);
#ifdef __AVX512F__
		std::println("AVX512 multiply took {}", duration2);
		std::println("AVX512 blocked tile took {}", duration_bt);
#endif
#ifdef __AVX2__
		std::println("AVX2 multiply took {}", duration3);
#endif
#ifdef __AVX512F__
		std::println("AVX512 was {:.1f} times faster than naive", static_cast<float>(duration1.count()) / (duration2.count() != 0 ? duration2.count() : 1));
#endif
#ifdef __AVX2__
	std::println("AVX2 was {:.1f} times faster than naive", static_cast<float>(duration1.count()) / (duration3.count() != 0 ? duration3.count() : 1));
#endif
		// add one above to avoid division by zero
	}
}
