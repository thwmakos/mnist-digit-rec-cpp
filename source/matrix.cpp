//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

#include "matrix.hpp"
#include "matrix_avx.hpp"
#include "network.hpp"
#include "matrix_view.hpp"

#include <format>
#include <print>

namespace thwmakos {

// blocked-tiled matrix multiplication
// see matrix_avx.cpp for comments on how this works
namespace matmul
{
	// the result matrix C = A * B is split into blocks and
	// each block into is further split into tiles

	// ideally we want:
	// -- num_A_block_cols (equivalently num_B_block_rows) by num_block_rows to fit in the L3 cache
	// -- num_block_cols by num_A_block_cols to fit in the L2 cache
	// -- num_A_block_cols by num_tile_rows to fit in the L1 cache
	constexpr int num_tile_rows = 4;
	constexpr int num_tile_cols = 8;

	// a num_time_rows by num_tile_cols sized
	constexpr int num_block_rows = num_tile_rows * 32;
	constexpr int num_block_cols = num_tile_cols * 6;

	constexpr int num_A_block_cols = 256;
	constexpr int num_B_block_rows = num_A_block_cols;
}

template<int num_tile_rows, int num_tile_cols>
	requires (num_tile_rows > 0 && num_tile_cols > 0)
void multiply_tile(matrix_view C, const_matrix_view A, const_matrix_view B)
{
	for(int i = 0; i < A.num_rows; ++i)
	{
		for(int j = 0; j < B.num_cols; ++j)
		{
			FloatType accumulator = 0.0;

			for(int k = 0; k < A.num_cols; ++k)
			{
				accumulator += A[i, k] * B[k, j];
			}

			C[i, j] += accumulator;
		}
	}
}

// this is implemented in matrix_avx.cpp
void load_block(const_matrix_view source, matrix_span dest);

void multiply_blocked_tiled(matrix_span C, const_matrix_span A, const_matrix_span B)
{
	// we want this to fit in L2 cache
	// no need to zero-initialise as we zero the relevant entries in load_block
	static std::array<FloatType, matmul::num_block_rows * matmul::num_A_block_cols> A_cache_storage;
	// we want this to fit in L3 cache
	static std::array<FloatType, matmul::num_B_block_rows * matmul::num_block_cols> B_cache_storage;

	matrix_span A_cache { matmul::num_block_rows, matmul::num_A_block_cols, A_cache_storage };
	matrix_span B_cache { matmul::num_B_block_rows, matmul::num_block_cols, B_cache_storage };

	for(int n = 0; n < C.num_columns; n += matmul::num_block_cols)
	{
		const int block_num_cols = std::min(matmul::num_block_cols, C.num_columns - n);

		for(int k = 0; k < B.num_rows; k += matmul::num_B_block_rows)
		{
			const int block_middle_num = std::min(matmul::num_B_block_rows, B.num_rows - k);

			const_matrix_view B_block { B, k, n, block_middle_num, block_num_cols };

			B_cache.num_rows    = block_middle_num;
			B_cache.num_columns = block_num_cols;

			load_block(B_block, B_cache);

			for(int m = 0; m < C.num_rows; m += matmul::num_block_rows)
			{
				const int block_num_rows = std::min(matmul::num_block_rows, C.num_rows - m);

				const_matrix_view A_block { A, m, k, block_num_rows, block_middle_num };

				A_cache.num_rows    = block_num_rows;
				A_cache.num_columns = block_middle_num;

				load_block(A_block, A_cache);

				for(int tile_n = 0; tile_n < block_num_cols; tile_n += matmul::num_tile_cols)
				{
					const int num_tile_cols = std::min(matmul::num_tile_cols, block_num_cols - tile_n);

					for(int tile_m = 0; tile_m < block_num_rows; tile_m += matmul::num_tile_rows)
					{
						const int num_tile_rows = std::min(matmul::num_tile_rows, block_num_rows - tile_m);

						matrix_view C_tile { C, m + tile_m, n + tile_n, num_tile_rows, num_tile_cols };
						const_matrix_view A_tile { A_cache, tile_m, 0, num_tile_rows, block_middle_num };
						const_matrix_view B_tile { B_cache, 0, tile_n, block_middle_num, num_tile_cols };

						multiply_tile<matmul::num_tile_rows, matmul::num_tile_cols>(C_tile, A_tile, B_tile);
					}
				}
			}
		}
	}
}

void multiply(matrix_span product, const_matrix_span left, const_matrix_span right)
{
	// no need to check for dimensions, was already done in caller
	// dispatch to appropriate implementation
#ifdef __AVX512F__
	multiply_avx512_blocked_tiled(product, left, right);
#elifdef __AVX2__
	//multiply_avx2(product, left, right);
	multiply_avx2_blocked_tiled(product, left, right);
#else
	multiply_naive(product, left, right);
#endif
}

void multiply_naive(matrix_span product, const_matrix_span left, const_matrix_span right)
{	
	// naive implementation of matrix multiplication
	
	// transpose right first to ensure sequential access 
	// to matrix elements
	// uses more memory but is ~4 times faster for large matrices
	std::vector<FloatType> right_transpose_storage(right.num_rows * right.num_columns);
	matrix_span right_transpose { right.num_columns, right.num_rows, right_transpose_storage };

	transpose(right_transpose, right);
	
	for(auto i = 0; i < left.num_rows; ++i)
	{
		for(auto j = 0; j < right.num_columns; ++j)
		{
			// equivalently we could check against right_num_rows
			FloatType accumulator {}; 
			
			for(auto k = 0; k < left.num_columns; ++k)
			{
				accumulator += left[i, k] * right_transpose[j, k];
			}

			product[i, j] = accumulator; 
		}
	}
}

void scalar_multiply(matrix_span mat, FloatType scalar)
{
#ifdef AVX512F
	scalar_multiply_avx512(mat, scalar);
#elif AVX2
	scalar_multiply_avx2(mat, scalar);
#else
	for(auto row = 0; row < mat.num_rows; ++row)
	{
		for(auto col = 0; col < mat.num_columns; ++col)
		{
			mat[row, col] *= scalar;
		}
	}
#endif
}

// TODO: this can be slow, need to make this cache friendly
// by splitting the matrix in blocks and transposing accordingly
void transpose(matrix_span result, const_matrix_span mat)
{
	for(auto i = 0; i < result.num_rows; ++i)
	{
		for(auto j = 0; j < result.num_columns; ++j)
		{
			result[i, j] = mat[j, i];
		}
	}
}


// does left += right
void add_to(matrix_span left, const_matrix_span right, FloatType scalar)
{
#ifdef AVX512F
	add_to_avx512(left, right, scalar);
#else

	for(auto i = 0; i < left.num_rows; ++i)
	{
		for(auto j = 0; j < left.num_columns; ++j)
		{
			left[i, j] += scalar * right[i, j];
		}
	}

#endif
}

// multiply each element of left by right,
// left[i, j] *= right[i, j]
void elementwise_multiply_by(matrix_span left, const_matrix_span right)
{
	const auto [num_rows, num_cols] = left.size();
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			left[i, j] *= right[i, j];
		}
	}
}

void add_column_to(matrix_span mat, const_matrix_span column)
{
	for(auto i = 0; i < mat.num_rows; ++i)
	{
		for(auto j = 0; j < mat.num_columns; ++j)
		{
			// FIXME: This should be += but -= gives stable and accurate results????
			// It is because I made a mistake in network::evaluate() with bias sign
			mat[i, j] += column.data[i];
		}
	}
}

bool is_equal(const_matrix_span left, const_matrix_span right)
{
	constexpr FloatType abs_eps = 1.0e-4; // for comparison with values near zero
	constexpr FloatType rel_eps = 1.0e-4; // for comparison 'away' from zero

	if(left.num_rows != right.num_rows ||
	   left.num_columns != right.num_columns)
	{
		return false;
	}

	for(int row = 0; row < left.num_rows; ++row)	
	{
		for(int col = 0; col < left.num_columns; ++col)
		{
			// if two elements are equal
			// TODO: there are better way to check float equality
			//if(left[row, col] - right[row, col] >= eps)
			//{
			//	return false;
			//}
			//if(left.at(row, col) != right.at(row, col))
			//	return false;

			FloatType a = left[row, col];
			FloatType b = right[row, col];

			FloatType abs_diff = std::abs(a - b);

			if(abs_diff <= abs_eps)
			{
				continue;
			}

			FloatType abs_a   = std::abs(a);
			FloatType abs_b   = std::abs(b);	
			FloatType largest = std::max(abs_a, abs_b);

			if(abs_diff / largest > rel_eps)
			{
				return false;
			}
		}
	}

	return true;
}

} // namespace thwmakos
