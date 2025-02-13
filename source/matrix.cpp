//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

#include "matrix.hpp"
#include "matrix_avx.hpp"

#include <format>
#include <print>

namespace thwmakos {

void multiply(matrix_span product, const_matrix_span left, const_matrix_span right)
{
	// no need to check for dimensions, was already done in caller
	// dispatch to appropriate implementation
#ifdef __AVX512F__
	multiply_avx512(product, left, right);
#elifdef __AVX2__
	multiply_avx2(product, left, right);
#else
	multiply_naive(product, left, right);
#endif
}

void multiply_naive(matrix_span product, const_matrix_span left, const_matrix_span right)
{	
	// naive implementation of matrix multiplication
	// TODO: good learning opportunity for intrinsics here
	
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
			mat[i, j] -= column.data[i];
		}
	}
}

bool is_equal(const_matrix_span left, const_matrix_span right)
{
	constexpr FloatType eps = 1.0e-3;

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
			if(left[row, col] - right[row, col] >= eps)
			{
				return false;
			}
			//if(left.at(row, col) != right.at(row, col))
			//	return false;
		}
	}

	return true;
}

matrix operator+(const matrix &mat)
{
	return mat;
}

matrix operator-(const matrix &mat)
{
	const auto [num_rows, num_cols] = mat.size();
	matrix res(num_rows, num_cols);
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			res[i, j] = -mat[i, j];
		}
	}

	return res;
}

} // namespace thwmakos
