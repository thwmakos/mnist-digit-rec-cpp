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
	for(auto row = 0; row < mat.num_rows; ++row)
	{
		for(auto col = 0; col < mat.num_columns; ++col)
		{
			mat[row, col] *= scalar;
		}
	}
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
void elementwise_multiply(matrix_span left, const_matrix_span right)
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


matrix &elementwise_multiply_inplace(matrix &left, const matrix &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(std::format("elementwise_multiply_inplace: mismatched dimensions ({}, {}) and ({}, {})",
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}	
	
	const auto [num_rows, num_cols] = left.size();
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			left[i, j] *= right[i, j];
		}
	}

	return left;
}

matrix elementwise_multiply(matrix left, const matrix &right)
{
	elementwise_multiply_inplace(left, right);
	return left;
}

matrix &add_column_inplace(matrix &mat, const matrix &column)
{
	if(column.num_cols() != 1 || mat.num_rows() != column.num_rows())
	{
		throw std::invalid_argument(std::format("add_column_inplace: mismatched dimensions ({}, {}) and ({}, {})",
				mat.num_rows(), mat.num_cols(), column.num_rows(), column.num_cols()));	
	}
	
	for(auto i = 0; i < mat.num_rows(); ++i)
	{
		for(auto j = 0; j < mat.num_cols(); ++j)
		{
			mat[i, j] -= column[i, 0];
		}
	}

	return mat;
}

matrix add_column(matrix mat, const matrix &column)
{
	add_column_inplace(mat, column);
	return mat;
}

matrix get_column(const matrix &mat, int index)
{
	if(index < 0 || index >= mat.num_cols())
	{
		throw std::invalid_argument(std::format("get_column: requested column {} but matrix has {} columns",
					index, mat.num_cols()));
	}

	matrix column(mat.num_rows(), 1);

	for(int i = 0; i < mat.num_rows(); ++i)
	{
		column[i, 0] = mat[i, index];
	}

	return column;
}

std::ostream &operator<<(std::ostream &os, const matrix& mat)
{
	// begin with [
	os << "[ ";

	for(auto row = 0; row < mat.num_rows(); ++row)
	{
		// for alignment print two spaces
		if(row > 0)
			os << ' ' << ' ';

		for(auto col = 0; col < mat.num_cols(); ++col)
		{
			os << mat[row, col] << ' ';
		}
		
		// if last row print ] instead of changing line
		if(row < mat.num_rows() - 1)
			os << '\n';
		else
			os << ']';
	}

	return os;
}

bool operator==(const matrix &left, const matrix &right)
{
	constexpr FloatType eps = 1.0e-3;

	if(left.num_rows() != right.num_rows() ||
	   left.num_cols() != right.num_cols())
	{
		return false;
	}

	for(auto row = 0; row < left.num_rows(); ++row)	
	{
		for(auto col = 0; col < left.num_cols(); ++col)
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

bool operator!=(const matrix &left, const matrix &right)
{
	return !(left == right);
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
