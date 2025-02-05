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

	
void multiply_naive(matrix_span product, const_matrix_span left, const_matrix_span right)
{	
	// naive implementation of matrix multiplication
	// TODO: good learning opportunity for intrinsics here
	
	// transpose right first to ensure sequential access 
	// to matrix elements
	// uses more memory but is ~4 times faster for large matrices
	//auto right_transpose = transpose(right);
	
	for(auto i = 0; i < left.num_rows; ++i)
	{
		for(auto j = 0; j < right.num_columns; ++j)
		{
			// equivalently we could check against right_num_rows
			FloatType accumulator {}; 
			
			for(auto k = 0; k < left.num_columns; ++k)
			{
				accumulator += left[i, k] * right[k, j];
			}

			product[i, j] = accumulator; 
		}
	}
}

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

// does left += right and then returns left
matrix &add_to(matrix &left, const matrix &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(
				std::format("matrix addition: mismatched matrix dimensions:"
				"({}, {}) and ({}, {})", 
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}

#ifdef AVX512F
	return add_to_avx512(left, right);
#else

	for(auto i = 0; i < left.num_rows(); ++i)
	{
		for(auto j = 0; j < left.num_cols(); ++j)
		{
			left[i, j] += right[i, j];
		}
	}

	return left;
#endif
}

// TODO: this is very slow, need to make this cache friendly
// by splitting the matrix in blocks and transposing accordingly
matrix transpose(const matrix &mat)
{
	// create the to be returned matrix with appropriate dimensions
	matrix result(mat.num_cols(), mat.num_rows());

	const auto [res_num_rows, res_num_cols] = result.size();	

	for(auto i = 0; i < res_num_rows; ++i)
	{
		for(auto j = 0; j < res_num_cols; ++j)
		{
			// trying out C++23 multiple subscripts
			result[i, j] = mat[j, i];
		}
	}

	return result;
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

// left is passed by value
// if a temporary is passed left then the move constructor
// is called, this expressions such as a + b + c are cheap
// causing only one allocation and multiple moves
// not ideal but simple to implement 
matrix operator+(matrix left, const matrix &right)
{
	left += right;
	return left;
}

matrix operator-(matrix left, const matrix &right)
{
	left -= right;
	return left;
}

matrix &operator+=(matrix &left, const matrix &right)
{
	return add_to(left, right);
}

matrix &operator-=(matrix &left, const matrix &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(
				std::format("matrix subtraction: mismatched matrix dimensions:"
				"({}, {}) and ({}, {})", 
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}

	for(auto i = 0; i < left.num_rows(); ++i)
	{
		for(auto j = 0; j < left.num_cols(); ++j)
		{
			left[i, j] -= right[i, j];
		}
	}

	return left;
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

} // namespace thwmakos
