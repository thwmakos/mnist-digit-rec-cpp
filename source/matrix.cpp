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

matrix multiply_naive(const matrix &left, const matrix &right)
{
	// store these so we don't have to call num_rows() and num_cols()
	// all the time, idk if this speeds up the functions, the above
	// functions should be inlined anyway 
	const auto [left_num_rows, left_num_cols]   = left.size();
	const auto [right_num_rows, right_num_cols] = right.size();

	matrix product(left_num_rows, right_num_cols);
	
	// naive implementation of matrix multiplication
	// TODO: good learning opportunity for intrinsics here
	
	// transpose right first to ensure sequential access 
	// to matrix elements
	// uses more memory but is ~4 times faster for large matrices
	auto right_transpose = transpose(right);
	
	for(auto i = 0; i < left_num_rows; ++i)
	{
		for(auto j = 0; j < right_num_cols; ++j)
		{
			// equivalently we could check against right_num_rows
			FloatType accumulator {}; 
			
			for(auto k = 0; k < left_num_cols; ++k)
			{
				accumulator += left[i, k] * right_transpose[j, k];
			}

			product[i, j] = accumulator; 
		}
	}

	return product;
}

matrix multiply(const matrix &left, const matrix &right)
{
	// make sure dimensions are matching
	if(left.num_cols() != right.num_rows())
	{
		throw std::invalid_argument("multiply: mismatching matrix dimensions");
	}
	
#ifdef __AVX512F__
	return multiply_avx512(left, right);
#elifdef __AVX2__
	return multiply_avx2(left, right);
#else
	return multiply_naive(left, right);
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

matrix elementwise_multiply(const matrix &left, const matrix &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument("elementwise_multiply: matrices must have same dimensions");
	}	

	const auto [num_rows, num_cols] = left.size();
	matrix result(num_rows, num_cols);
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			result[i, j] = left[i, j] * right[i, j];
		}
	}

	return result;
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

matrix operator*(const matrix &left, const matrix &right)
{
	return multiply(left, right);
}

matrix operator*(FloatType scalar, const matrix &mat)
{
	matrix res(mat.size());

	for(auto row = 0; row < mat.num_rows(); ++row)
	{
		for(auto col = 0; col < mat.num_cols(); col++)
		{
			res[row, col] = scalar * mat[row, col];	
		}
	}

	return res;
}

} // namespace thwmakos
