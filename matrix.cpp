//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

#include "matrix.hpp"

//#include <limits>

namespace thwmakos {

matrix multiply(const matrix& left, const matrix& right)
{
	// store these so we don't have to call num_rows() and num_cols()
	// all the time, idk if this speeds up the functions, the above
	// functions should be inlined anyway 
	auto [left_num_rows, left_num_cols]   = left.size();
	auto [right_num_rows, right_num_cols] = right.size();

	// make sure dimensions are matching
	if(left_num_cols != right_num_rows)
	{
		throw std::invalid_argument("multiply: mismatching matrix dimensions");
	}
	
	matrix product(left_num_rows, right_num_cols);
	
	// naive implementation O(n^3) of matrix multiplication
	// TODO: good learning opportunity for intrinsics here
	for(auto i = 0; i < left_num_rows; ++i)
	{
		for(auto j = 0; j < right_num_cols; ++j)
		{
			// equivalently we could check against right_num_rows
			for(auto k = 0; k < left_num_cols; ++k)
			{
				product.at(i, j) += left.at(i, k) * right.at(k, j);
			}
		}
	}

	return product;
}

std::ostream & operator<<(std::ostream &os, const matrix& mat)
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
			os << mat.at(row, col) << ' ';
		}
		
		// if last row print ] instead of changing line
		if(row < mat.num_rows() - 1)
			os << '\n';
		else
			os << ']';
	}

	return os;
}

bool operator==(const matrix& left, const matrix& right)
{
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
			if(left.at(row, col) - right.at(row, col) >= 1.0e-5)
			{
				return false;
			}
			//if(left.at(row, col) != right.at(row, col))
			//	return false;
		}
	}

	return true;
}

matrix operator+(const matrix& left, const matrix& right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument("matrix addition: non-matching sizes");
	}

	auto [num_rows, num_cols] = left.size();
	matrix res(num_rows, num_cols);
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			res.at(i, j) = left.at(i, j) + right.at(i, j);
		}
	}

	return res;
}

matrix operator-(const matrix& left, const matrix& right)
{
	return left + (-right);
}

matrix operator-(const matrix& mat)
{
	auto [num_rows, num_cols] = mat.size();
	matrix res(num_rows, num_cols);
	
	for(auto i = 0; i < num_rows; ++i)
	{
		for(auto j = 0; j < num_cols; ++j)
		{
			res.at(i, j) = -mat.at(i, j);
		}
	}

	return res;
}

matrix operator*(const matrix& left, const matrix& right)
{
	return multiply(left, right);
}

} // namespace thwmakos