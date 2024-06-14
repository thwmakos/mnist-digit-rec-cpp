//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

#include "matrix.hpp"

#include <limits>

namespace thwmakos {

std::ostream & operator<<(std::ostream &os, const Matrix& matrix)
{
	// begin with [
	os << "[ ";

	for(auto row = 0; row < matrix.num_rows(); ++row)
	{
		// for alignment print two spaces
		if(row > 0)
			os << ' ' << ' ';

		for(auto col = 0; col < matrix.num_cols(); ++col)
		{
			os << matrix.at(row, col) << ' ';
		}
		
		// if last row print ] instead of changing line
		if(row < matrix.num_rows() - 1)
			os << '\n';
		else
			os << ']';
	}

	return os;
}

bool operator==(const Matrix& left, const Matrix& right)
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
			if(left.at(row, col) - right.at(row, col) >= std::numeric_limits<FloatType>::epsilon())
			{
				return false;
			}
		}
	}

	return true;
}

} // namespace thwmakos
