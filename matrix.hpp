//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#include <vector>
#include <exception>
#include <ostream>
#include <cassert>

namespace thwmakos {

#ifdef NDEBUG
constexpr auto debug_build = false;
#else
constexpr auto debug_build = true;
#endif

// floating point format used for weights and biases
using FloatType = float;

// simple matrix class, with storage backed by an std::vector, so 
// heap allocated
//
// supports only the operations that are needed and the element
// type is determined at before compilation for simplicity
class Matrix
{
	public:
		// construct new matrix with given number of rows and columns
		Matrix(int num_rows, int num_columns) : 
			m_num_rows(num_rows),
			m_num_columns(num_columns)
			// initialise the storage vector with enough space
			// using the default value of the data type
		{
			assert(m_num_rows > 0);
			assert(m_num_columns > 0);

			m_data.resize(m_num_rows * m_num_columns, FloatType());
			assert(static_cast<int>(m_data.size()) == m_num_rows * m_num_columns);
		}

		// construct a matrix with given data
		// each initializer list is a row, e.g. Matrix mat ({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
		Matrix(std::initializer_list<std::initializer_list<FloatType>> init_data)
		{
			// there should be the same size for all initializer lists of init_data,
			// that is, all rows should have the same amount of elements
			m_num_rows = init_data.size();
			m_num_columns = init_data.begin()->size();

			assert(m_num_rows > 0);
			assert(m_num_columns > 0);
			
			// allocate space and fill with default value
			// TODO: can be improved by not defaulting all the elements?
			m_data.reserve(m_num_rows * m_num_rows);

			for(auto& row : init_data)
			{
				// ensure all columns have the same length
				assert(static_cast<int>(row.size()) == m_num_columns);
				m_data.insert(m_data.cend(), row);
			}
		}

		Matrix(const Matrix &) = default;
		Matrix(Matrix &&) = default;

		int num_rows() const { return m_num_rows; }
		int num_cols() const { return m_num_columns; }
		
		// member function to access the matrix elements
		// only being accessed with bounds checking for now
		// TODO: disable bounds check after being done
		// TODO: at() is probably a bad name if I access value
		//       without bounds check due to standard convention
		FloatType& at(int row, int col)
		{
			if constexpr(debug_build)
				return m_data.at(row * m_num_columns + col);
			else
				return m_data[row * m_num_columns + col];
		}

		FloatType at(int row, int col) const
		{
			if constexpr(debug_build)
				return m_data.at(row * m_num_columns + col);
			else
				return m_data[row * m_num_columns + col];
		}

	private:
		int m_num_rows, m_num_columns; // number of rows and columns in the matrix

		std::vector<FloatType> m_data;
};



std::ostream & operator<<(std::ostream &os, const Matrix& matrix);
bool operator==(const Matrix& left, const Matrix& right);

// multiply two matrices 
Matrix multiply(const Matrix& left, const Matrix& right);

} // namespace thwmakos

#endif // MATRIX_HPP_INCLUDED
