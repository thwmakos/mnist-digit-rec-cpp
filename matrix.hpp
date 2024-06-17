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
#include <stdexcept>
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
class matrix
{
	public:
		// construct new matrix with given number of rows and columns
		matrix(int num_rows, int num_columns) : 
			m_num_rows(num_rows),
			m_num_columns(num_columns)
			// initialise the storage vector with enough space
			// using the default value of the data type
		{
			if(num_rows <= 0 || num_columns <= 0)
			{
				throw std::invalid_argument("number of rows and columns has to be strictly positive");
			}

			m_data.resize(m_num_rows * m_num_columns, FloatType());
		}

		// construct a matrix with given data
		// each initializer list is a row, e.g. Matrix mat ({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
		matrix(std::initializer_list<std::initializer_list<FloatType>> init_data)
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

		matrix(const matrix &) = default;
		matrix(matrix &&) = default;
		
		// resize the matrix
		// this operation discards all the matrix data
		// and zeroes out all of the new entries
		void resize(int new_num_rows, int new_num_cols)
		{
			if(new_num_rows <= 0 || new_num_cols <= 0)
			{
				throw std::invalid_argument("number of rows and columns has to be strictly positive");
			}
			
			m_num_rows    = new_num_rows;
			m_num_columns = new_num_cols;
			m_data.resize(m_num_rows * m_num_columns, FloatType());
		}

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

		FloatType& operator()(int row, int col)
		{
			return at(row, col);
		}

		FloatType operator()(int row, int col) const
		{
			return at(row, col);
		}

	private:
		int m_num_rows, m_num_columns; // number of rows and columns in the matrix

		std::vector<FloatType> m_data;
};

std::ostream & operator<<(std::ostream &os, const matrix& matrix);
bool operator==(const matrix& left, const matrix& right);

// multiply two matrices 
matrix multiply(const matrix& left, const matrix& right);

} // namespace thwmakos

#endif // MATRIX_HPP_INCLUDED
