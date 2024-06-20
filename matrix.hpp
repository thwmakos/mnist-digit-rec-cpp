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
// also used to represent vector (column or row vectors) by setting
// the appropriate dimension to 1
//
// supports only the operations that are needed and the element
// type is determined at before compilation for simplicity
class matrix
{
	public:
		// default constructor, a matrix without elements
		matrix() : 
			m_num_rows(0),
			m_num_columns(0),
			m_data(0)
		{}
			
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

		explicit matrix(std::pair<int, int> mat_size) : 
			matrix(mat_size.first, mat_size.second)
		{}

		// construct a matrix with given data
		// each initializer list is a row, e.g. Matrix mat ({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
		explicit matrix(std::initializer_list<std::initializer_list<FloatType>> init_data)
		{
			// there should be the same size for all initializer lists of init_data,
			// that is, all rows should have the same amount of elements
			m_num_rows = init_data.size();
			m_num_columns = init_data.begin()->size();
			
			if(m_num_rows == 0 || m_num_columns == 0)
			{
				throw std::invalid_argument("number of rows or columns is zero");
			}
			
			// allocate space and fill with default value
			// TODO: can be improved by not defaulting all the elements?
			m_data.reserve(m_num_rows * m_num_rows);

			for(auto& row : init_data)
			{
				// ensure all columns have the same length
				if(static_cast<int>(row.size()) != m_num_columns)
				{
					throw std::invalid_argument("all rows must have the same number of elements");
				}

				m_data.insert(m_data.cend(), row);
			}
		}
		
		// constructors, destructor, assignment operators are sufficient, 
		// delegate these to the std::vector m_data
		matrix(const matrix &) = default;
		matrix(matrix &&) = default;
		matrix& operator=(const matrix &) = default;
		matrix& operator=(matrix &&) = default;
		~matrix() = default;
		
		std::pair<int, int> size() const
		{
			return std::make_pair(m_num_rows, m_num_columns);
		}

		// resize the matrix
		// this operation discards all the matrix data
		// and zeroes out all of the new entries
		void set_size(int new_num_rows, int new_num_cols)
		{
			if(new_num_rows <= 0 || new_num_cols <= 0)
			{
				throw std::invalid_argument("number of rows and columns has to be strictly positive");
			}
			
			m_num_rows    = new_num_rows;
			m_num_columns = new_num_cols;
			m_data.resize(m_num_rows * m_num_columns, FloatType {});
		}

		// get number of rows of the matrix
		int num_rows() const { return m_num_rows; }

		// get number of columns of the matrix
		int num_cols() const { return m_num_columns; }

		// get the backing vector of the matrix containing all
		// the entries
		const std::vector<FloatType> &data() const { return m_data; }

		// member function to access the matrix elements
		// does bound checking
		FloatType& at(int row, int col)
		{
			return m_data.at(row * m_num_columns + col);
		}

		FloatType at(int row, int col) const
		{
			return m_data.at(row * m_num_columns + col);
		}

		// member function to access the matrix elements
		// does not do bound checking on debug builds
		FloatType& operator()(int row, int col)
		{
			if constexpr(debug_build)
				return m_data.at(row * m_num_columns + col);
			else
				return m_data[row * m_num_columns + col];
		}

		FloatType operator()(int row, int col) const
		{
			if constexpr(debug_build)
				return m_data.at(row * m_num_columns + col);
			else
				return m_data[row * m_num_columns + col];
		}
		
		// C++23 multiple subscript feature
		// same as operator() 	
		FloatType& operator[](int row, int col)
		{
			return operator()(row, col);
		}

		FloatType  operator[](int row, int col) const
		{
			return operator()(row, col);
		}

	private:
		int m_num_rows, m_num_columns; // number of rows and columns in the matrix

		std::vector<FloatType> m_data;
};

std::ostream & operator<<(std::ostream &os, const matrix& matrix);
bool operator==(const matrix& left, const matrix& right);

// multiply two matrices 
matrix multiply(const matrix& left, const matrix& right);
// return transpose of a matrix
matrix transpose(const matrix &mat);

// operator overloads
matrix operator+(const matrix& left, const matrix& right);
matrix operator-(const matrix& left, const matrix& right);
matrix operator-(const matrix& mat);
matrix operator*(const matrix& left, const matrix& right);

} // namespace thwmakos

#endif // MATRIX_HPP_INCLUDED
