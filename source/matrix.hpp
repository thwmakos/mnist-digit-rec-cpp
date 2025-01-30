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
#include <concepts>
#include <algorithm>
#include <ostream>
#include <format>
#include <sstream> // for the formatter

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
// type is determined at compilation for simplicity

// denotes runtime-determined dimension
constexpr int Dynamic = -1;

template<int Rows, int Columns>
concept is_dynamic = (Rows == Dynamic || Columns == Dynamic);

template<int Rows, int Columns>
concept is_both_dynamic = (Rows == Dynamic && Columns == Dynamic);

template<int Rows, int Columns>
concept is_row_vector = (Rows == 1);

template<int Rows, int Columns>
concept is_dynamic_row = (Rows == 1 && Columns == Dynamic);

template<int Rows, int Columns>
concept is_column_vector = (Columns == 1);

template<int Rows, int Columns>
concept is_dynamic_column = (Rows == Dynamic && Columns == 1);

template<int Rows, int Columns>
concept is_vector = (is_row_vector<Rows, Columns> || is_column_vector<Rows, Columns>);

template<int Rows, int Columns>
concept is_dynamic_vector = (is_dynamic<Rows, Columns> && is_vector<Rows, Columns>);

template<int Rows, int Columns>
concept is_fixed = (Rows != Dynamic && Columns != Dynamic);


// support only <Dynamic, Dynamic>, <1 Dynamic> and <Dynamic, 1> for now
// TODO: support fixed size matrices
template<int Rows, int Columns> 
requires (is_both_dynamic<Rows, Columns> || is_dynamic_vector<Rows, Columns>)
class matrix2d
{
	public:
		// default constructor, a matrix without elements
		matrix2d() 
			requires is_both_dynamic<Rows, Columns> : 
			m_num_rows(0),
			m_num_columns(0),
			m_data(0)
		{}

		matrix2d() 
			requires is_dynamic_vector<Rows, Columns> :
			m_num_rows(Rows == 1 ? 1 : 0),
			m_num_columns(Columns == 1 ? 1 : 0),
			m_data(0)
		{}
	
	private:
		struct private_constructor_tag {};

		// main constructor, independent of any template parameters
		// private in order to avoid mismatch between the templated
		// numbers of rows and columns and the actual number of rows
		// and columns passed as arguments
		explicit matrix2d(int num_rows, int num_columns, private_constructor_tag) :
			m_num_rows(num_rows),
			m_num_columns(num_columns),
			m_data()
		{
			if(num_rows <= 0 || num_columns <= 0)
			{
				throw std::invalid_argument("number of rows and columns has to be strictly positive");
			}

			m_data.resize(m_num_rows * m_num_columns);
		}
		
		// construct a matrix from raw data, moving data argument into m_data
		explicit matrix2d(int num_rows, int num_columns, std::vector<FloatType> &&data, private_constructor_tag) :
			m_num_rows(num_rows),
			m_num_columns(num_columns),
			m_data(std::move(data))
		{
			if(num_rows <= 0 || num_columns <= 0)
			{
				throw std::invalid_argument("number of rows and columns has to be strictly positive");
			}
			
			if(static_cast<int>(m_data.size()) != m_num_rows * m_num_columns)
			{
				throw std::invalid_argument(std::format("data should have exactly num_rows * num_columns = {} elements instead of {}",
							m_num_rows * m_num_columns, m_data.size()));
			}
		}

	public:
		// construct new matrix with given number of rows and columns
		explicit matrix2d(int num_rows, int num_columns) 
			requires is_both_dynamic<Rows, Columns> : 
			matrix2d(num_rows, num_columns, private_constructor_tag {})
		{}
		
		// construct zero matrix with given size
		explicit matrix2d(std::pair<int, int> mat_size) 
			requires is_both_dynamic<Rows, Columns> : 
			matrix2d(mat_size.first, mat_size.second, private_constructor_tag {})
		{}

		explicit matrix2d(int length)
			requires is_dynamic_vector<Rows, Columns> :
			matrix2d(Rows == 1 ? 1 : length, Columns == 1 ? length : 1, private_constructor_tag {})
		{}
		
		// construct a matrix from raw data, moving data argument into m_data
		explicit matrix2d(int num_rows, int num_columns, std::vector<FloatType> data)
			requires is_both_dynamic<Rows, Columns> :
			matrix2d(num_rows, num_columns, std::move(data), private_constructor_tag {})
		{}

		explicit matrix2d(int length, std::vector<FloatType> data)
			requires is_dynamic_vector<Rows, Columns> :
			matrix2d(Rows == 1 ? 1 : length, Columns == 1 ? length : 1, std::move(data), private_constructor_tag {})
		{}	
		
		// construct a matrix with given data
		// each initializer list is a row, e.g. matrix mat ({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
		matrix2d(std::initializer_list<std::initializer_list<FloatType>> init_data)
			requires is_both_dynamic<Rows, Columns>
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

			for(auto &row : init_data)
			{
				// ensure all columns have the same length
				if(static_cast<int>(row.size()) != m_num_columns)
				{
					throw std::invalid_argument("all rows must have the same number of elements");
				}

				m_data.insert(m_data.cend(), row);
			}
		}

		matrix2d(std::initializer_list<FloatType> init_data)
			requires is_dynamic_vector<Rows, Columns> :
			m_num_rows(Rows == 1 ? 1 : static_cast<int>(init_data.size()), 
					Columns == 1 ? static_cast<int>(init_data.size()) : 1, 
					std::move(init_data), 
					private_constructor_tag {})
		{}
		
		// constructors, destructor, assignment operators are sufficient, 
		// delegate these to the std::vector m_data
		matrix2d(const matrix2d &) = default;
		matrix2d(matrix2d &&) = default;
		matrix2d& operator=(const matrix2d &) = default;
		matrix2d& operator=(matrix2d &&) = default;
		~matrix2d() = default;
		
		std::pair<int, int> size() const
		{
			return std::make_pair(m_num_rows, m_num_columns);
		}

		int length() const 
			requires is_vector<Rows, Columns>
		{
			if constexpr (Rows == 1)
			{
				return m_num_columns;
			}
			else
			{
				return m_num_rows;
			}
		}

		// resize the matrix to a new arbitrary size
		// this operation discards all the matrix data
		// and zeroes out all of the new entries
		void set_size(int new_num_rows, int new_num_cols)
			requires is_both_dynamic<Rows, Columns>
		{
			if constexpr (!is_both_dynamic<Rows, Columns>)
			{
				if(Rows != Dynamic && new_num_rows != Rows)
				{
					throw std::invalid_argument(std::format("set_size: number of rows must be {}", Rows));  
				}
				
				if(Columns != Dynamic && new_num_cols != Columns)
				{
					throw std::invalid_argument(std::format("set_size: number of columns must be {}", Columns));
				}
			}

			if(new_num_rows <= 0 || new_num_cols <= 0)
			{
				throw std::invalid_argument("set_size: number of rows and columns has to be strictly positive");
			}
			
			m_num_rows    = new_num_rows;
			m_num_columns = new_num_cols;
			m_data.resize(m_num_rows * m_num_columns, FloatType {});
		}
		
		// overload of set_size for convenience
		void set_size(std::pair<int, int> new_size)
			requires is_both_dynamic<Rows, Columns>
		{
			set_size(new_size.first, new_size.second);
		}

		// reinterpret the size of the matrix while keeping its data
		// the total amount of elements before and after has to be the same
		// main use it to convert a column vector to a row vector 
		// leaves the matrix it is called on to a moved-from state unless
		// both Rows and Columns are dynamic
		matrix2d &reshape(int new_num_rows, int new_num_cols)
			requires is_both_dynamic<Rows, Columns>
		{
			if(new_num_rows * new_num_cols != m_num_rows * m_num_columns)
			{
				throw std::invalid_argument(std::format("cannot reshape ({}, {}) to ({}, {})", 
							m_num_rows, m_num_columns, new_num_rows, new_num_cols));
			}
			
			m_num_rows    = new_num_rows;
			m_num_columns = new_num_cols;

			return *this;
		}	

		// get number of rows of the matrix
		int num_rows() const { return m_num_rows; }

		// get number of columns of the matrix
		int num_cols() const { return m_num_columns; }

		// get the backing vector of the matrix containing all
		// the entries
		const std::vector<FloatType> &data() const { return m_data; }
		// careful when using this version, do not add or remove elements
		// from m_data
		std::vector<FloatType> &data() { return m_data; }

		// member function to access the matrix elements
		// does bound checking
		FloatType &at(int row, int col)
		{
			if(row < 0 || row >= m_num_rows || col < 0 || col >= m_num_columns)
			{
				throw std::out_of_range(std::format("matrix subscripts ({}, {}) out of range", row, col));
			}

			return m_data[row * m_num_columns + col];
		}

		FloatType at(int row, int col) const
		{
			if(row < 0 || row >= m_num_rows || col < 0 || col >= m_num_columns)
			{
				throw std::out_of_range(std::format("matrix subscripts ({}, {}) out of range", row, col));
			}

			return m_data[row * m_num_columns + col];
		}

		// single index access for row and column vectors
		FloatType &at(int n)
			requires is_vector<Rows, Columns>
		{
			if constexpr(Rows == 1)
			{
				return at(1, n);
			}
			else
			{
				return at(n, 1);
			}
		}

		FloatType at(int n) const
			requires is_vector<Rows, Columns>
		{
			if constexpr(Rows == 1)
			{
				return at(1, n);
			}
			else
			{
				return at(n, 1);
			}
		}

		// member function to access the matrix elements
		// does not do bound checking on debug builds
		FloatType &operator()(int row, int col)
		{
			return m_data[row * m_num_columns + col];
		}

		FloatType operator()(int row, int col) const
		{
			return m_data[row * m_num_columns + col];
		}

		FloatType &operator()(int n)
			requires is_vector<Rows, Columns>
		{
			return m_data[n];
		}

		FloatType operator()(int n) const
			requires is_vector<Rows, Columns>
		{
			return m_data[n];
		}
		
		// C++23 multiple subscript operator []
		// same as operator() 	
		FloatType &operator[](int row, int col)
		{
			return operator()(row, col);
		}

		FloatType operator[](int row, int col) const
		{
			return operator()(row, col);
		}

		FloatType &operator[](int n)
		{
			return m_data[n];
		}

		FloatType operator[](int n) const
		{
			return m_data[n];
		}
		
		// calculate the index in the vector m_data
		// of the element (row, col) of the matrix
		int index(int row, int col) const
		{
			return row * m_num_columns + col;
		}

		// iterators to matrix data
		// access matrix entries sequentially, in row major order
		auto begin() { return m_data.begin(); }
		auto end() { return m_data.end(); }
		auto cbegin() const { return m_data.cbegin(); }
		auto cend() const { return m_data.cend(); }
		auto rbegin() { return m_data.rbegin(); }
		auto rend() { return m_data.rend(); }
		auto crbegin() const { return m_data.crbegin(); }
		auto crend() const { return m_data.crend(); }

	private:
		int m_num_rows;
		int m_num_columns; // number of rows and columns in the matrix

		std::vector<FloatType> m_data;
};

// a 2d-matrix view into a continuous extend of FloatType values
struct matrix2d_span
{
	const int num_rows;
	const int num_columns;

	std::span<FloatType> data;
};

using matrix        = matrix2d<Dynamic, Dynamic>;
using row_vector    = matrix2d<1, Dynamic>;
using column_vector = matrix2d<Dynamic, 1>;

// multiply two matrices
matrix multiply(const matrix&, const matrix&);
// return transpose of a matrix
matrix transpose(const matrix &);
// simple non-SIMD matrix multiplication fallback
matrix multiply_naive(const matrix &, const matrix &);
// simple non-SIMD function to add the second operand to the first
matrix &add_to(matrix &, const matrix &);

// multiply every element of left by the corresponding element of right
// left[i, j] *= right[i, j]
// modifies left argument
// matrices must have the same dimensions
matrix &elementwise_multiply_inplace(matrix &left, const matrix &right);

// multiply two matrices element wise, that is c_{ij} = a_{ij} * b_{ij}
// and return a *new* matrix 
// first argument taken by value to allow move construction of temporaries
// similarly to the addition/multiply by scalar operations
matrix elementwise_multiply(matrix left, const matrix &right);

// apply func to every element of the matrix and return a new matrix
// test concepts btw
// TODO: add tests for this function -- DONE
// TODO: parallelise this use <execution>
matrix elementwise_apply(const matrix &mat, std::regular_invocable<FloatType> auto func)
{
	const auto [num_rows, num_cols] = mat.size();
	matrix result(num_rows, num_cols);
	
	std::transform(mat.cbegin(), mat.cend(), result.begin(), func);

	return result;
}

// add the column to each column of mat, modifying mat
matrix &add_column_inplace(matrix &mat, const matrix &column);
// add column to each column of mat returning a new matrix
matrix add_column(matrix mat, const matrix &column);

// extract the column of mat indicated by index
matrix get_column(const matrix &mat, int index);

// operator overloads

// unary plus, returns its argument
matrix operator+(const matrix &);
// unary minus, negates every element
matrix operator-(const matrix &mat);

// add/subtract two matrices together (element-wise)
// implemented in terms of += and -= respectively
// take first argument by value to allow moving
// of temporaries when writing things like a+b+c+d
matrix operator+(matrix left, const matrix &right);
matrix operator-(matrix left, const matrix &right);

// compound add/subtract/(multiply by scalar)
// they modify their argument instead of creating new matrix
matrix &operator+=(matrix &left, const matrix &right);
matrix &operator-=(matrix &left, const matrix &right);
matrix &operator*=(matrix &left, FloatType scalar);

// matrix multiplication
matrix operator*(const matrix &left, const matrix &right);

// multiply a matrix by a scalar (element-wise)
matrix operator*(FloatType scalar, matrix mat);
matrix operator*(matrix mat, FloatType scalar);

bool operator==(const matrix &, const matrix &);
bool operator!=(const matrix &, const matrix &);

std::ostream & operator<<(std::ostream&, const matrix&);

} // namespace thwmakos


// std::formatter specialisation
template<> struct std::formatter<thwmakos::matrix>
{
	constexpr auto parse(std::format_parse_context &pc)
	{
		return pc.begin();
	}

	auto format(const thwmakos::matrix &mat, std::format_context &fc) const
	{
		std::ostringstream stream;
		stream << mat;
		return std::format_to(fc.out(), "{}", stream.str());
	}
};


#endif // MATRIX_HPP_INCLUDED
