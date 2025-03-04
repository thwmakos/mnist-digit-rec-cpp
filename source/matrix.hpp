//
// ~thwmakos~
//
// 14/6/2024
//
// matrix.hpp
//

// TODO: Matrix initialisation is slow as all the elements are zeroed due to using a std::vector as
// storage. Almost 1/3 of runtime (seen using perf and gprofng) when training the model is spent
// memsetting the elements to zero when creating a copy during operator calls. A solution for this
// would be to use std::make_unique_for_overwrite and use raw storage instead of going through
// vector. This will mean though that matrices will not be zero-initialised which is desirable in
// most cases. Need to consider how this can be achieved. Possibility is to opt-in for
// zero-initialisation through a flag in constructor. This will also necessitate changing the 
// type of m_data in the class.

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#include <vector>
#include <stdexcept>
#include <concepts>
#include <algorithm>
#include <span>
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

// TODO: decide if e.g. a matrix2d<Dynamic, Dynamic> with dimensions
// (n, m) should be treated as a matrix2d<n, m>. At the they are different

// a 2d-matrix view into a continuous extend of FloatType values
// non-owning and non-templated, to be used with functions 
// performing matrix operations
template<typename T>
struct matrix2d_span
{
	const int num_rows;
	const int num_columns;

	std::span<T> data;

	T &operator[](int row, int col)
		requires (!std::is_const_v<T>)
	{
		return data[row * num_columns + col];
	}

	T operator[](int row, int col) const
	{
		return data[row * num_columns + col];
	}

	std::tuple<int, int> size() const
	{
		return { num_rows, num_columns };
	}
	
	// calculate the index in the span data
	// of the element (row, col) of the matrix
	int index(int row, int col) const
	{
		return row * num_columns + col;
	}
};

// immutable and mutable spans
using const_matrix_span = matrix2d_span<const FloatType>;
using matrix_span       = matrix2d_span<FloatType>;

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

			// TODO: check for mismatches between Rows and num_rows and Columns and num_columns

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
		explicit matrix2d(int num_rows, int num_columns) :
			matrix2d(num_rows, num_columns, private_constructor_tag {})
		{}
		
		// construct zero matrix with given size
		explicit matrix2d(std::tuple<int, int> mat_size) 
			requires is_both_dynamic<Rows, Columns> : 
			matrix2d(std::get<0>(mat_size), std::get<1>(mat_size), private_constructor_tag {})
		{}

		explicit matrix2d(int length)
			requires is_dynamic_vector<Rows, Columns> :
			matrix2d(Rows == 1 ? 1 : length, Columns == 1 ? 1 : length, private_constructor_tag {})
		{}
		
		// construct a matrix from raw data, moving data argument into m_data
		explicit matrix2d(int num_rows, int num_columns, std::vector<FloatType> data) :
			matrix2d(num_rows, num_columns, std::move(data), private_constructor_tag {})
		{}

		explicit matrix2d(int length, std::vector<FloatType> data)
			requires is_dynamic_vector<Rows, Columns> :
			matrix2d(Rows == 1 ? 1 : length, Columns == 1 ? 1 : length, std::move(data), private_constructor_tag {})
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
			m_num_rows(Rows == 1 ? 1 : static_cast<int>(init_data.size())), 
			m_num_columns(Columns == 1 ? 1 : static_cast<int>(init_data.size())), 
			m_data(init_data)
		{}
		
		// constructors, destructor, assignment operators are sufficient, 
		// delegate these to the std::vector m_data
		matrix2d(const matrix2d &) = default;
		matrix2d(matrix2d &&) = default;
		matrix2d& operator=(const matrix2d &) = default;
		matrix2d& operator=(matrix2d &&) = default;
		~matrix2d() = default;
		
		std::tuple<int, int> size() const
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
		void set_size(std::tuple<int, int> new_size)
			requires is_both_dynamic<Rows, Columns>
		{
			set_size(std::get<0>(new_size), std::get<1>(new_size));
		}

		void set_size(int new_length)
			requires is_dynamic_vector<Rows, Columns>
		{
			if constexpr (Rows == 1)
			{
				m_num_columns = new_length;
			}
			else
			{
				m_num_rows = new_length;
			}

			// TODO: add checks for negative numbers here
			m_data.resize(new_length, FloatType {});
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
		
		// switch between column and row vectors
		matrix2d<Dynamic, 1> to_column() const &
			requires is_row_vector<Rows, Columns>
		{
			return matrix2d<Dynamic, 1>(m_num_columns, m_data);
		}
		
		matrix2d<Dynamic, 1> to_column() &&
			requires is_row_vector<Rows, Columns>
		{
			return matrix2d<Dynamic, 1>(m_num_columns, std::move(m_data));
		}
		
		matrix2d<1, Dynamic> to_row() const &
			requires is_column_vector<Rows, Columns>
		{
			return matrix2d<1, Rows>(m_num_rows, m_data);
		}
		
		matrix2d<1, Dynamic> to_row() &&
			requires is_column_vector<Rows, Columns>
		{
			return matrix2d<1, Dynamic>(m_num_rows, std::move(m_data));
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
			requires is_vector<Rows, Columns>
		{
			return m_data[n];
		}

		FloatType operator[](int n) const
			requires is_vector<Rows, Columns>
		{
			return m_data[n];
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
		
		operator const_matrix_span() const
		{
			return const_matrix_span { m_num_rows, m_num_columns, m_data };
		}

		operator matrix_span()
		{
			return matrix_span { m_num_rows, m_num_columns, m_data };
		}

	private:
		int m_num_rows;
		int m_num_columns; // number of rows and columns in the matrix

		std::vector<FloatType> m_data;
};

// TODO: inconsistent naming, think about this
template<int N>
using row_matrix = matrix2d<1, N>;

template<int N>
using column_matrix = matrix2d<N, 1>;

using matrix        = matrix2d<Dynamic, Dynamic>;
using row_vector    = matrix2d<1, Dynamic>;
using column_vector = matrix2d<Dynamic, 1>;

//
// Operations on matrix spans, the checks for appropriate dimensions
// during runtime are to be done by the caller. Due to the implicit
// conversions from matrix2d<Rows, Columns> to matrix_span, the functions
// can be called with matrix2d arguments
//

// multiply two matrices and write result in product
void multiply(matrix_span product, const_matrix_span left, const_matrix_span right);

// simple non-SIMD matrix multiplication fallback
// used in testing to compare against SIMD accelerated versions
void multiply_naive(matrix_span product, const_matrix_span left, const_matrix_span right);

// multiply each element of mat by scalar
void scalar_multiply(matrix_span mat, FloatType scalar);

// transpose mat and write result in result
void transpose(matrix_span result, const_matrix_span mat);

// performs the operation left[i, j] += scalar * right[i, j]
// the third argument is used to implement subtraction by setting scalar to -1
void add_to(matrix_span left, const_matrix_span right, FloatType scalar = 1.0);

// multiply each element of left by right,
// left[i, j] *= right[i, j]
void elementwise_multiply_by(matrix_span left, const_matrix_span right);

// add column to each column of mat, column has to have num_rows == 1, or
// num_columns == 1 (so it has to be a span to a vector, either row or column)
// this should be checked by the caller
void add_column_to(matrix_span mat, const_matrix_span column);

// check equality
bool is_equal(const_matrix_span left, const_matrix_span right);

//
// Operations on matrix2d arguments, these do dimensions checks
// and call the corresponding operations on matrix spans
//

// matrix multiplication
template<int RowsLeft, int ColumnsLeft, int RowsRight, int ColumnsRight>
	requires (ColumnsLeft == RowsRight)
auto operator*(const matrix2d<RowsLeft, ColumnsLeft> &left, const matrix2d<RowsRight, ColumnsRight> &right)
	-> matrix2d<RowsLeft, ColumnsRight>
{
	// make sure dimensions are matching (can happen if inner dimensions are dynamic)
	if(left.num_cols() != right.num_rows())
	{
		throw std::invalid_argument(std::format("multiply: mismatching matrix dimensions ({}, {}) and ({}, {})",
					left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}

	matrix2d<RowsLeft, ColumnsRight> product(left.num_rows(), right.num_cols());
	
	multiply(product, left, right);

	return product;
}

// compound multiplication by scalar
template<int Rows, int Columns>
matrix2d<Rows, Columns> &operator*=(matrix2d<Rows, Columns> &left, FloatType scalar)
{
	scalar_multiply(left, scalar);

	return left;
}

// multiply a matrix by a scalar (element-wise)
template<int Rows, int Columns>
matrix2d<Rows, Columns> operator*(FloatType scalar, matrix2d<Rows, Columns> mat)
{
	mat *= scalar;
	return mat;
}

template<int Rows, int Columns>
matrix2d<Rows, Columns> operator*(matrix2d<Rows, Columns> mat, FloatType scalar)
{
	mat *= scalar;
	return mat;
}

// compound add/subtract
// they modify their argument instead of creating new matrix
template<int Rows, int Columns>
matrix2d<Rows, Columns> &operator+=(matrix2d<Rows, Columns> &left, const matrix2d<Rows, Columns> &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(
				std::format("matrix addition: mismatched matrix dimensions:"
				"({}, {}) and ({}, {})", 
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}

	add_to(left, right);

	return left;
}

template<int Rows, int Columns>
matrix2d<Rows, Columns> &operator-=(matrix2d<Rows, Columns> &left, const matrix2d<Rows, Columns> &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(
				std::format("matrix addition: mismatched matrix dimensions:"
				"({}, {}) and ({}, {})", 
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}

	add_to(left, right, -1.0);

	return left;
}

// add/subtract two matrices together (element-wise)
// implemented in terms of += and -= respectively
// take first argument by value to allow moving
// of temporaries when writing e.g. a+b+c+d
template<int Rows, int Columns>
matrix2d<Rows, Columns> operator+(matrix2d<Rows, Columns> left, const matrix2d<Rows, Columns> &right)
{
	left += right;
	return left;
}

template<int Rows, int Columns>
matrix2d<Rows, Columns> operator-(matrix2d<Rows, Columns> left, const matrix2d<Rows, Columns> &right)
{
	left -= right;
	return left;
}

// return transpose of a matrix
template<int Rows, int Columns>
matrix2d<Columns, Rows> transpose(const matrix2d<Rows, Columns> &mat)
{
	matrix2d<Rows, Columns> result(mat.num_cols(), mat.num_rows());

	transpose(result, mat);

	return result;
}


// multiply every element of left by the corresponding element of right
// left[i, j] *= right[i, j]
// modifies left argument
// matrices must have the same dimensions
template<int Rows, int Columns>
matrix2d<Rows, Columns> &elementwise_multiply_inplace(matrix2d<Rows, Columns> &left, const matrix2d<Rows, Columns> &right)
{
	if(left.size() != right.size())
	{
		throw std::invalid_argument(std::format("elementwise_multiply_inplace: mismatched dimensions ({}, {}) and ({}, {})",
				left.num_rows(), left.num_cols(), right.num_rows(), right.num_cols()));
	}	

	elementwise_multiply_by(left, right);

	return left;
}

// add the column to each column of mat, modifying mat
template<int Rows, int Columns>
matrix2d<Rows, Columns> &add_column_inplace(matrix2d<Rows, Columns> &mat, const column_matrix<Rows> &column)
{
	if(mat.num_rows() != column.length())
	{
		throw std::invalid_argument(std::format("Column vector has {} elements, expected {}",
				column.length(), mat.num_rows()));
	}

	add_column_to(mat, column);

	return mat;
}

// add column to each column of mat returning a new matrix
template<int Rows, int Columns>
matrix2d<Rows, Columns> add_column(matrix2d<Rows, Columns> mat, const matrix2d<Rows, 1> &column)
{
	add_column_inplace(mat, column);
	return mat;
}

// extract the column of mat indicated by index
template<int Rows, int Columns>
column_matrix<Rows> get_column(const matrix2d<Rows, Columns> &mat, int index)
{
	if(index < 0 || index >= mat.num_cols())
	{
		throw std::invalid_argument(std::format("get_column: requested column {} but matrix has {} columns",
					index, mat.num_cols()));
	}

	column_matrix<Rows> column(mat.num_rows());

	for(auto i = 0; i < mat.num_rows(); ++i)
	{
		column[i] = mat[i, index];
	}	

	return column;	
}

// multiply two matrices element wise, that is c_{ij} = a_{ij} * b_{ij}
// and return a *new* matrix 
// first argument taken by value to allow move construction of temporaries
// similarly to the addition/multiply by scalar operations
template<int Rows, int Columns>
matrix2d<Rows, Columns> elementwise_multiply(matrix2d<Rows, Columns> left, const matrix2d<Rows, Columns> &right)
{
	elementwise_multiply_by(left, right);
	return left;
}

// apply func to every element of the matrix and return a new matrix
// test concepts btw
// TODO: probably not worth parallelising this using <execution>
template<int Rows, int Columns>
matrix2d<Rows, Columns> elementwise_apply(const matrix2d<Rows, Columns> &mat, std::regular_invocable<FloatType> auto func)
{
	const int num_rows = mat.num_rows();
	const int num_cols = mat.num_cols();
	//matrix2d<Rows, Columns> result(num_rows, num_cols);
	
	std::vector<FloatType> result_data {};
	result_data.reserve(num_rows * num_cols);	
	
	std::transform(mat.cbegin(), mat.cend(), std::back_inserter(result_data), func);

	return matrix2d<Rows, Columns>(num_rows, num_cols, std::move(result_data));
}

template<int Rows, int Columns>
bool operator==(const matrix2d<Rows, Columns> &left, const matrix2d<Rows, Columns> &right)
{
	return is_equal(left, right);
}

template<int Rows, int Columns>
bool operator!=(const matrix2d<Rows, Columns> &left, const matrix2d<Rows, Columns> &right)
{
	return !is_equal(left, right);
}

// unary plus, returns its argument
template<int Rows, int Columns>
matrix2d<Rows, Columns> operator+(const matrix2d<Rows, Columns> &mat)
{
	return mat;
}

// unary minus, negates every element
// take mat by value to allow move construction when temporaries are
// passed as argument
template<int Rows, int Columns>
matrix2d<Rows, Columns> operator-(matrix2d<Rows, Columns> mat)
{
	return -1.0 * mat;
}

template<int Rows, int Columns>
std::ostream &operator<<(std::ostream &os, const matrix2d<Rows, Columns> &mat)
{
	os << std::format("{:.3}", mat);
	return os;
}

} // namespace thwmakos


// std::formatter specialisation
template<int Rows, int Columns, typename CharT> 
struct std::formatter<thwmakos::matrix2d<Rows, Columns>, CharT>
{
	std::formatter<thwmakos::FloatType, CharT> value_formatter;

	constexpr auto parse(std::format_parse_context &pc)
	{
		return value_formatter.parse(pc);
	}

	auto format(const thwmakos::matrix2d<Rows, Columns> &mat, std::format_context &fc) const
	{
		auto out = fc.out();

		*out++ = '[';

		for(int i = 0; i < mat.num_rows(); ++i)
		{
			// two spaces for alignment
			if(i > 0)
			{
				*out++ = ' ';
				*out++ = ' ';
			}

			for(int j = 0; j < mat.num_cols(); ++j)
			{
				out = value_formatter.format(mat[i, j], fc);
				*out++ = ' ';
			}
			
			// if last row print ] instead of changing line
			if(i < mat.num_rows() - 1)
			{
				*out++ = '\n';
			}
			else
			{
				*out++ = ']';
			}
		}

		return out;
	}
};

#endif // MATRIX_HPP_INCLUDED
