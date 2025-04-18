//
// matrix_view.hpp
//
// 3/4/2025
//
// ~thwmakos~
//

// a non-owning, mutable/immutable view into a matrix2d defined by a 
// starting (row, column) element in the matrix and extending 
// num_rows, num_columns in each direction respectively

#include "matrix.hpp"

namespace thwmakos {

template<typename T>
struct matrix2d_view
{
	matrix2d_span<T> parent_span;
	const int start_row;
	const int start_col;
	const int num_rows;
	const int num_cols;

	// constructor for a view into a submatrix
	matrix2d_view(matrix2d_span<T> _parent_span, int _start_row, int _start_col, int _num_rows, int _num_cols) :
		parent_span(_parent_span),
		start_row(_start_row),
		start_col(_start_col),
		num_rows(_num_rows),
		num_cols(_num_cols) 
	{
		if (start_row < 0 || start_col < 0 || 
			start_row + num_rows > parent_span.num_rows || 
			start_col + num_cols > parent_span.num_columns) 
		{
			throw std::out_of_range(std::format("matrix_view: submatrix ({}, {}) with size ({}, {}) "
				"extends beyond parent matrix of size ({}, {})",
				start_row, start_col, num_rows, num_cols, 
				parent_span.num_rows, parent_span.num_columns));
		}
	}

	// constructor for creating a view of the entire matrix
	explicit matrix2d_view(matrix2d_span<T> _parent_span) :
		matrix2d_view(_parent_span, 0, 0, _parent_span.num_rows, _parent_span.num_columns) {}
	
	// conversion operator to const view so that matrix2d_view<const T> can be passed
	// where matrix2d_view<T> is expected
	operator matrix2d_view<const std::remove_const_t<T>>() const 
		requires (!std::is_const_v<T>)
	{
		using non_const_T = std::remove_const_t<T>;
		
		matrix2d_span<const non_const_T> const_parent
		{
			parent_span.num_rows,
			parent_span.num_columns,
			std::span<const non_const_T>(parent_span.data)
		};
		
		return matrix2d_view<const non_const_T>
		{
			const_parent,
			start_row,
			start_col,
			num_rows,
			num_cols
		};
	}

	FloatType &at(int row, int col) 
		requires (!std::is_const_v<T>)
	{
		if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) 
		{
			throw std::out_of_range(std::format("matrix_view subscripts ({}, {}) out of range", row, col));
		}
		
		return parent_span[start_row + row, start_col + col];
	}

	FloatType at(int row, int col) const
	{
		if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) 
		{
			throw std::out_of_range(std::format("matrix_view subscripts ({}, {}) out of range", row, col));
		}
		
		return parent_span[start_row + row, start_col + col];
	}

	FloatType &operator()(int row, int col)
		requires (!std::is_const_v<T>)
	{
		if constexpr(debug_build)
		{
			return at(row, col);
		}

		return parent_span[start_row + row, start_col + col];
	}
		
	FloatType operator()(int row, int col) const
	{
		if constexpr(debug_build)
		{
			return at(row, col);
		}

		return parent_span[start_row + row, start_col + col];
	}

	FloatType &operator[](int row, int col)
		requires (!std::is_const_v<T>)
	{
		if constexpr(debug_build)
		{
			return at(row, col);
		}

		return operator()(row, col);
	}

	FloatType operator[](int row, int col) const
	{
		if constexpr(debug_build)
		{
			return at(row, col);
		}

		return operator()(row, col);
	}

	std::tuple<int, int> size() const
	{
		return { num_rows, num_cols };
	}
	
	// get address of element, is used to load/store during matrix operations	
	const float *address(int row, int col) const
	{
		if constexpr(debug_build)
		{
			if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) 
			{
				throw std::out_of_range(std::format("matrix_view subscripts ({}, {}) out of range", row, col));
			}
		}

		return parent_span.data.data() + parent_span.index(start_row + row, start_col + col);
	}
	
	// get address of element, is used to load/store during matrix operations	
	float *address(int row, int col)
		requires (!std::is_const_v<T>)
	{
		if constexpr(debug_build)
		{
			if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) 
			{
				throw std::out_of_range(std::format("matrix_view subscripts ({}, {}) out of range", row, col));
			}
		}

		return parent_span.data.data() + parent_span.index(start_row + row, start_col + col);
	}

};

using matrix_view       = matrix2d_view<FloatType>;
using const_matrix_view = matrix2d_view<const FloatType>;

} // namespace thwmakos
