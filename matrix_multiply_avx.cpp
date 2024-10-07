//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_mutliply_avx.cpp
//

#include "matrix_multiply_avx.hpp"

#include <cassert>
#include <immintrin.h>

#include <array>
#include <print>

// TODO: update submatrix functions with std::array and std::span

namespace thwmakos { 

// we calculate the product C = A * B by computing
// C in smaller submatrices of dimensions given
// by the parameters below
constexpr int num_lanes = 16;
constexpr int num_submatrix_rows = 8;             // submatrix size needs to be adjusted to CPU
constexpr int num_submatrix_cols = 4 * num_lanes; // these parameters provide 20x performance boots to
										          // naive implementation on intel tgl (i7 11800H CPU)
// the number of columns is a multiple of 16 which 
// is how many single precision floats an avx512 
// register stores
// 
// calculating each submatrix is called calculating
// a 'kernel' usually

// calculate the elements of the submatrix of C
// given by the rows in range [num_row, num_row + num_submatrix_rows)
// and columns in range [num_col, num_col + num_submatrix_cols
void submatrix(const matrix &A, const matrix &B, matrix &C, int num_row, int num_col)
{
	// if m <- num_submatrix_rows and K <- A.num_cols() (== B.num_rows()) 
	// then each row (c_(1,1), c_(1, 2), ..., c_(1, m))
	// is given by a_(1, 1) * (b_(1, 1), b_(1, 2), ..., b_(1, m))
	// + a_(1, 2) * (b_(2, 1), b_(2, 2), ..., b_(2, m))
	// + ...
	// + a_(1, K) * (b_(K, 1), b_(K, 2), ..., b_(K, m)) 
	//
	// where a_(i, j), b_(i, j) are the corresponding submatrices of A, B
	//
	// each a_(1, j) will be broadcasted to an __m512 variable and
	// each row of b's will be loaded to a small number of __m512 variables
	// equal to num_submatrix_cols / 16
	

	__m512 C_submatrix[num_submatrix_rows][num_submatrix_cols / num_lanes];
	__m512 a;
	__m512 b_row[num_submatrix_cols / num_lanes];

	// get pointers to raw data first
	const float *A_data = A.data().data();
	const float *B_data = B.data().data();
	float       *C_data = C.data().data();

	// we first load the submatrix of C into C_submatrix
	for(int i = 0; i < num_submatrix_rows; ++i)
	{
		const int C_load_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			C_submatrix[i][j] = _mm512_loadu_ps(&C_data[C_load_index + num_lanes * j]);
		}
	}

	// loop over every row of C_submatrix calculating its value
	// to do this loop over the rows of B as described above
	for(int i = 0; i < B.num_rows(); ++i)
	{
		// load num_submatrix_cols elements from the current row of B
		const int B_load_index = B.index(i, num_col);
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			b_row[j] = _mm512_loadu_ps(&B_data[B_load_index + num_lanes * j]);
		}

		// update each row of C_submatrix
		for(int n = 0; n < num_submatrix_rows; ++n)
		{
			const int A_load_index = A.index(num_row + n, i);
			a = _mm512_set1_ps(A_data[A_load_index]);

			for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
			{
				C_submatrix[n][j] = _mm512_fmadd_ps(a, b_row[j], C_submatrix[n][j]);
			}
		}
	}

	// store results back to C
	for(int i = 0; i < num_submatrix_rows; ++i)
	{
		const int C_store_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			_mm512_storeu_ps(&C_data[C_store_index + num_lanes * j], C_submatrix[i][j]);
		}
	}
}

// same as submatrix() except that we do not assume that we have num_submatrix_rows after num_row
// or num_submatrix_cols after num_col
void submatrix_masked(const matrix &A, const matrix &B, matrix &C, int num_row, int num_col)
{
	__m512 C_submatrix[num_submatrix_rows][num_submatrix_cols / num_lanes];
	__m512 a;
	__m512 b_row[num_submatrix_cols / num_lanes];
	__mmask16 masks[num_submatrix_cols / num_lanes];

	// get pointers to raw data first
	const float *A_data = A.data().data();
	const float *B_data = B.data().data();
	float       *C_data = C.data().data();
	
	// we only process up to a submatrix size	
	const int actual_rows = std::min(C.num_rows() - num_row, num_submatrix_rows);
	const int actual_cols = std::min(C.num_cols() - num_col, num_submatrix_cols);
	
	// FIXME: this is redundant
	if(actual_rows <= 0 || actual_cols <= 0)
	{
		return;
	}

	//assert(actual_rows <= num_submatrix_rows);
	//assert(actual_cols <= num_submatrix_cols);
	
	// calculate masks for data loading/storing for C and B
	for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
	{
		const int lanes_remaining = actual_cols - j * num_lanes;

		if(lanes_remaining >= num_lanes)
		{
			// TODO: make this constant generic (determine it from num_lanes)
			masks[j] = (1u << num_lanes) - 1u;
		}
		else if(lanes_remaining > 0)
		{
			masks[j] = (1u << lanes_remaining) - 1u;
		}
		else
		{
			masks[j] = 0u;
		}
	}
	
	// we first load the submatrix of C into C_submatrix
	for(int i = 0; i < actual_rows; ++i)
	{
		const int C_load_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			C_submatrix[i][j] = _mm512_maskz_loadu_ps(masks[j], &C_data[C_load_index + num_lanes * j]);
		}
	}
	
	// loop over every row of C_submatrix calculating its value
	// to do this loop over the rows of B as described above
	for(int i = 0; i < B.num_rows(); ++i)
	{
		// load num_submatrix_cols elements from the current row of B
		const int B_load_index = B.index(i, num_col);
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			b_row[j] = _mm512_maskz_loadu_ps(masks[j], &B_data[B_load_index + num_lanes * j]);
		}

		// update each row of C_submatrix
		for(int n = 0; n < actual_rows; ++n)
		{
			const int A_load_index = A.index(num_row + n, i);
			a = _mm512_set1_ps(A_data[A_load_index]);

			for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
			{
				C_submatrix[n][j] = _mm512_fmadd_ps(a, b_row[j], C_submatrix[n][j]);
			}
		}
	}
	
	// store results back to C
	for(int i = 0; i < actual_rows; ++i)
	{
		const int C_store_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			_mm512_mask_storeu_ps(&C_data[C_store_index + num_lanes * j], masks[j], C_submatrix[i][j]);
		}
	}
}

matrix multiply_avx512(const matrix &A, const matrix &B)
{
	// the function only works with single precision floats
	static_assert(std::is_same_v<FloatType, float>);

	// no maskings loads/stores introduced yet, so we assume
	// matrix dimensions satisfy the following for now
	//assert(A.num_rows() % num_submatrix_rows == 0);
	//assert(B.num_cols() % num_submatrix_cols == 0);

	// matrix to be returned
	matrix C(A.num_rows(), B.num_cols());

	int i = 0;	
	
	// make sure there are enough rows for a submatrix
	for(; i + num_submatrix_rows <= C.num_rows(); i += num_submatrix_rows)
	{
		int j = 0;
		// if there are enough columns for a submatrix we call submatrix()
		// otherwise we need to mask some columns at the trailing edge 
		// so we call submatrix_masked()
		for(; j + num_submatrix_cols <= C.num_cols(); j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix(A, B, C, i, j);
		}

		submatrix_masked(A, B, C, i, j);
	}

	// handle any leftover columns
	for(int j = 0; j < C.num_cols(); j += num_submatrix_cols)
	{
		submatrix_masked(A, B, C, i, j);
	}

	return C;
}

} // namespace thwmakos
