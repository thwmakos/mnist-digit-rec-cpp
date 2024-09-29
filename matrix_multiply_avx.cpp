//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_mutliply_avx.cpp
//

#include "matrix_multiply_avx.hpp"

#include <cassert>
#include <array>
#include <immintrin.h>

#include <print>

namespace thwmakos { 

// we calculate the product C = A * B by computing
// C in smaller submatrices of dimensions given
// by the parameters below
constexpr int num_submatrix_rows = 8;      // submatrix size needs to be adjusted to CPU
constexpr int num_submatrix_cols = 4 * 16; // these parameters provide 20x performance boots to
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
	
	constexpr int num_lanes = 16;

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
		int C_load_index = C.index(num_row + i, num_col);

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
		int B_load_index = B.index(i, num_col);
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			b_row[j] = _mm512_loadu_ps(&B_data[B_load_index + num_lanes * j]);
		}

		// update each row of C_submatrix
		for(int n = 0; n < num_submatrix_rows; ++n)
		{
			int A_load_index = A.index(num_row + n, i);
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
		int C_store_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			_mm512_storeu_ps(&C_data[C_store_index + num_lanes * j], C_submatrix[i][j]);
		}
	}
	
}

matrix multiply_avx512(const matrix &A, const matrix &B)
{
	// the function only works with single precision floats
	static_assert(std::is_same_v<FloatType, float>);

	// no maskings loads/stores introduced yet, so we assume
	// matrix dimensions satisfy the following for now
	assert(A.num_rows() % num_submatrix_rows == 0);
	assert(B.num_cols() % num_submatrix_cols == 0);

	// matrix to be returned
	matrix C(A.num_rows(), B.num_cols());
	
	for(int i = 0; i < C.num_rows(); i += num_submatrix_rows)
	{
		for(int j = 0; j < C.num_cols(); j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix(A, B, C, i, j);
		}
	}

	return C;
}

} // namespace thwmakos
