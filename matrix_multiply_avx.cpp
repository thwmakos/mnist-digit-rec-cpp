//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_mutliply_avx.cpp
//

#include "matrix_multiply_avx.hpp"


#include <immintrin.h>

#include <array>
#include <span>

// TODO: update submatrix functions with std::array and std::span

namespace thwmakos { 

template<typename  T, int num_rows, int num_cols>
using array2d = std::array<std::array<T, num_cols>, num_rows>;	

#ifdef __AVX512F__

// calculate the elements of the submatrix of C
// given by the rows in range [num_row, num_row + num_submatrix_rows)
// and columns in range [num_col, num_col + num_submatrix_cols
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
template<int num_submatrix_rows, int num_submatrix_cols, bool masked>
void submatrix_avx512(const matrix &A, const matrix &B, matrix &C, int num_row, int num_col)
{
	constexpr int num_lanes = 16;

	static_assert(num_submatrix_rows > 0 && num_submatrix_cols > 0);
	static_assert(num_submatrix_cols % num_lanes == 0);

	array2d<__m512, num_submatrix_rows, num_submatrix_cols / num_lanes> C_submatrix {};	
	__m512 a {};
	std::array<__m512, num_submatrix_cols / num_lanes>    b_row {};
	std::array<__mmask16, num_submatrix_cols / num_lanes> masks {};

	// get pointers to raw data first
	//std::span<const float> A_data { A.data().data(), A.data().size() };
	//std::span<const float> B_data { B.data().data(), B.data().size() };
	//std::span<float> C_data { C.data().data(), C.data().size() };
	const float *A_data = A.data().data();
	const float *B_data = B.data().data();
	float *C_data = C.data().data();
	
	// process an appropriately sized part of C, or less if that is not available 	
	const auto [actual_rows, actual_cols] = [&] {
		if constexpr(masked)
		{
			return std::pair { std::min(C.num_rows() - num_row, num_submatrix_rows),
					std::min(C.num_cols() - num_col, num_submatrix_cols) };
		}
		else
		{
			return std::pair { num_submatrix_rows, num_submatrix_cols };
		}
	} ();

	// calculate masks for data loading/storing for C and B
	if constexpr(masked)
	{
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
	}
	
	// we first load the submatrix of C into C_submatrix
	for(int i = 0; i < actual_rows; ++i)
	{
		const int C_load_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			if constexpr(masked)
			{
				if(masks[j] != 0)
				{
					C_submatrix[i][j] = _mm512_maskz_loadu_ps(masks[j], &C_data[C_load_index + num_lanes * j]);
				}
			}
			else
			{
				C_submatrix[i][j] = _mm512_loadu_ps(&C_data[C_load_index + num_lanes * j]);
			}
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
			if constexpr(masked)
			{
				if(masks[j] != 0)
				{
					b_row[j] = _mm512_maskz_loadu_ps(masks[j], &B_data[B_load_index + num_lanes * j]);
				}
			}
			else
			{
				b_row[j] = _mm512_loadu_ps(&B_data[B_load_index + num_lanes * j]);
			}
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
			if constexpr(masked)
			{
				if(masks[j] != 0)
				{
					_mm512_mask_storeu_ps(&C_data[C_store_index + num_lanes * j], masks[j], C_submatrix[i][j]);
				}
			}
			else
			{
				_mm512_storeu_ps(&C_data[C_store_index + num_lanes * j], C_submatrix[i][j]);
			}
		}
	}
}

matrix multiply_avx512(const matrix &A, const matrix &B)
{
	// the function only works with single precision floats
	static_assert(std::is_same_v<FloatType, float>);

	// matrix to be returned
	matrix C(A.num_rows(), B.num_cols());
	
	// we calculate the product C = A * B by computing
	// C in smaller submatrices of dimensions given
	// by the parameters below
	constexpr int num_lanes = 16;
	constexpr int num_submatrix_rows = 12;             // submatrix size needs to be adjusted to CPU
	constexpr int num_submatrix_cols = 8 * num_lanes; // these parameters provide 20x performance boost to
													  // naive implementation on intel tgl (i7 11800H CPU)
	// the number of columns is a multiple of 16 which 
	// is how many single precision floats an avx512 
	// register stores
	// 
	// calculating each submatrix is called calculating
	// a 'kernel' usually
	int i = 0;	
	
	// make sure there are enough rows for a submatrix
	for(; i + num_submatrix_rows <= C.num_rows(); i += num_submatrix_rows)
	{
		int j = 0;
		// if there are enough columns for a submatrix we call submatrix()
		// with masked set to false
		// otherwise we need to mask some columns at the trailing edge 
		// so we call submatrix with masked set to true
		for(; j + num_submatrix_cols <= C.num_cols(); j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix_avx512<num_submatrix_rows, num_submatrix_cols, false>(A, B, C, i, j);
		}

		submatrix_avx512<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	// handle any leftover columns
	for(int j = 0; j < C.num_cols(); j += num_submatrix_cols)
	{
		submatrix_avx512<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	return C;
}

#endif // __AVX512F__

#ifdef __AVX2__ 

// AVX2 implementation of previous functions
template<int num_submatrix_rows, int num_submatrix_cols, bool masked>
void submatrix_avx2(const matrix &A, const matrix &B, matrix &C, int num_row, int num_col)
{
	constexpr int num_lanes = 8;

	static_assert(num_submatrix_rows > 0 && num_submatrix_cols > 0);
	static_assert(num_submatrix_cols % num_lanes == 0);

	array2d<__m256, num_submatrix_rows, num_submatrix_cols / num_lanes> C_submatrix {};	
	__m256 a {};
	std::array<__m256, num_submatrix_cols / num_lanes>  b_row {};
	std::array<__m256i, num_submatrix_cols / num_lanes> masks {};

	// get pointers to raw data first 
	// can't use spans here because it is not simple to compare a mask to zero and compiler
	// complains for out of bounds access when all lanes are disabled in the mask
	const float *A_data = A.data().data();
	const float *B_data = B.data().data();
	float  *C_data =C.data().data();
	
	// process an appropriately sized part of C, or less if that is not available 	
	const auto [actual_rows, actual_cols] = [&] {
		if constexpr(masked)
		{
			return std::pair { std::min(C.num_rows() - num_row, num_submatrix_rows),
					std::min(C.num_cols() - num_col, num_submatrix_cols) };
		}
		else
		{
			return std::pair { num_submatrix_rows, num_submatrix_cols };
		}
	} ();
	
	// calculate masks for data loading/storing for C and B
	if constexpr(masked)
	{
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			const int lanes_remaining = actual_cols - j * num_lanes;

			if(lanes_remaining >= num_lanes)
			{
				// activate all 8 lanes
				// the most significant bit activates a lane
				masks[j] = _mm256_set1_epi32(0xFFFFFFFF);
			}
			else if(lanes_remaining > 0)
			{
				// from the bit mask below we need to convert to
				// AVX2 mask
				int32_t mask = (1 << lanes_remaining) - 1u;
				masks[j] = _mm256_set1_epi32(mask);
				__m256i c = _mm256_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7);
				masks[j] = _mm256_and_si256(masks[j], c);
				masks[j] = _mm256_cmpeq_epi32(masks[j], c);
			}
			else
			{
				masks[j] = _mm256_set1_epi32(0);
			}
		}
	}
	
	// we first load the submatrix of C into C_submatrix
	for(int i = 0; i < actual_rows; ++i)
	{
		const int C_load_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			if constexpr(masked)
			{
				// if non-zero mask 
				C_submatrix[i][j] = _mm256_maskload_ps(&C_data[C_load_index + num_lanes * j], masks[j]);
			}
			else
			{
				C_submatrix[i][j] = _mm256_loadu_ps(&C_data[C_load_index + num_lanes * j]);
			}
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
			if constexpr(masked)
			{
				b_row[j] = _mm256_maskload_ps(&B_data[B_load_index + num_lanes * j], masks[j]);
			}
			else
			{
				b_row[j] = _mm256_loadu_ps(&B_data[B_load_index + num_lanes * j]);
			}
		}

		// update each row of C_submatrix
		for(int n = 0; n < actual_rows; ++n)
		{
			const int A_load_index = A.index(num_row + n, i);
			a = _mm256_set1_ps(A_data[A_load_index]);

			for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
			{
				C_submatrix[n][j] = _mm256_fmadd_ps(a, b_row[j], C_submatrix[n][j]);
			}
		}
	}
	
	// store results back to C
	for(int i = 0; i < actual_rows; ++i)
	{
		const int C_store_index = C.index(num_row + i, num_col);

		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			if constexpr(masked)
			{
				_mm256_maskstore_ps(&C_data[C_store_index + num_lanes * j], masks[j], C_submatrix[i][j]);
			}
			else
			{
				_mm256_storeu_ps(&C_data[C_store_index + num_lanes * j], C_submatrix[i][j]);
			}
		}
	}
}

matrix multiply_avx2(const matrix &A, const matrix &B)
{
	// the function only works with single precision floats
	static_assert(std::is_same_v<FloatType, float>);

	// matrix to be returned
	matrix C(A.num_rows(), B.num_cols());
	
	// we calculate the product C = A * B by computing
	// C in smaller submatrices of dimensions given
	// by the parameters below
	constexpr int num_lanes = 8;
	constexpr int num_submatrix_rows = 12;             // submatrix size needs to be adjusted to CPU
	constexpr int num_submatrix_cols = 8 * num_lanes;  // found these values by trial & error 
													   
	// the number of columns is a multiple of num_lanes which 
	// is how many single precision floats an avx512 
	// register stores
	// 
	// calculating each submatrix is called calculating
	// a 'kernel' usually
	int i = 0;	
	
	// make sure there are enough rows for a submatrix
	for(; i + num_submatrix_rows <= C.num_rows(); i += num_submatrix_rows)
	{
		int j = 0;
		// if there are enough columns for a submatrix we call submatrix()
		// with masked set to false
		// otherwise we need to mask some columns at the trailing edge 
		// so we call submatrix with masked set to true
		for(; j + num_submatrix_cols <= C.num_cols(); j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix_avx2<num_submatrix_rows, num_submatrix_cols, false>(A, B, C, i, j);
		}

		submatrix_avx2<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	// handle any leftover columns
	for(int j = 0; j < C.num_cols(); j += num_submatrix_cols)
	{
		submatrix_avx2<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	return C;
}

#endif // __AVX2__
	
// fallback scalar implementation
matrix multiply_naive(const matrix& left, const matrix& right)
{
	// store these so we don't have to call num_rows() and num_cols()
	// all the time, idk if this speeds up the functions, the above
	// functions should be inlined anyway 
	const auto [left_num_rows, left_num_cols]   = left.size();
	const auto [right_num_rows, right_num_cols] = right.size();

	matrix product(left_num_rows, right_num_cols);
	
	// naive implementation of matrix multiplication
	// TODO: good learning opportunity for intrinsics here
	
	// transpose right first to ensure sequential access 
	// to matrix elements
	// uses more memory but is ~4 times faster for large matrices
	auto right_transpose = transpose(right);
	
	for(auto i = 0; i < left_num_rows; ++i)
	{
		for(auto j = 0; j < right_num_cols; ++j)
		{
			// equivalently we could check against right_num_rows
			FloatType accumulator {}; 
			
			for(auto k = 0; k < left_num_cols; ++k)
			{
				accumulator += left[i, k] * right_transpose[j, k];
			}

			product[i, j] = accumulator; 
		}
	}

	return product;
}

} // namespace thwmakos

