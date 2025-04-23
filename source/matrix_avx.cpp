//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_avx.cpp
//

#include "matrix_avx.hpp"
#include "matrix_view.hpp"

#ifdef __x86_64
#include <immintrin.h>
#endif

#include <array>
#include <span>
#include <cassert>

// TODO: update submatrix functions with std::array and std::span

namespace thwmakos { 

// the functions only work with single precision floats
static_assert(std::is_same_v<FloatType, float>);

template<typename  T, int num_rows, int num_cols>
using array2d = std::array<std::array<T, num_cols>, num_rows>;	

#ifdef __AVX512F__

namespace avx512
{
	constexpr int num_lanes = 16; // a ZMM register hold 16 4-byte floats

	// the result matrix C = A * B is split into blocks and 
	// each block into is further split into tiles

	// ideally we want:
	// -- num_A_block_cols (equivalently num_B_block_rows) by num_block_rows to fit in the L3 cache
	// -- num_block_cols by num_A_block_cols to fit in the L2 cache
	// -- num_A_block_cols by num_tile_rows to fit in the L1 cache
	constexpr int num_tile_rows = 8;
	constexpr int num_tile_cols = 5 * num_lanes;

	// a num_time_rows by num_tile_cols sized
	constexpr int num_block_rows = num_tile_rows * 128;
	constexpr int num_block_cols = num_tile_cols * 4;
		
	constexpr int num_A_block_cols = 1024;
	constexpr int num_B_block_rows = num_A_block_cols;

	static_assert(num_tile_cols % num_lanes == 0);
}

// assume that A, B are padded and we do not need masks 
// masks are passed as argument; either all of them will be fully enabled
// or partially enabled if the tile overflows on the right of C but they
// will be the same for all call to this function, hence only need to be
// calculated once from the caller
template<int num_tile_rows, int num_tile_cols>
	requires (num_tile_rows > 0 && num_tile_cols > 0 && num_tile_cols % avx512::num_lanes == 0)
void muliply_tile_avx512(matrix_view C, const_matrix_view A, const_matrix_view B,
		const std::array<__mmask16, num_tile_cols / avx512::num_lanes> &masks)
{
	constexpr int num_col_regs = num_tile_cols / avx512::num_lanes;

	// no need to zero-initialise these, we will load values into them anyway
	array2d<__m512, num_tile_rows, num_col_regs> C_tile;
	__m512 a;
	std::array<__m512, num_col_regs> b_row;
	
	const int actual_rows = std::min(C.num_rows, num_tile_rows);

	// load C_tile into registers, a row at a time
	for(int m = 0; m < actual_rows; ++m)
	{
		for(int j = 0; j < num_col_regs; ++j)
		{
			C_tile[m][j] = _mm512_maskz_loadu_ps(masks[j], C.address(m, avx512::num_lanes * j));
		}
	}

	// B.num_rows should be equal to A.num_cols
	assert(B.num_rows == A.num_cols);	

	// loop over every row of C_tile calculating its value
	// to do this loop over the rows of B as described above
	for(int k = 0; k < B.num_rows; ++k)
	{
		// first load num_tile_cols elements from the current row of B
		for(int j = 0; j < num_col_regs; ++j)
		{
			b_row[j] = _mm512_loadu_ps(B.address(k, avx512::num_lanes * j));
		}

		// update a row of C_tile
		for(int m = 0; m < actual_rows; ++m)
		{
			a = _mm512_set1_ps(*A.address(m, k));

			for(int j = 0; j < num_col_regs; ++j)
			{
				C_tile[m][j] = _mm512_fmadd_ps(a, b_row[j], C_tile[m][j]);
			}
		}
	}
	
	// store updated entries back to C
	for(int m = 0; m < actual_rows; ++m)
	{
		for(int j = 0; j < num_col_regs; ++j)
		{
			_mm512_mask_storeu_ps(C.address(m, avx512::num_lanes * j), masks[j], C_tile[m][j]);
		}
	}
}

// copy the view source into the dest span
// source must fit into dest
// if dest is larger, the extra entries are zeroed
void load_block(const_matrix_view source, matrix_span dest)
{
	assert(dest.num_rows >= source.num_rows && dest.num_columns >= source.num_cols);

	for(int i = 0; i < source.num_rows; ++i)
	{
		for(int j = 0; j < source.num_cols; ++j)
		{
			dest[i, j] = source[i, j];
		}

		for(int j = source.num_cols; j < dest.num_columns; ++j)
		{
			dest[i, j] = 0.0;
		}
	}	

	for(int i = source.num_rows; i < dest.num_rows; ++i)
	{
		for(int j = 0; j < dest.num_columns; ++j)
		{
			dest[i, j] = 0.0;
		}
	}
}

// calculate the matrix product C = A * B
// C    -- where the result will be stored
// A, B -- input matrix spans to be multiplied
// dimensions should be checked by caller
//
// dimensions of C, A, B are m by n, m by k, k by n respectively
// hence the names of the counter variables in the loop
void multiply_avx512_blocked_tiled(matrix_span C, const_matrix_span A, const_matrix_span B)
{
	// we want this to fit in L2 cache
	static std::array<FloatType, avx512::num_block_rows * avx512::num_A_block_cols> A_cache_storage {};
	// we want this to fit in L3 cache
	static std::array<FloatType, avx512::num_B_block_rows * avx512::num_block_cols> B_cache_storage {};

	matrix_span A_cache { avx512::num_block_rows, avx512::num_A_block_cols, A_cache_storage };
	matrix_span B_cache { avx512::num_B_block_rows, avx512::num_block_cols, B_cache_storage };

	// trivial masks can be computed at compile time!
	constexpr auto trivial_masks = []
	{
		std::array<__mmask16, avx512::num_tile_cols / avx512::num_lanes> arr;
		arr.fill(0b1111111111111111); // enable all the lanes
		return arr;
	} ();


	const auto nontrivial_masks = [&C]
	{
		std::array<__mmask16, avx512::num_tile_cols / avx512::num_lanes> arr;
		
		// available lanes at the right edge C		
		const int available_lanes = C.num_columns % avx512::num_tile_cols;
		
		for(int j = 0; j < avx512::num_tile_cols / avx512::num_lanes; ++j)
		{
			const int lanes_remaining = available_lanes - j * avx512::num_lanes;

			if(lanes_remaining >= avx512::num_lanes)
			{
				// TODO: make this constant generic (determine it from num_lanes)
				arr[j] = (1u << avx512::num_lanes) - 1u;
			}
			else if(lanes_remaining > 0)
			{
				arr[j] = (1u << lanes_remaining) - 1u;
			}
			else
			{
				arr[j] = 0u;
			}
		}

		return arr;
	}();

	for(int n = 0; n < C.num_columns; n += avx512::num_block_cols)
	{
		const int block_num_cols = std::min(avx512::num_block_cols, C.num_columns - n);

		for(int k = 0; k < B.num_rows; k += avx512::num_B_block_rows)
		{
			// actual rows available for block of B
			// equivalently, actual columns available for the block of A
			const int block_middle_num = std::min(avx512::num_B_block_rows, B.num_rows - k);

			const_matrix_view B_block { B, k, n, block_middle_num, block_num_cols };
			load_block(B_block, B_cache);

			for(int m = 0; m < C.num_rows; m += avx512::num_block_rows)
			{
				const int block_num_rows = std::min(avx512::num_block_rows, C.num_rows - m);

				const_matrix_view A_block { A, m, k, block_num_rows, block_middle_num };
				load_block(A_block, A_cache);

				// accumulate tiles from the (m, n)-th block of C
				for(int tile_n = 0; tile_n < block_num_cols; tile_n += avx512::num_tile_cols)
				{
					const int num_tile_cols = std::min(avx512::num_tile_cols, block_num_cols - tile_n);

					const auto &masks = (num_tile_cols == avx512::num_tile_cols) ? trivial_masks : nontrivial_masks;

					for(int tile_m = 0; tile_m < block_num_rows; tile_m += avx512::num_tile_rows)
					{
						const int num_tile_rows = std::min(avx512::num_tile_rows, block_num_rows - tile_m);
						
						// prepare matrix view to be passed for multiplication
						matrix_view C_tile { C, m + tile_m, n + tile_n, num_tile_rows, num_tile_cols };
						const_matrix_view A_tile { A_cache, tile_m, 0, num_tile_rows, block_middle_num };
						const_matrix_view B_tile { B_cache, 0, tile_n, block_middle_num, num_tile_cols };

						muliply_tile_avx512<avx512::num_tile_rows, avx512::num_tile_cols>(C_tile, A_tile, B_tile, masks);
					}
				}
			}
		}
	}
}



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
void submatrix_avx512(const_matrix_span A, const_matrix_span B, matrix_span C, int num_row, int num_col)
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
	const float *A_data = A.data.data(); 	
	const float *B_data = B.data.data();
	float *C_data = C.data.data();
	
	// process an appropriately sized part of C, or less if that is not available 	
	const auto [actual_rows, actual_cols] = [&] {
		if constexpr(masked)
		{
			return std::pair { std::min(C.num_rows - num_row, num_submatrix_rows),
					std::min(C.num_columns - num_col, num_submatrix_cols) };
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
					C_submatrix[i][j] = _mm512_maskz_loadu_ps(masks[j], C_data + C_load_index + num_lanes * j);
				}
			}
			else
			{
				C_submatrix[i][j] = _mm512_loadu_ps(C_data + C_load_index + num_lanes * j);
			}
		}
	}
	
	// loop over every row of C_submatrix calculating its value
	// to do this loop over the rows of B as described above
	for(int i = 0; i < B.num_rows; ++i)
	{
		// load num_submatrix_cols elements from the current row of B
		const int B_load_index = B.index(i, num_col);
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			if constexpr(masked)
			{
				if(masks[j] != 0)
				{
					b_row[j] = _mm512_maskz_loadu_ps(masks[j], B_data + B_load_index + num_lanes * j);
				}
			}
			else
			{
				b_row[j] = _mm512_loadu_ps(B_data + B_load_index + num_lanes * j);
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
					_mm512_mask_storeu_ps(C_data + C_store_index + num_lanes * j, masks[j], C_submatrix[i][j]);
				}
			}
			else
			{
				_mm512_storeu_ps(C_data + C_store_index + num_lanes * j, C_submatrix[i][j]);
			}
		}
	}
}

void multiply_avx512(matrix_span C, const_matrix_span A, const_matrix_span B)
{
	// we calculate the product C = A * B by computing
	// C in smaller submatrices of dimensions given
	// by the parameters below
	constexpr int num_lanes = 16;
	constexpr int num_submatrix_rows = 8;             // submatrix size needs to be adjusted to CPU
													  // and matrix sizes
	constexpr int num_submatrix_cols = 3 * num_lanes; // these parameters provide 20x performance boost to
													  // naive implementation on intel tgl (i7 11800H CPU)
	// the number of columns is a multiple of 16 which 
	// is how many single precision floats an avx512 
	// register stores
	int i = 0;	
	
	// make sure there are enough rows for a submatrix
	for(; i + num_submatrix_rows <= C.num_rows; i += num_submatrix_rows)
	{
		int j = 0;
		// if there are enough columns for a submatrix we call submatrix()
		// with masked set to false
		// otherwise we need to mask some columns at the trailing edge 
		// so we call submatrix with masked set to true
		for(; j + num_submatrix_cols <= C.num_columns; j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix_avx512<num_submatrix_rows, num_submatrix_cols, false>(A, B, C, i, j);
		}

		submatrix_avx512<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	// handle any leftover columns
	for(int j = 0; j < C.num_columns; j += num_submatrix_cols)
	{
		submatrix_avx512<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}
}

// does left += right and then returns left
// dimensions of dimensions are checked in caller
// processes N * num_lanes elements every iteration
template<int N>
void add_to_avx512_internal(matrix_span left, const_matrix_span right, FloatType scalar)
{
	constexpr int num_lanes = 16;
	
	float       *left_data  = left.data.data();
	const float *right_data = right.data.data();
	
	const int num_elements = static_cast<int>(left.data.size()); 

	int i = 0;

	std::array<__m512, N> left_regs;
	std::array<__m512, N> right_regs;
	__m512 scalar_reg = _mm512_set1_ps(scalar);

	for(; i + N * num_lanes < num_elements; i += N * num_lanes)
	{
		for(int j = 0; j < N; ++j)
		{
			left_regs[j]  = _mm512_loadu_ps(left_data + i * j);
			right_regs[j] = _mm512_loadu_ps(right_data + i * j);
		}
		
		for(int j = 0; j < N; ++j)
		{
			left_regs[j] = _mm512_fmadd_ps(scalar_reg, right_regs[j], left_regs[j]);
		}
		
		for(int j = 0; j < N; ++j)
		{
			_mm512_storeu_ps(left_data + i * j, left_regs[j]);
		}
	}
	
	// if there are less that N * num_lanes but more than num_lanes elements
	// process them by num_lanes at once	
	for(; i + num_lanes < num_elements; i += num_lanes)
	{
		__m512 left_reg  = _mm512_loadu_ps(left_data + i);
		__m512 right_reg = _mm512_loadu_ps(right_data + i); 

		left_reg = _mm512_fmadd_ps(scalar_reg, right_reg, left_reg);

		_mm512_storeu_ps(left_data + i, left_reg);
	}
	
	// handle the rest of the elements
	__mmask16 mask = (1u << (num_elements - i)) - 1u;

	__m512 left_reg  = _mm512_maskz_loadu_ps(mask, left_data + i);
	__m512 right_reg = _mm512_maskz_loadu_ps(mask, right_data + i);

	left_reg = _mm512_fmadd_ps(scalar_reg, right_reg, left_reg);

	_mm512_mask_storeu_ps(left_data + i, mask, left_reg);
}

// performs the operation left[i, j] += scalar * right[i, j]
void add_to_avx512(matrix_span left, const_matrix_span right, FloatType scalar)
{
	// testing shows that changing 1 below to a different 
	// value is not beneficial
	return add_to_avx512_internal<1>(left, right, scalar);
}


void scalar_multiply_avx512(matrix_span mat, FloatType scalar)
{
	constexpr int num_lanes = 16;

	float *mat_data = mat.data.data();
	const int num_elements = static_cast<int>(mat.data.size());

	int i = 0;
	
	__m512 scalar_reg = _mm512_set1_ps(scalar);

	for(; i + num_lanes < num_elements; i += num_lanes)
	{
		__m512 mat_reg = _mm512_loadu_ps(mat_data + i);
		mat_reg = _mm512_mul_ps(mat_reg, scalar_reg);
		_mm512_storeu_ps(mat_data + i, mat_reg);
	}

	// handle tail
	__mmask16 mask = (1u << (num_elements - i)) - 1u;
	__m512 mat_reg = _mm512_maskz_loadu_ps(mask, mat_data + i);
	mat_reg = _mm512_mul_ps(mat_reg, scalar_reg);
	_mm512_mask_storeu_ps(mat_data + i, mask, mat_reg);
}

#endif // __AVX512F__

#ifdef __AVX2__ 

// from the bit mask below, we convert to the 8 rightmost most bits to an AVX2 mask
__m256i make_avx2_mask(int32_t bitmask)
{
	//int32_t mask = (1 << lanes_remaining) - 1u;
	__m256i mask = _mm256_set1_epi32(bitmask);
	__m256i c = _mm256_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7);
	mask = _mm256_and_si256(mask, c);
	mask = _mm256_cmpeq_epi32(mask, c);

	return mask;
}

// AVX2 implementation of previous functions
template<int num_submatrix_rows, int num_submatrix_cols, bool masked>
void submatrix_avx2(const_matrix_span A, const_matrix_span B, matrix_span C, int num_row, int num_col)
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
	const float *A_data = A.data.data();
	const float *B_data = B.data.data();
	float  *C_data = C.data.data();
	
	// process an appropriately sized part of C, or less if that is not available 	
	const auto [actual_rows, actual_cols] = [&] {
		if constexpr(masked)
		{
			return std::pair { std::min(C.num_rows - num_row, num_submatrix_rows),
					std::min(C.num_columns - num_col, num_submatrix_cols) };
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
				masks[j] = make_avx2_mask((1 << lanes_remaining) - 1u);
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
				C_submatrix[i][j] = _mm256_maskload_ps(C_data + C_load_index + num_lanes * j, masks[j]);
			}
			else
			{
				C_submatrix[i][j] = _mm256_loadu_ps(C_data + C_load_index + num_lanes * j);
			}
		}
	}
	
	// loop over every row of C_submatrix calculating its value
	// to do this loop over the rows of B as described above
	for(int i = 0; i < B.num_rows; ++i)
	{
		// load num_submatrix_cols elements from the current row of B
		const int B_load_index = B.index(i, num_col);
		for(int j = 0; j < num_submatrix_cols / num_lanes; ++j)
		{
			if constexpr(masked)
			{
				b_row[j] = _mm256_maskload_ps(B_data + B_load_index + num_lanes * j, masks[j]);
			}
			else
			{
				b_row[j] = _mm256_loadu_ps(B_data + B_load_index + num_lanes * j);
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
				_mm256_maskstore_ps(C_data + C_store_index + num_lanes * j, masks[j], C_submatrix[i][j]);
			}
			else
			{
				_mm256_storeu_ps(C_data + C_store_index + num_lanes * j, C_submatrix[i][j]);
			}
		}
	}
}

void multiply_avx2(matrix_span C, const_matrix_span A, const_matrix_span B)
{
	// the function only works with single precision floats
	static_assert(std::is_same_v<FloatType, float>);
	
	// we calculate the product C = A * B by computing
	// C in smaller submatrices of dimensions given
	// by the parameters below
	constexpr int num_lanes = 8;
	constexpr int num_submatrix_rows = 8;             // submatrix size needs to be adjusted to CPU
	constexpr int num_submatrix_cols = 4 * num_lanes;  // found these values by trial & error 
													   
	// the number of columns is a multiple of num_lanes which 
	// is how many single precision floats an avx512 
	// register stores
	// 
	// calculating each submatrix is called calculating
	// a 'kernel' usually
	int i = 0;	
	
	// make sure there are enough rows for a submatrix
	for(; i + num_submatrix_rows <= C.num_rows; i += num_submatrix_rows)
	{
		int j = 0;
		// if there are enough columns for a submatrix we call submatrix()
		// with masked set to false
		// otherwise we need to mask some columns at the trailing edge 
		// so we call submatrix with masked set to true
		for(; j + num_submatrix_cols <= C.num_columns; j += num_submatrix_cols)
		{
			//std::println("submatrix starting at {}, {}", i, j);
			submatrix_avx2<num_submatrix_rows, num_submatrix_cols, false>(A, B, C, i, j);
		}

		submatrix_avx2<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}

	// handle any leftover columns
	for(int j = 0; j < C.num_columns; j += num_submatrix_cols)
	{
		submatrix_avx2<num_submatrix_rows, num_submatrix_cols, true>(A, B, C, i, j);
	}
}

void scalar_multiply_avx2(matrix_span mat, FloatType scalar)
{
	constexpr int num_lanes = 8;

	float *mat_data = mat.data.data();
	const int num_elements = static_cast<int>(mat.data.size());

	int i = 0;
	
	__m256 scalar_reg = _mm256_set1_ps(scalar);

	for(; i + num_lanes < num_elements; i += num_lanes)
	{
		__m256 mat_reg = _mm256_loadu_ps(mat_data + i);
		mat_reg = _mm256_mul_ps(mat_reg, scalar_reg);
		_mm256_storeu_ps(mat_data + i, mat_reg);
	}

	// handle tail
	__m256i mask = make_avx2_mask((1u << (num_elements - i)) - 1u);
	__m256 mat_reg = _mm256_maskload_ps(mat_data + i, mask);
	mat_reg = _mm256_mul_ps(mat_reg, scalar_reg);
	_mm256_maskstore_ps(mat_data + i, mask, mat_reg);
}

#endif // __AVX2__
	
} // namespace thwmakos

