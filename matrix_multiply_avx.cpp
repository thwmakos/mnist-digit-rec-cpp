//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_mutliply_avx.cpp
//

#include "matrix_multiply_avx.hpp"

#include <immintrin.h>

namespace thwmakos { 

matrix multiply_avx512(const matrix &A, const matrix &B)
{
	// matrix to be returned
	matrix C(A.num_rows(), B.num_cols());

	return C;
}

} // namespace thwmakos
