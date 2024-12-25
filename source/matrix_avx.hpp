//
// ~thwmakos~
//
// 27/9/2024
//
// matrix_mutliply_avx.hpp
//

#ifndef MATRIX_MULTIPLY_AVX_INCLUDED
#define MATRIX_MULTIPLY_AVX_INCLUDED

#include "matrix.hpp"

namespace thwmakos {

//
// calculate C = A * B matrix multiplication using AVX512 intrinsics 
//
// matrices are assumed to have
// appropriate dimensions, should be from the caller of this function
matrix multiply_avx512(const matrix &A, const matrix &B);
matrix multiply_avx2(const matrix &A, const matrix &B);

} // namespace thwmakos

#endif
