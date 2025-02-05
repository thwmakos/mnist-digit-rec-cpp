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
void multiply_avx512(matrix_span C, const_matrix_span A, const_matrix_span B);
void multiply_avx2(matrix_span C, const_matrix_span A, const_matrix_span B);

matrix &add_to_avx512(matrix &, const matrix &);

} // namespace thwmakos

#endif
