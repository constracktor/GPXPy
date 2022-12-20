#ifndef UBLAS_ADAPTER_H_INCLUDED
#define UBLAS_ADAPTER_H_INCLUDED

//#define CALC_TYPE double
#define CALC_TYPE float

// BLAS operations for tiled cholkesy
// Cholesky decomposition of A -> return factorized matrix L
std::vector<CALC_TYPE> potrf(std::vector<CALC_TYPE> A,
                             std::size_t N);
// solve L * X = A^T where L triangular
std::vector<CALC_TYPE> trsm(std::vector<CALC_TYPE> L,
                            std::vector<CALC_TYPE> A,
                            std::size_t N);
//  A = A - B * B^T
std::vector<CALC_TYPE> syrk(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::size_t N);
//C = C - A * B^T
std::vector<CALC_TYPE> gemm(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::vector<CALC_TYPE> C,
                            std::size_t N);

// BLAS operations for tiled triangular solve
// solve L * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_l(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N);
// solve L^T * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_u(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N);
// b = b - A * a
std::vector<CALC_TYPE> gemv_l(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N);
// b = b - A^T * a
std::vector<CALC_TYPE> gemv_u(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N);

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<CALC_TYPE> gemv_p(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N_row,
                            std::size_t N_col);
// ||a - b||^2
CALC_TYPE norm_2(std::vector<CALC_TYPE> a,
                 std::vector<CALC_TYPE> b,
                 std::size_t N);

#endif
