#ifndef UBLAS_ADAPTER_H_INCLUDED
#define UBLAS_ADAPTER_H_INCLUDED

// BLAS operations for tiled cholkesy
// Cholesky decomposition of A -> return factorized matrix L
template <typename T>
std::vector<T> potrf(std::vector<T> A,
                     std::size_t N);
// solve L * X = A^T where L triangular
template <typename T>
std::vector<T> trsm(std::vector<T> L,
                    std::vector<T> A,
                    std::size_t N);
//  A = A - B * B^T
template <typename T>
std::vector<T> syrk(std::vector<T> A,
                    std::vector<T> B,
                    std::size_t N);
//C = C - A * B^T
template <typename T>
std::vector<T> gemm(std::vector<T> A,
                    std::vector<T> B,
                    std::vector<T> C,
                    std::size_t N);

// BLAS operations for tiled triangular solve
// solve L * x = a where L lower triangular
template <typename T>
std::vector<T> trsm_l(std::vector<T> L,
                      std::vector<T> a,
                      std::size_t N);
// solve L^T * x = a where L lower triangular
template <typename T>
std::vector<T> trsm_u(std::vector<T> L,
                      std::vector<T> a,
                      std::size_t N);
// b = b - A * a
template <typename T>
std::vector<T> gemv_l(std::vector<T> A,
                      std::vector<T> a,
                      std::vector<T> b,
                      std::size_t N);
// b = b - A^T * a
template <typename T>
std::vector<T> gemv_u(std::vector<T> A,
                      std::vector<T> a,
                      std::vector<T> b,
                      std::size_t N);

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
template <typename T>
std::vector<T> gemv_p(std::vector<T> A,
                      std::vector<T> a,
                      std::vector<T> b,
                      std::size_t N_row,
                      std::size_t N_col);
// ||a - b||^2
template <typename T>
T norm_2(std::vector<T> a,
         std::vector<T> b,
         std::size_t N);

#endif
