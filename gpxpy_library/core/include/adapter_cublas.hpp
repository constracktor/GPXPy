#ifndef ADAPTER_CUBLAS_H
#define ADAPTER_CUBLAS_H

#include <hpx/local/future.hpp>
#include <hpx/modules/async_cuda.hpp>

// =============================================================================
// BLAS operations on GPU with cuBLAS (and cuSOLVER)
// =============================================================================

// BLAS operations for tiled cholkesy -------------------------------------- {{{

/**
 * @brief In-place cholesky decomposition on tile of A
 */
std::vector<double> potrf(hpx::cuda::experimental::cublas_executor& cublas,
                          std::vector<double> A,
                          std::size_t N);

/**
 * @brief Solve the triangular linear system with multiple right-hand-sides
 *        (TRSM) inplace for lower triangular L: X * L^T = A
 */
hpx::shared_future<std::vector<double>>
trsm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> L,
     hpx::shared_future<std::vector<double>> A,
     std::size_t N);

/**
 * @brief Calculate symmetric rank-k update (SYRK): A = A - B * B^T
 */
hpx::shared_future<std::vector<double>>
syrk(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> A,
     hpx::shared_future<std::vector<double>> B,
     std::size_t N);

/**
 * @brief Calculate general matrix-matrix multiplication (GEMM): C = C - A * B^T
 */
hpx::shared_future<std::vector<double>>
gemm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> A,
     hpx::shared_future<std::vector<double>> B,
     hpx::shared_future<std::vector<double>> C,
     std::size_t N);

// in-place solve L * x = a where L lower triangular
std::vector<double>
trsv_l(std::vector<double> L, std::vector<double> a, std::size_t N);

// b = b - A * a
std::vector<double> gemv_l(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N);

// in-place solve L^T * x = a where L lower triangular
std::vector<double>
trsv_u(std::vector<double> L, std::vector<double> a, std::size_t N);

// b = b - A^T * a
std::vector<double> gemv_u(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N);

// A = y*beta^T + A
std::vector<double> ger(std::vector<double> A,
                        std::vector<double> x,
                        std::vector<double> y,
                        std::size_t N);

// C = C + A * B^T
std::vector<double> gemm_diag(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> C,
                              std::size_t N);

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<double> gemv_p(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N_row,
                           std::size_t N_col);

// }}} ------------------------------- end of BLAS operations for tiled cholkesy

// BLAS operations used in uncertainty computation ------------------------- {{{

// in-place solve X * L = A where L lower triangular
std::vector<double> trsm_l_KcK(std::vector<double> L,
                               std::vector<double> A,
                               std::size_t N,
                               std::size_t M);

// C = C - A * B
std::vector<double> gemm_l_KcK(std::vector<double> A,
                               std::vector<double> B,
                               std::vector<double> C,
                               std::size_t N,
                               std::size_t M);

// C = C - A^T * B
std::vector<double> gemm_cross_tcross_matrix(std::vector<double> A,
                                             std::vector<double> B,
                                             std::vector<double> C,
                                             std::size_t N,
                                             std::size_t M);

// }}} --------------------------------- end of BLAS for uncertainty computation

// BLAS operations used in optimization step ------------------------------- {{{

// in-place solve L * X = A where L lower triangular
std::vector<double> trsm_l_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M);

// C = C - A * B
std::vector<double> gemm_l_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N,
                                  std::size_t M);

// in-place solve L^T * X = A where L upper triangular
std::vector<double> trsm_u_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M);

// C = C - A^T * B
std::vector<double> gemm_u_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N,
                                  std::size_t M);

// Dot product used in dot calculation
double dot(std::size_t N, std::vector<double> A, std::vector<double> B);

// C = C - A * B
std::vector<double> dot_uncertainty(std::vector<double> A,
                                    std::vector<double> R,
                                    std::size_t N,
                                    std::size_t M);

// C = C - A * B
std::vector<double> gemm_grad(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> R,
                              std::size_t N,
                              std::size_t M);

// }}} --------------------------------------- end of BLAS for optimization step

// TODO: add other BLAS operations

#endif // end of ADAPTER_CUBLAS_H
