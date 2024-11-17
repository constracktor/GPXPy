#ifndef CUBLAS_ADAPTER_H
#define CUBLAS_ADAPTER_H

#include <hpx/local/future.hpp>
#include <hpx/modules/async_cuda.hpp>

// BLAS operations on GPU with cuBLAS (and cuSOLVER)

// BLAS operations for tiled cholkesy -------------------------------------- {{{

/**
 * In-place cholesky decomposition on tile of A
 */
hpx::shared_future<std::vector<double>> potrf(
    hpx::cuda::experimental::cublas_executor& cublas,
    hpx::shared_future<std::vector<double>> A, std::size_t N);

/**
 * Solve the triangular linear system with multiple right-hand-sides (TRSM)
 * inplace for lower triangular L: X * L^T = A
 */
hpx::shared_future<std::vector<double>> trsm(
    hpx::cuda::experimental::cublas_executor& cublas,
    hpx::shared_future<std::vector<double>> L,
    hpx::shared_future<std::vector<double>> A, std::size_t N);

/**
 * Calculate symmetric rank-k update (SYRK): A = A - B * B^T
 */
hpx::shared_future<std::vector<double>> syrk(
    hpx::cuda::experimental::cublas_executor& cublas,
    hpx::shared_future<std::vector<double>> A,
    hpx::shared_future<std::vector<double>> B, std::size_t N);

/**
 * Calculate general matrix-matrix multiplication (GEMM): C = C - A * B^T
 */
hpx::shared_future<std::vector<double>> gemm(
    hpx::cuda::experimental::cublas_executor& cublas,
    hpx::shared_future<std::vector<double>> A,
    hpx::shared_future<std::vector<double>> B,
    hpx::shared_future<std::vector<double>> C, std::size_t N);

// }}} ------------------------------- end of BLAS operations for tiled cholkesy

// TODO: add other BLAS operations

#endif
