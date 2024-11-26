#ifndef ADAPTER_CUBLAS_H
#define ADAPTER_CUBLAS_H

#include <hpx/local/future.hpp>
#include <hpx/modules/async_cuda.hpp>

// Constants are compatible with cuBLAS
typedef enum BLAS_TRANSPOSE {
    Blas_no_trans = 0,
    Blas_trans = 1
} BLAS_TRANSPOSE;
typedef enum BLAS_SIDE { Blas_left = 0, Blas_right = 1 } BLAS_SIDE;
typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// =============================================================================
// BLAS operations on GPU with cuBLAS (and cuSOLVER)
// =============================================================================

// BLAS level 3 operations ------------------------------------------------- {{{

/**
 * @brief In-place Cholesky decomposition of A
 *
 * Assumes that cuda stream has already been created with
 * cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) and will be
 * destroyed by the caller.
 *
 * @param
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 *
 * @return factorized, lower triangular matrix f_L
 */
hpx::shared_future<std::vector<double>>
potrf(cudaStream_t stream,
      hpx::shared_future<std::vector<double>> f_A,
      const std::size_t N);

/**
 * @brief In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 *
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 *
 * @return solution matrix f_X
 */
hpx::shared_future<std::vector<double>>
trsm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> f_L,
     hpx::shared_future<std::vector<double>> f_A,
     const std::size_t N,
     const std::size_t M,
     const BLAS_TRANSPOSE transpose_L,
     const BLAS_SIDE side_L);

/**
 * @brief Symmetric rank-k update: A = A - B * B^T
 *
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 *
 * @return updated matrix f_A
 */
hpx::shared_future<std::vector<double>>
syrk(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> f_A,
     hpx::shared_future<std::vector<double>> f_B,
     const std::size_t N);

/**
 * @brief General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 *
 * @param f_C Base matrix
 * @param f_B Right update matrix
 * @param f_A Left update matrix
 * @param M Number of rows of matrix A
 * @param N Number of columns of matrix B
 * @param K Number of columns of matrix A / rows of matrix B
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 *
 * @return updated matrix f_X
 */
hpx::shared_future<std::vector<double>>
gemm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> f_A,
     hpx::shared_future<std::vector<double>> f_B,
     hpx::shared_future<std::vector<double>> f_C,
     const std::size_t M,
     const std::size_t N,
     const std::size_t K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B);

// }}} ------------------------------------------ end of BLAS level 3 operations

// BLAS level 2 operations ------------------------------------------------- {{{

/**
 * @brief In-place solve L(^T) * x = a where L lower triangular
 *
 * @param f_L Cholesky factor matrix
 * @param f_a right hand side vector
 * @param N matrix dimension
 * @param transpose_L transpose Cholesky factor
 *
 * @return solution vector f_x
 */
hpx::shared_future<std::vector<double>>
trsv(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> f_L,
     hpx::shared_future<std::vector<double>> f_a,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_L);

/**
 * @brief General matrix-vector multiplication: b = b - A(^T) * a
 *
 * @param f_A update matrix
 * @param f_a update vector
 * @param f_b base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param transpose_A transpose update matrix
 *
 * @return updated vector f_b
 */
hpx::shared_future<std::vector<double>>
gemv(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> f_A,
     hpx::shared_future<std::vector<double>> f_a,
     hpx::shared_future<std::vector<double>> f_b,
     const std::size_t N,
     const std::size_t M,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A);

/**
 * @brief General matrix rank-1 update: A = A - x*y^T
 *
 * @param f_A base matrix
 * @param f_x first update vector
 * @param f_y second update vector
 * @param N matrix dimension
 *
 * @return updated vector f_b
 */
hpx::shared_future<std::vector<double>>
ger(hpx::cuda::experimental::cublas_executor& cublas,
    hpx::shared_future<std::vector<double>> f_A,
    hpx::shared_future<std::vector<double>> f_x,
    hpx::shared_future<std::vector<double>> f_y,
    const std::size_t N);

/**
 * @brief Vector update with diagonal SYRK: r = r + diag(A^T * A)
 *
 * @param f_A update matrix
 * @param f_r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 *
 * @return updated vector f_r
 */
hpx::shared_future<std::vector<double>>
dot_diag_syrk(hpx::cuda::experimental::cublas_executor& cublas,
              hpx::shared_future<std::vector<double>> f_A,
              hpx::shared_future<std::vector<double>> f_r,
              const std::size_t N,
              const std::size_t M);

/**
 * @brief Vector update with diagonal GEMM: r = r + diag(A * B)
 *
 * @param f_A first update matrix
 * @param f_B second update matrix
 * @param f_r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector f_r
 */
hpx::shared_future<std::vector<double>>
dot_diag_gemm(hpx::cuda::experimental::cublas_executor& cublas,
              hpx::shared_future<std::vector<double>> f_A,
              hpx::shared_future<std::vector<double>> f_B,
              hpx::shared_future<std::vector<double>> f_r,
              const std::size_t N,
              const std::size_t M);

// }}} ------------------------------------------ end of BLAS level 2 operations

// BLAS level 1 operations ------------------------------------------------- {{{

/**
 * @brief Dot product: a * b
 * @param f_a left vector
 * @param f_b right vector
 * @param N vector length
 * @return f_a * f_b
 */
double dot(hpx::cuda::experimental::cublas_executor& cublas,
           std::vector<double> a,
           std::vector<double> b,
           const std::size_t N);

// }}} ------------------------------------------ end of BLAS level 1 operations

// Helper functions -------------------------------------------------------- {{{

/**
 * @brief Inverts the value of a BLAS_TRANSPOSE enum
 *
 * This function has been added because cuBLAS computes in column-major order.
 * Therefore the implementations of adapter_cublas.cpp call the cuBLAS routines
 * with inversely transposed matrices and utilize this function to invert the
 * parameter given by the callee.
 *
 * @param trans The BLAS_TRANSPOSE value to invert
 *
 * @return The inverted BLAS_TRANSPOSE value
 */
inline BLAS_TRANSPOSE invert_transpose(BLAS_TRANSPOSE trans)
{
    return (trans == Blas_no_trans) ? Blas_trans : Blas_no_trans;
}

// }}} ------------------------------------------------- end of Helper functions

#endif // end of ADAPTER_CUBLAS_H
