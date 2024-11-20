#include <mkl_cblas.h>
#include <mkl_lapacke.h>

#include <cusolverDn.h>
#include <hpx/async_cuda/cublas_executor.hpp>
#include <hpx/future.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <vector>

// NOTE: include mkl blas and lapacke as long as the cublas implementation does
//       not support all operations
// NOTE: currently only supports double precision

// frequently used names
using hpx::cuda::experimental::check_cuda_error;
using cublas_future = hpx::cuda::experimental::cuda_executor::future_type;

// =============================================================================
// BLAS operations on GPU with cuBLAS (and cuSOLVER)
// =============================================================================

// BLAS operations for tiled cholkesy -------------------------------------- {{{

/**
 * @brief In-place Cholesky decomposition of A to calculate factorized lower
 *        triangular matrix L: L*L^T = A
 */
hpx::shared_future<std::vector<double>>
potrf(hpx::shared_future<cudaStream_t> stream_f,
      hpx::shared_future<std::vector<double>> A_f,
      hpx::shared_future<std::size_t> N_f)
{
    // <t>potrf2's recursive version offers better stability
    // caution with dpotrf
    // LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);

    cusolverDnHandle_t cusolver;
    cusolverDnCreate(&cusolver);
    // NOTE: assumes that stream has already been created with
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
    cudaStream_t stream = stream_f.get();
    cusolverDnSetStream(cusolver, stream);

    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int* d_info = nullptr;
    size_t workspaceInBytesOnDevice;
    void* d_work = nullptr;
    size_t workspaceInBytesOnHost;
    void* h_work = nullptr;

    std::vector<double> h_A = A_f.get();
    double* d_A;
    check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&d_A),
                                h_A.size() * sizeof(double)));
    check_cuda_error(cudaMemcpyAsync(d_A, h_A.data(),
                                     h_A.size() * sizeof(double),
                                     cudaMemcpyHostToDevice, stream));

    std::size_t N = N_f.get();
    cusolverDnXpotrf_bufferSize(cusolver, params, uplo, N, CUDA_R_64F, d_A, N,
                                CUDA_R_64F, &workspaceInBytesOnDevice,
                                &workspaceInBytesOnHost);

    check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&d_work),
                                workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    cusolverDnXpotrf(cusolver, params, uplo, N, CUDA_R_64F, d_A, N, CUDA_R_64F,
                     d_work, workspaceInBytesOnDevice, h_work,
                     workspaceInBytesOnHost, d_info);

    check_cuda_error(cudaMemcpyAsync(h_A.data(), d_A,
                                     sizeof(double) * h_A.size(),
                                     cudaMemcpyDeviceToHost, stream));
    check_cuda_error(cudaMemcpyAsync(&d_info, d_info, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));

    check_cuda_error(cudaStreamSynchronize(stream));

    check_cuda_error(cudaFree(d_A));
    check_cuda_error(cudaFree(d_work));
    if (h_work != nullptr) {
        free(h_work);
    }
    check_cuda_error(cudaFree(d_info));

    // NOTE: cuda stream still exists, an must be destroyed sometime by callee
    cusolverDnDestroy(cusolver);

    return hpx::make_ready_future(h_A);
}

/**
 * @brief Solve the triangular linear system with multiple right-hand-sides
 *        (TRSM) inplace for lower triangular L: X * L^T = A
 *
 * @param cublas cuBLAS executor
 * @param L lower triangular matrix
 * @param A right-hand-side matrix
 * @param N size of the square matrices
 *
 * @return future with the result of the TRSM operation (X, returned within A)
 */
hpx::shared_future<std::vector<double>>
trsm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> L,
     hpx::shared_future<std::vector<double>> A,
     std::size_t N)
{
    // TRSM constants
    const double alpha = 1.0;
    std::size_t matrixSize = N * N;

    // Allocate device memory
    double *d_L, *d_A;
    check_cuda_error(cudaMalloc((void**)&d_L, matrixSize * sizeof(double)));
    check_cuda_error(cudaMalloc((void**)&d_A, matrixSize * sizeof(double)));

    // Copy data from host to device
    std::vector<double> h_L = L.get();
    hpx::post(cublas, cudaMemcpyAsync, d_A, h_L.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> h_A = A.get();
    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    // Compute TRSM on device (X, returned as A)
    // formula here:      X * L^T = alpha * A
    // formula in cublas: X * A^T = alpha * B
    hpx::post(cublas, cublasDtrsm, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, N, N, &alpha, d_L, N, d_A, N);

    // Copy result back to host
    cublas_future copy_device_to_host =
        hpx::async(cublas, cudaMemcpyAsync, h_A.data(), d_A,
                   matrixSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Synchronize, then free device memory
    copy_device_to_host.get();
    check_cuda_error(cudaFree(d_L));
    check_cuda_error(cudaFree(d_A));

    // Return vector
    return hpx::make_ready_future(h_A);
}

/**
 * @brief Calculate symmetric rank-k update (SYRK): A = A - B * B^T
 */
hpx::shared_future<std::vector<double>>
syrk(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> A,
     hpx::shared_future<std::vector<double>> B,
     std::size_t N)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    std::size_t matrixSize = N * N;

    // Allocate device memory
    double *d_A, *d_B;
    check_cuda_error(cudaMalloc((void**)&d_A, matrixSize * sizeof(double)));
    check_cuda_error(cudaMalloc((void**)&d_B, matrixSize * sizeof(double)));

    // Copy data from host to device
    std::vector<double> h_A = A.get();
    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> h_B = B.get();
    hpx::post(cublas, cudaMemcpyAsync, d_B, h_B.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    // Compute SYRK on device
    // formula here:      A = beta * A + alpha * B * B^T
    // formula in cublas: C = beta * C + αlpha * A * A^T
    hpx::post(cublas, cublasDsyrk, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, N,
              &alpha, d_B, N, &beta, d_A, N);

    // Copy result back to host
    cublas_future copy_device_to_host =
        hpx::async(cublas, cudaMemcpyAsync, h_A.data(), d_A,
                   matrixSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Synchronize, then free device memory
    copy_device_to_host.get();
    check_cuda_error(cudaFree(d_A));
    check_cuda_error(cudaFree(d_B));

    // Return future
    return hpx::make_ready_future(h_A);
}

/**
 * Calculate general matrix-matrix multiplication (GEMM): C = C - A * B^T
 */
hpx::shared_future<std::vector<double>>
gemm(hpx::cuda::experimental::cublas_executor& cublas,
     hpx::shared_future<std::vector<double>> A,
     hpx::shared_future<std::vector<double>> B,
     hpx::shared_future<std::vector<double>> C,
     std::size_t N)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    std::size_t matrixSize = N * N;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc((void**)&d_A, matrixSize * sizeof(double)));
    check_cuda_error(cudaMalloc((void**)&d_B, matrixSize * sizeof(double)));
    check_cuda_error(cudaMalloc((void**)&d_C, matrixSize * sizeof(double)));

    // Copy data from host to device
    std::vector<double> h_A = A.get();
    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> h_B = B.get();
    hpx::post(cublas, cudaMemcpyAsync, d_B, h_B.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> h_C = C.get();
    hpx::post(cublas, cudaMemcpyAsync, d_C, h_C.data(),
              matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    // Compute GEMM on device
    hpx::post(cublas, cublasDgemm, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha,
              d_A, N, d_B, N, &beta, d_C, N);

    // copy the result back to the host
    cublas_future copy_device_to_host =
        hpx::async(cublas, cudaMemcpyAsync, h_C.data(), d_C,
                   matrixSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Synchronize, then free device memory
    copy_device_to_host.get();
    check_cuda_error(cudaFree(d_A));
    check_cuda_error(cudaFree(d_B));
    check_cuda_error(cudaFree(d_C));

    // Return future
    return hpx::make_ready_future(h_C);
}

/**
 * in-place solve L * x = a where L lower triangular
 */
std::vector<double>
trsv_l(std::vector<double> L, std::vector<double> a, std::size_t N)
{
    // TRSV kernel
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, N,
                L.data(), N, a.data(), 1);
    // return vector
    return a;
}

/**
 * b = b - A * a
 */
std::vector<double> gemv_l(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N)
{
    // GEMV constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMV kernel
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, alpha, A.data(), N, a.data(),
                1, beta, b.data(), 1);
    // return vector
    return b;
}

/**
 * in-place solve L^T * x = a where L lower triangular
 */
std::vector<double>
trsv_u(std::vector<double> L, std::vector<double> a, std::size_t N)
{
    // TRSV kernel
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, N,
                L.data(), N, a.data(), 1);
    // return vector
    return a;
}

/**
 * b = b - A^T * a
 */
std::vector<double> gemv_u(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N)
{
    // GEMV constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMV kernel
    cblas_dgemv(CblasRowMajor, CblasTrans, N, N, alpha, A.data(), N, a.data(),
                1, beta, b.data(), 1);
    // return vector
    return b;
}

/**
 * A = y*beta^T + A
 */
std::vector<double> ger(std::vector<double> A,
                        std::vector<double> x,
                        std::vector<double> y,
                        std::size_t N)
{
    // GER constants
    const double alpha = -1.0;
    // GER kernel
    cblas_dger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(),
               N);
    // return A
    return A;
}

/**
 * C = C + A * B^T
 */
std::vector<double> gemm_diag(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> C,
                              std::size_t N)
{
    // GEMM constants
    const double alpha = 1.0;
    const double beta = 1.0;
    // GEMM kernel
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha,
                A.data(), N, B.data(), N, beta, C.data(), N);
    // return vector
    return C;
}

// }}} ------------------------------- end of BLAS operations for tiled cholkesy

// BLAS operations for tiled prediction ------------------------------------ {{{

/**
 * b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
 */
std::vector<double> gemv_p(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           std::size_t N_row,
                           std::size_t N_col)
{
    // GEMV constants
    const double alpha = 1.0;
    const double beta = 1.0;
    // GEMV kernel
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_row, N_col, alpha, A.data(),
                N_col, a.data(), 1, beta, b.data(), 1);
    // return vector
    return b;
}

// }}} ----------------------------- end of BLAS operations for tiled prediction

// BLAS operations for uncertainty computation ----------------------------- {{{

/**
 * in-place solve X * L = A where L lower triangular
 */
std::vector<double> trsm_l_KcK(std::vector<double> L,
                               std::vector<double> A,
                               std::size_t N,
                               std::size_t M)
{
    // TRSM constants
    const double alpha = 1.0;
    // TRSM kernel - caution with dtrsm
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
    // return vector
    return A;
}

/**
 * C = C - A * B
 */
std::vector<double> gemm_l_KcK(std::vector<double> A,
                               std::vector<double> B,
                               std::vector<double> C,
                               std::size_t N,
                               std::size_t M)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM kernel - caution with dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, N, alpha,
                A.data(), N, B.data(), M, beta, C.data(), M);
    // return vector
    return C;
}

/**
 * C = C - A^T * B
 */
std::vector<double> gemm_cross_tcross_matrix(std::vector<double> A,
                                             std::vector<double> B,
                                             std::vector<double> C,
                                             std::size_t N,
                                             std::size_t M)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM kernel - caution with dgemm
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, M, N, alpha,
                A.data(), M, B.data(), M, beta, C.data(), M);
    // return vector
    return C;
}

// }}} ---------------------- end of BLAS operations for uncertainty computation

// BLAS operations for optimization step ----------------------------------- {{{

/**
 * in-place solve L * X = A where L lower triangular
 */
std::vector<double> trsm_l_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M)
{
    // TRSM constants
    const double alpha = 1.0;
    // TRSM kernel - caution with dtrsm
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
    // return vector
    return A;
}

/**
 * C = C - A * B
 */
std::vector<double> gemm_l_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N,
                                  std::size_t M)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM kernel - caution with dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, N, alpha,
                A.data(), N, B.data(), M, beta, C.data(), M);
    // return vector
    return C;
}

/**
 * in-place solve L^T * X = A where L upper triangular
 */
std::vector<double> trsm_u_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M)
{
    // TRSM constants
    const double alpha = 1.0;
    // TRSM kernel - caution with dtrsm
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                N, M, alpha, L.data(), N, A.data(), M);
    // return vector
    return A;
}

/**
 * C = C - A^T * B
 */
std::vector<double> gemm_u_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N,
                                  std::size_t M)
{
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM kernel - caution with dgemm
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, M, N, alpha,
                A.data(), N, B.data(), M, beta, C.data(), M);
    // return vector
    return C;
}

/**
 * Dot product used in dot calculation
 */
double dot(std::size_t N, std::vector<double> A, std::vector<double> B)
{
    return cblas_ddot(N, A.data(), 1, B.data(), 1);
}

/**
 * C = C - A * B
 */
std::vector<double> dot_uncertainty(std::vector<double> A,
                                    std::vector<double> R,
                                    std::size_t N,
                                    std::size_t M)
{
    for (int j = 0; j < M; ++j) {
        // Extract the j-th column and compute its dot product with itself
        R[j] += cblas_ddot(N, &A[j], M, &A[j], M);
    }

    return R;
}

/**
 * C = C - A * B
 */
std::vector<double> gemm_grad(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> R,
                              std::size_t N,
                              std::size_t M)
{
    for (std::size_t i = 0; i < N; ++i) {
        R[i] += cblas_ddot(M, &A[i * M], 1, &B[i], N);
    }
    return R;
}

// }}} ---------------------------- end of BLAS operations for optimization step
