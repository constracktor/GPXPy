#ifndef MKL_ADAPTER_H
#define MKL_ADAPTER_H

#include "mkl_cblas.h"
#include "mkl_lapacke.h"

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place Cholesky decomposition of A -> return factorized matrix L
std::vector<double> mkl_potrf(std::vector<double> A,
                              std::size_t N)
{
  // use ?potrf2 recursive version for better stability
  // POTRF - caution with dpotrf
  LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
  // return vector
  return A;
}

// in-place solve X * L^T = A where L lower triangular
std::vector<double> mkl_trsm(std::vector<double> L,
                             std::vector<double> A,
                             std::size_t N)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return A;
}

// A = A - B * B^T
std::vector<double> mkl_syrk(std::vector<double> A,
                             std::vector<double> B,
                             std::size_t N)
{
  // SYRK constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // SYRK kernel - caution with dsyrk
  cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
  // return vector
  return A;
}

// C = C - A * B^T
std::vector<double> mkl_gemm(std::vector<double> A,
                             std::vector<double> B,
                             std::vector<double> C,
                             std::size_t N)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// in-place solve L * x = a where L lower triangular
std::vector<double> mkl_trsv_l(std::vector<double> L,
                               std::vector<double> a,
                               std::size_t N)
{
  // TRSV kernel
  cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return a;
}

// b = b - A * a
std::vector<double> mkl_gemv_l(std::vector<double> A,
                               std::vector<double> a,
                               std::vector<double> b,
                               std::size_t N)
{
  // GEMV constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

// in-place solve L^T * x = a where L lower triangular
std::vector<double> mkl_trsv_u(std::vector<double> L,
                               std::vector<double> a,
                               std::size_t N)
{

  // TRSV kernel
  cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return a;
}

// b = b - A^T * a
std::vector<double> mkl_gemv_u(std::vector<double> A,
                               std::vector<double> a,
                               std::vector<double> b,
                               std::size_t N)
{
  // GEMV constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

// A = y*beta^T + A
std::vector<double> mkl_ger(std::vector<double> A,
                            std::vector<double> x,
                            std::vector<double> y,
                            std::size_t N)
{
  // GER constants
  const double alpha = -1.0;
  // GER kernel
  cblas_dger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(), N);
  // return A
  return A;
}

// C = C + A * B^T
std::vector<double> mkl_gemm_diag(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N)
{
  // GEMM constants
  const double alpha = 1.0;
  const double beta = 1.0;
  // GEMM kernel
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<double> mkl_gemv_p(std::vector<double> A,
                               std::vector<double> a,
                               std::vector<double> b,
                               std::size_t N_row,
                               std::size_t N_col)
{
  // GEMV constants
  const double alpha = 1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N_row, N_col, alpha,
              A.data(), N_col, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations used in uncertainty computation
// in-place solve X * L = A where L lower triangular
std::vector<double> mkl_trsm_l_KcK(std::vector<double> L,
                                   std::vector<double> A,
                                   std::size_t N,
                                   std::size_t M)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return A;
}

// C = C - A * B
std::vector<double> mkl_gemm_l_KcK(std::vector<double> A,
                                   std::vector<double> B,
                                   std::vector<double> C,
                                   std::size_t N,
                                   std::size_t M)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// C = C - A^T * B
std::vector<double> mkl_gemm_cross_tcross_matrix(std::vector<double> A,
                                                 std::vector<double> B,
                                                 std::vector<double> C,
                                                 std::size_t N,
                                                 std::size_t M)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              M, M, N, alpha, A.data(), M, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations used in optimization step
// in-place solve L * X = A where L lower triangular
std::vector<double> mkl_trsm_l_matrix(std::vector<double> L,
                                      std::vector<double> A,
                                      std::size_t N,
                                      std::size_t M)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return A;
}

// C = C - A * B
std::vector<double> mkl_gemm_l_matrix(std::vector<double> A,
                                      std::vector<double> B,
                                      std::vector<double> C,
                                      std::size_t N,
                                      std::size_t M)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// in-place solve L^T * X = A where L upper triangular
std::vector<double> mkl_trsm_u_matrix(std::vector<double> L,
                                      std::vector<double> A,
                                      std::size_t N,
                                      std::size_t M)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return A;
}

// C = C - A^T * B
std::vector<double> mkl_gemm_u_matrix(std::vector<double> A,
                                      std::vector<double> B,
                                      std::vector<double> C,
                                      std::size_t N,
                                      std::size_t M)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// Dot product used in dot calculation
double dot(std::size_t N,
           std::vector<double> A,
           std::vector<double> B)
{
  return cblas_ddot(N, A.data(), 1, B.data(), 1);
}

// C = C - A * B
std::vector<double> mkl_dot_uncertainty(std::vector<double> A,
                                        std::vector<double> R,
                                        std::size_t N,
                                        std::size_t M)
{
  for (int j = 0; j < M; ++j)
  {
    // Extract the j-th column and compute its dot product with itself
    R[j] += cblas_ddot(N, &A[j], M, &A[j], M);
  }

  return R;
}

// C = C - A * B
std::vector<double> mkl_gemm_grad(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> R,
                                  std::size_t N,
                                  std::size_t M)
{
  for (std::size_t i = 0; i < N; ++i)
  {
    R[i] += cblas_ddot(M, &A[i * M], 1, &B[i], N);
  }
  return R;
}

#endif