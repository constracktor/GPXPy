// #include "mkl.h
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place Cholesky decomposition of A -> return factorized matrix L
template <typename T>
std::vector<T> mkl_potrf(std::vector<T> A,
                         std::size_t N)
{
  // use ?potrf2 recursive version for better stability
  // POTRF - caution with dpotrf
  LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
  // return vector
  return A;
}

// in-place solve L * X = A^T where L triangular
template <typename T>
std::vector<T> mkl_trsm(std::vector<T> L,
                        std::vector<T> A,
                        std::size_t N)
{
  // TRSM constants
  const T alpha = 1.0f;
  // TRSM kernel - caution with dtrsm
  cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return A;
}

// A = A - B * B^T
template <typename T>
std::vector<T> mkl_syrk(std::vector<T> A,
                        std::vector<T> B,
                        std::size_t N)
{
  // SYRK constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  // SYRK kernel - caution with dsyrk
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
  // return vector
  return A;
}

// C = C - A * B^T
template <typename T>
std::vector<T> mkl_gemm(std::vector<T> A,
                        std::vector<T> B,
                        std::vector<T> C,
                        std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  // GEMM kernel - caution with dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// A = y*beta^T + A
template <typename T>
std::vector<T> mkl_sger(std::vector<T> A,
                        std::vector<T> x,
                        std::vector<T> y,
                        std::size_t N)
{
  // GER constants
  const T alpha = -1.0f;
  // GER kernel
  cblas_sger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(), N);
  // return A
  return A;
}

// C = C - A * B^T
template <typename T>
std::vector<T> mkl_gemm_diag(std::vector<T> A,
                             std::vector<T> B,
                             std::vector<T> C,
                             std::size_t N)
{
  // GEMM constants
  const T alpha = 1.0f;
  const T beta = 1.0f;
  // GEMM kernel
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}