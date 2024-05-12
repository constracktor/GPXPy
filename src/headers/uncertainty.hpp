// #include "mkl.h
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

#include <cmath>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place solve L * X = A where L lower triangular
template <typename T>
std::vector<T> mkl_trsm_l_matrix(std::vector<T> L,
                                 std::vector<T> A,
                                 std::size_t N,
                                 std::size_t M)
{
  // TRSM constants
  const T alpha = 1.0f;
  // TRSM kernel - caution with dtrsm
  cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return A;
}

// C = C - A * B
template <typename T>
std::vector<T> mkl_gemm_l_matrix(std::vector<T> A,
                                 std::vector<T> B,
                                 std::vector<T> C,
                                 std::size_t N,
                                 std::size_t M)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  // GEMM kernel - caution with dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// in-place solve L^T * X = A where L upper triangular
template <typename T>
std::vector<T> mkl_trsm_u_matrix(std::vector<T> L,
                                 std::vector<T> A,
                                 std::size_t N,
                                 std::size_t M)
{
  // TRSM constants
  const T alpha = 1.0f;
  // TRSM kernel - caution with dtrsm
  cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return A;
}

// C = C - A^T * B
template <typename T>
std::vector<T> mkl_gemm_u_matrix(std::vector<T> A,
                                 std::vector<T> B,
                                 std::vector<T> C,
                                 std::size_t N,
                                 std::size_t M)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  // GEMM kernel - caution with dgemm
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// C = C - A * B
template <typename T>
std::vector<T> mkl_gemm_uncertainty_matrix(std::vector<T> A,
                                           std::vector<T> B,
                                           std::vector<T> C,
                                           std::size_t N,
                                           std::size_t M)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  // GEMM kernel - caution with dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// retrieve diagonal elements of posterior covariance matrix
template <typename T>
std::vector<T> diag(std::vector<T> &A,
                    std::size_t M)
{
  // Initialize tile
  std::vector<T> tile;
  tile.reserve(M);

  for (std::size_t i = 0; i < M; ++i)
  {
    tile.push_back(A[i * M + i]);
  }

  return std::move(tile);
}