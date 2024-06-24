#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H


// #include "mkl.h
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

#include <cmath>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
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

// C = C - A * B
std::vector<double> mkl_gemm_uncertainty_matrix(std::vector<double> A,
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
              M, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return C;
}

// retrieve diagonal elements of posterior covariance matrix
std::vector<double> diag(const std::vector<double> &A,
                         std::size_t M)
{
  // Initialize tile
  std::vector<double> tile;
  tile.reserve(M);

  for (std::size_t i = 0; i < M; ++i)
  {
    tile.push_back(A[i * M + i]);
  }

  return std::move(tile);
}

// in-place solve X * L = A where L lower triangular
std::vector<double> mkl_trsm_l_KK(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, M, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return A;
}

// C = C - A * B
std::vector<double> mkl_gemm_l_KK(std::vector<double> A,
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
              M, N, N, alpha, B.data(), N, A.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// in-place solve X * L^T = A where L upper triangular
std::vector<double> mkl_trsm_u_KK(std::vector<double> L,
                                  std::vector<double> A,
                                  std::size_t N,
                                  std::size_t M)
{
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, M, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return A;
}

// C = C - A^T * B
std::vector<double> mkl_gemm_u_KK(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  std::size_t N,
                                  std::size_t M)
{
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              M, N, N, alpha, B.data(), N, A.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

#endif