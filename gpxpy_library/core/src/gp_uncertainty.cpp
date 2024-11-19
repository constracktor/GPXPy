#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H

#include "mkl_adapter.hpp"

#include <cmath>
#include <vector>

// retrieve diagonal elements of posterior covariance matrix
std::vector<double> diag_posterior(const std::vector<double> &A,
                                   const std::vector<double> &B,
                                   std::size_t M)
{
  // Initialize tile
  std::vector<double> tile;
  tile.reserve(M);

  for (std::size_t i = 0; i < M; ++i)
  {
    tile.push_back(A[i] - B[i]);
  }

  return std::move(tile);
}

// retrieve diagonal elements of posterior covariance matrix
std::vector<double> diag_tile(const std::vector<double> &A,
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

// // in-place solve X * L^T = A where L upper triangular
// std::vector<double> mkl_trsm_u_KK(std::vector<double> L,
//                                   std::vector<double> A,
//                                   std::size_t N,
//                                   std::size_t M)
// {
//   // TRSM constants
//   const double alpha = 1.0;
//   // TRSM kernel - caution with dtrsm
//   cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, M, N, alpha, L.data(), N, A.data(), N);
//   // return vector
//   return A;
// }

// // C = C - A^T * B
// std::vector<double> mkl_gemm_u_KK(std::vector<double> A,
//                                   std::vector<double> B,
//                                   std::vector<double> C,
//                                   std::size_t N,
//                                   std::size_t M)
// {
//   // GEMM constants
//   const double alpha = -1.0;
//   const double beta = 1.0;
//   // GEMM kernel - caution with dgemm
//   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//               M, N, N, alpha, B.data(), N, A.data(), N, beta, C.data(), N);
//   // return vector
//   return C;
// }

#endif