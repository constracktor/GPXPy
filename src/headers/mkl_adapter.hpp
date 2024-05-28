// #include "mkl.h
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place Cholesky decomposition of A -> return factorized matrix L
std::vector<float> mkl_potrf(std::vector<float> A,
                             std::size_t N)
{
  // use ?potrf2 recursive version for better stability
  // POTRF - caution with dpotrf
  LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
  // return vector
  return A;
}

// in-place solve L * X = A^T where L triangular
std::vector<float> mkl_trsm(std::vector<float> L,
                            std::vector<float> A,
                            std::size_t N)
{
  // TRSM constants
  const float alpha = 1.0f;
  // TRSM kernel - caution with dtrsm
  cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return A;
}

// A = A - B * B^T
std::vector<float> mkl_syrk(std::vector<float> A,
                            std::vector<float> B,
                            std::size_t N)
{
  // SYRK constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // SYRK kernel - caution with dsyrk
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
  // return vector
  return A;
}

// C = C - A * B^T
std::vector<float> mkl_gemm(std::vector<float> A,
                            std::vector<float> B,
                            std::vector<float> C,
                            std::size_t N)
{
  // GEMM constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // GEMM kernel - caution with dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// A = y*beta^T + A
std::vector<float> mkl_ger(std::vector<float> A,
                           std::vector<float> x,
                           std::vector<float> y,
                           std::size_t N)
{
  // GER constants
  const float alpha = -1.0f;
  // GER kernel
  cblas_sger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(), N);
  // return A
  return A;
}

// C = C - A * B^T
std::vector<float> mkl_gemm_diag(std::vector<float> A,
                                 std::vector<float> B,
                                 std::vector<float> C,
                                 std::size_t N)
{
  // GEMM constants
  const float alpha = 1.0f;
  const float beta = 1.0f;
  // GEMM kernel
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return C;
}

// in-place solve L * x = a where L lower triangular
std::vector<float> mkl_trsv_l(std::vector<float> L,
                              std::vector<float> a,
                              std::size_t N)
{
  // TRSV kernel
  cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return a;
}

// b = b - A * a
std::vector<float> mkl_gemv_l(std::vector<float> A,
                              std::vector<float> a,
                              std::vector<float> b,
                              std::size_t N)
{
  // GEMV constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // GEMV kernel
  cblas_sgemv(CblasRowMajor, CblasNoTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

// in-place solve L^T * x = a where L lower triangular
std::vector<float> mkl_trsv_u(std::vector<float> L,
                              std::vector<float> a,
                              std::size_t N)
{

  // TRSV kernel
  cblas_strsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return a;
}

// b = b - A^T * a
std::vector<float> mkl_gemv_u(std::vector<float> A,
                              std::vector<float> a,
                              std::vector<float> b,
                              std::size_t N)
{
  // GEMV constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // GEMV kernel
  cblas_sgemv(CblasRowMajor, CblasTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<float> mkl_gemv_p(std::vector<float> A,
                              std::vector<float> a,
                              std::vector<float> b,
                              std::size_t N_row,
                              std::size_t N_col)
{
  // GEMV constants
  const float alpha = 1.0f;
  const float beta = 1.0f;
  // GEMV kernel
  cblas_sgemv(CblasRowMajor, CblasNoTrans, N_row, N_col, alpha,
              A.data(), N_col, a.data(), 1, beta, b.data(), 1);
  // return vector
  return b;
}

//////////////////////////////////////////////////////////
/////////  functions with double precision  //////////////
//////////////////////////////////////////////////////////

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

// in-place solve L * X = A^T where L triangular
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

// C = C - A * B^T
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