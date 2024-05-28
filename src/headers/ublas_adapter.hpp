#ifndef UBLAS_ADAPTER_H_INCLUDED
#define UBLAS_ADAPTER_H_INCLUDED
// Disable ublas debug mode
#ifndef NDEBUG
#define BOOST_UBLAS_NDEBUG
#endif

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>
namespace ublas = boost::numeric::ublas;
////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place Cholesky decomposition of A -> return factorized matrix L
template <typename T>
std::vector<T> potrf(std::vector<T> A,
                     std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<T, ublas::row_major, std::vector<T>> A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix<T, ublas::row_major, std::vector<T>> L_blas(N, N);
  // POTRF (compute Cholesky)
  for (size_t k = 0; k < N; k++)
  {
    // compute squared diagonal entry
    T qL_kk = A_blas(k, k) - ublas::inner_prod(
                                 ublas::project(ublas::row(L_blas, k), ublas::range(0, k)),
                                 ublas::project(ublas::row(L_blas, k), ublas::range(0, k)));
    // check if positive
    if (qL_kk <= 0)
    {
      std::cout << qL_kk << '\n'
                << std::flush;
    }
    else
    {
      // set diagonal entry
      T L_kk = std::sqrt(qL_kk);
      L_blas(k, k) = L_kk;
      // compute corresponding column
      ublas::matrix_column<ublas::matrix<T, ublas::row_major, std::vector<T>>> cLk(L_blas, k);
      ublas::project(cLk, ublas::range(k + 1, N)) = (ublas::project(ublas::column(A_blas, k), ublas::range(k + 1, N)) - ublas::prod(ublas::project(L_blas, ublas::range(k + 1, N), ublas::range(0, k)),
                                                                                                                                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k)))) /
                                                    L_kk;
    }
  }
  // reformat to std::vector
  A = L_blas.data();
  return A;
}

// in-place solve L * X = A^T where L triangular
template <typename T>
std::vector<T> trsm(std::vector<T> L,
                    std::vector<T> A,
                    std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<T, ublas::row_major, std::vector<T>> L_blas(N, N);
  L_blas.data() = L;
  ublas::matrix<T, ublas::column_major, std::vector<T>> A_blas(N, N); // use column_major because A^T
  A_blas.data() = A;
  // TRSM
  ublas::inplace_solve(L_blas, A_blas, ublas::lower_tag());
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

//  A = A - B * B^T
template <typename T>
std::vector<T> syrk(std::vector<T> A,
                    std::vector<T> B,
                    std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<T, ublas::row_major, std::vector<T>> A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix<T, ublas::row_major, std::vector<T>> B_blas(N, N);
  B_blas.data() = B;
  // SYRK
  A_blas = A_blas - ublas::prod(B_blas, ublas::trans(B_blas));
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

// C = C - A * B^T
template <typename T>
std::vector<T> gemm(std::vector<T> A,
                    std::vector<T> B,
                    std::vector<T> C,
                    std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<T, ublas::row_major, std::vector<T>> A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix<T, ublas::row_major, std::vector<T>> B_blas(N, N);
  B_blas.data() = B;
  ublas::matrix<T, ublas::row_major, std::vector<T>> C_blas(N, N);
  C_blas.data() = C;
  // GEMM
  C_blas = C_blas - ublas::prod(A_blas, ublas::trans(B_blas));
  // reformat to std::vector
  C = C_blas.data();
  return C;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled triangular solve
// in-place solve L * x = a where L lower triangular
std::vector<float> trsm_l(std::vector<float> L,
                          std::vector<float> a,
                          std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<float, ublas::row_major, std::vector<float>> L_blas(N, N);
  L_blas.data() = L;
  ublas::vector<float, std::vector<float>> a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(L_blas, a_blas, ublas::lower_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// in-place solve L^T * x = a where L lower triangular
std::vector<float> trsm_u(std::vector<float> L,
                          std::vector<float> a,
                          std::size_t N)
{
  // convert to boost matrices
  ublas::matrix<float, ublas::row_major, std::vector<float>> L_blas(N, N);
  L_blas.data() = L;
  ublas::vector<float, std::vector<float>> a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(ublas::trans(L_blas), a_blas, ublas::upper_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// b = b - A * a
std::vector<float> gemv_l(std::vector<float> A,
                          std::vector<float> a,
                          std::vector<float> b,
                          std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix<float, ublas::row_major, std::vector<float>> A_blas(N, N);
  A_blas.data() = A;
  ublas::vector<float, std::vector<float>> a_blas(N);
  a_blas.data() = a;
  ublas::vector<float, std::vector<float>> b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(A_blas, a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

// b = b - A^T * a
std::vector<float> gemv_u(std::vector<float> A,
                          std::vector<float> a,
                          std::vector<float> b,
                          std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix<float, ublas::row_major, std::vector<float>> A_blas(N, N);
  A_blas.data() = A;
  ublas::vector<float, std::vector<float>> a_blas(N);
  a_blas.data() = a;
  ublas::vector<float, std::vector<float>> b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(ublas::trans(A_blas), a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<float> gemv_p(std::vector<float> A,
                          std::vector<float> a,
                          std::vector<float> b,
                          std::size_t N_row,
                          std::size_t N_col)
{
  // convert to boost matrix and vectors
  ublas::matrix<float, ublas::row_major, std::vector<float>> A_blas(N_row, N_col);
  A_blas.data() = A;
  ublas::vector<float, std::vector<float>> a_blas(N_col);
  a_blas.data() = a;
  ublas::vector<float, std::vector<float>> b_blas(N_row);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas + ublas::prod(A_blas, a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}
#endif
