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

#include "ublas_adapter.hpp"
////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// Cholesky decomposition of A -> return factorized matrix L
std::vector<CALC_TYPE> potrf(std::vector<CALC_TYPE> A,
                             std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  // POTRF (compute Cholesky)
  for (size_t k=0 ; k < N; k++)
  {
    // compute squared diagonal entry
    CALC_TYPE qL_kk = A_blas(k,k) - ublas::inner_prod( ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ), ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ) );
    // check if positive
    if (qL_kk <= 0)
    {
      std::cout << qL_kk << '\n' << std::flush;
    }
    else
    {
      // set diagonal entry
      CALC_TYPE L_kk = std::sqrt( qL_kk );
      L_blas(k,k) = L_kk;
      // compute corresponding column
      ublas::matrix_column<ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> >> cLk(L_blas, k);
      ublas::project( cLk, ublas::range(k+1, N) )
        = ( ublas::project( ublas::column(A_blas, k), ublas::range(k+1, N) )
            - ublas::prod( ublas::project(L_blas, ublas::range(k+1, N), ublas::range(0, k)),
                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  // reformat to std::vector
  A = L_blas.data();
  return A;
}

// solve L * X = A^T where L triangular
std::vector<CALC_TYPE> trsm(std::vector<CALC_TYPE> L,
                            std::vector<CALC_TYPE> A,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::matrix< CALC_TYPE, ublas::column_major, std::vector<CALC_TYPE> > A_blas(N, N);//use column_major because A^T
  A_blas.data() = A;
  // TRSM
  ublas::inplace_solve(L_blas, A_blas, ublas::lower_tag());
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

//  A = A - B * B^T
std::vector<CALC_TYPE> syrk(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  //SYRK
  A_blas = A_blas - ublas::prod(B_blas,ublas::trans(B_blas));
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

//C = C - A * B^T
std::vector<CALC_TYPE> gemm(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::vector<CALC_TYPE> C,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
  C_blas.data() = C;
  // GEMM
  C_blas = C_blas - ublas::prod(A_blas, ublas::trans(B_blas));
  // reformat to std::vector
  C = C_blas.data();
  return C;
}

// BLAS operations for tiled triangular solve
// solve L * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_l(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(L_blas, a_blas, ublas::lower_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// solve L^T * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_u(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(ublas::trans(L_blas), a_blas, ublas::upper_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// b = b - A * a
std::vector<CALC_TYPE> gemv_l(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(A_blas, a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

// b = b - A^T * a
std::vector<CALC_TYPE> gemv_u(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(ublas::trans(A_blas), a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<CALC_TYPE> gemv_p(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N_row,
                            std::size_t N_col)
{
  // convert to boost matrix and vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N_row, N_col);
  A_blas.data() = A;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N_col);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N_row);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas  + ublas::prod(A_blas, a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

// ||a - b||^2
CALC_TYPE norm_2(std::vector<CALC_TYPE> a,
                 std::vector<CALC_TYPE> b,
                 std::size_t N)
{
  // convert to boost vectors
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N);
  b_blas.data() = b;
  // NORM
  CALC_TYPE error = ublas::norm_2(a_blas - b_blas);
  return error;
}
