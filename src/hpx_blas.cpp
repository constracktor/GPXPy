// Disable ublas debug mode
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

#include <cmath>
#include <random>
#include <iostream>
#include <chrono>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

namespace ublas = boost::numeric::ublas;
#define CALC_TYPE double

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

int main(int argc, char* argv[])
{ // loop size for averaging
  std::size_t n_loop = 500;
  // define exponents of 10
  std::size_t exp_start = 1;
  std::size_t exp_end = 3;
  // define early stop;
  std::size_t early_stop = 15 * pow(10, exp_end - 1);
  // runtime data holder
  std::size_t total_potrf;
  std::size_t total_trsm;
  std::size_t total_gemm;
  // create logscale n vector
  std::vector<std::size_t> n_vector;
  n_vector.resize(9 * (exp_end - exp_start) + 1);
  for (size_t i = exp_start; i < exp_end; i++)
  {
    for (size_t j = 1; j < 10; j++)
    {
      n_vector[(i - exp_start) * 9 + j - 1] = j * pow(10, i);
    }
  }
  n_vector[9 * (exp_end - exp_start)] = pow(10, exp_end);
  // genereate header
  std::cout << "N;POTRF;TRSM;GEMM;loop;" << n_loop << "\n";
  // warm up
  std::size_t warmup = 0;
  for (size_t k = 0; k < 100000; k++)
  {
    warmup = warmup + 1;
  }
  // loop
  for (size_t k = 0; k < n_vector.size(); k++)
  {
    std::size_t n_dim = n_vector[k];
    std::size_t m_size = n_dim * n_dim;
    // early stopping
    if (early_stop < n_dim)
    {
      break;
    }
    // reset data holders
    total_potrf = 0;
    total_trsm = 0;
    total_gemm = 0;
    for (size_t loop = 0; loop < n_loop; loop++)
    {
      //////////////////////////////////////////////////////////////////////////
      // create random matrices
      // setup number generator
      size_t seed = (k + 1) * loop;
      std::mt19937 generator ( seed );
      std::uniform_real_distribution< CALC_TYPE > distribute( 0, 1 );
      // create two positive definite matrices
      // first create two random matrices
      std::vector<CALC_TYPE> M1;
      M1.resize(m_size);
      for (size_t i = 0; i < m_size; i++)
      {
        M1[i] = distribute( generator );
      }
      // then create symmetric matrices
      for (size_t i = 0; i < n_dim; i++)
      {
        for (size_t j = 0; j <= i; j++)
        {
          M1[i * n_dim + j] = 0.5 * (M1[i * n_dim + j] + M1[j * n_dim + i]) / n_dim;
          M1[j * n_dim + i] = M1[i * n_dim + j];
        }
      }
      // add 1 on diagonal
      for (size_t i = 0; i < n_dim; i++)
      {
        M1[i * n_dim + i] = M1[i * n_dim + i] + 1.0;
      }
      ////////////////////////////////////////////////////////////////////////////
      // benchmark
      // time cholesky decomposition
      auto start_potrf = std::chrono::steady_clock::now();
      std::vector<CALC_TYPE> L1 = potrf(M1, n_dim);
      auto end_potrf = std::chrono::steady_clock::now();
      // time triangular solve
      auto start_trsm = std::chrono::steady_clock::now();
      std::vector<CALC_TYPE> M_solved_1 = trsm(M1, L1, n_dim);
      auto end_trsm = std::chrono::steady_clock::now();
      // time matrix multiplication
      auto start_gemm = std::chrono::steady_clock::now();
      std::vector<CALC_TYPE> M_gemm = syrk(M1, M_solved_1, n_dim);
      auto end_gemm = std::chrono::steady_clock::now();
      ////////////////////////////////////////////////////////////////////////////
      // add time difference to total time
      total_potrf += std::chrono::duration_cast<std::chrono::microseconds>(end_potrf - start_potrf).count();
      total_trsm += std::chrono::duration_cast<std::chrono::microseconds>(end_trsm - start_trsm).count();
      total_gemm += std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm).count();
    }
    std::cout <<  n_dim << ";"
              <<  total_potrf / 1000000.0 / n_loop << ";"
              <<  total_trsm / 1000000.0 / n_loop << ";"
              <<  total_gemm / 1000000.0 / n_loop << ";\n";
  }
}
