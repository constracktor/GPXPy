// Disable ublas debug mode
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

#define CALC_TYPE double
#define TYPE "%lf"
#include <cmath>
#include <cassert>
#include <iostream>

#include <hpx/local/chrono.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/iostream.hpp>

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
// GP functions to assemble K
std::vector<CALC_TYPE> compute_regressor_vector(std::size_t row,
                                                std::size_t n_regressors,
                                                std::vector<CALC_TYPE> input)
{
  std::vector<CALC_TYPE> regressor_vector;
  regressor_vector.resize(n_regressors);

  for (std::size_t i = 0; i < n_regressors; i++)
  {
   int index = row - n_regressors + 1 + i;
   if (index < 0)
   {
     regressor_vector[i] = 0.0;
   }
   else
   {
     regressor_vector[i] = input[index];
   }
  }
  return regressor_vector;
}

CALC_TYPE compute_covariance_function(std::size_t n_regressors,
                                      CALC_TYPE* hyperparameters,
                                      std::vector<CALC_TYPE> z_i,
                                      std::vector<CALC_TYPE> z_j)
{
  // Compute the Squared Exponential Covariance Function
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  CALC_TYPE distance = 0.0;
  for (std::size_t i = 0; i < n_regressors; i++)
  {
    distance += pow(z_i[i] - z_j[i],2);
  }
  return hyperparameters[1] * exp(-0.5 * hyperparameters[0] * distance);
}

std::vector<CALC_TYPE> gen_tile_covariance(std::size_t row,
                                           std::size_t col,
                                           std::size_t N,
                                           std::size_t n_regressors,
                                           CALC_TYPE *hyperparameters,
                                           std::vector<CALC_TYPE> input)
{
   std::size_t i_global,j_global;
   CALC_TYPE covariance_function;
   std::vector<std::vector<CALC_TYPE>> z_row, z_col;
   z_row.resize(N);
   z_col.resize(N);
   // compute row regressor vectors beforehand
   for(std::size_t i = 0; i < N; i++)
   {
     i_global = N * row + i;
     z_row[i] = compute_regressor_vector(i_global, n_regressors, input);
   }
   // compute column regressor vectors beforehand
   if (row == col)
   {
     // symmetric diagonal tile
     z_col = z_row;
   }
   else
   {
     for(std::size_t j = 0; j < N; j++)
     {
       j_global = N * col + j;
       z_col[j] = compute_regressor_vector(j_global, n_regressors, input);
     }
   }
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N * N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      for(std::size_t j = 0; j < N; j++)
      {
        j_global = N * col + j;
        // compute covariance function
        covariance_function = compute_covariance_function(n_regressors, hyperparameters, z_row[i], z_col[j]);
        if (i_global==j_global)
        {
          covariance_function += hyperparameters[2];
        }
        tile[i * N + j] = covariance_function;
      }
   }
   return tile;
}

std::vector<CALC_TYPE> gen_tile_output(std::size_t row,
                                       std::size_t N,
                                       std::vector<CALC_TYPE> output)
{
   std::size_t i_global;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      tile[i] = output[i_global];
   }
   return tile;
}

std::vector<CALC_TYPE> gen_tile_cross_covariance(std::size_t row,
                                                 std::size_t col,
                                                 std::size_t N_row,
                                                 std::size_t N_col,
                                                 std::size_t n_regressors,
                                                 CALC_TYPE *hyperparameters,
                                                 std::vector<CALC_TYPE> row_input,
                                                 std::vector<CALC_TYPE> col_input)
{
   std::size_t i_global,j_global;
   CALC_TYPE covariance_function;
   std::vector<std::vector<CALC_TYPE>> z_row, z_col;
   z_row.resize(N_row);
   z_col.resize(N_col);
   // compute row regressor vectors beforehand
   for(std::size_t i = 0; i < N_row; i++)
   {
     i_global = N_row * row + i;
     z_row[i] = compute_regressor_vector(i_global, n_regressors, row_input);
   }
   // compute column regressor vectors beforehand
   for(std::size_t j = 0; j < N_col; j++)
   {
     j_global = N_col * col + j;
     z_col[j] = compute_regressor_vector(j_global, n_regressors, col_input);
   }
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N_row * N_col);
   for(std::size_t i = 0; i < N_row; i++)
   {
      for(std::size_t j = 0; j < N_col; j++)
      {
         // compute covariance function
         covariance_function = compute_covariance_function(n_regressors, hyperparameters, z_row[i], z_col[j]);
         tile[i * N_col + j] = covariance_function;
      }
   }
   return tile;
}

std::vector<CALC_TYPE> gen_tile_zeros(std::size_t N)
{
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N);
   std::fill(tile.begin(),tile.end(),0.0);
   return tile;
}

std::vector<CALC_TYPE> assemble(std::vector<std::vector<CALC_TYPE>> tiles,
                                std::size_t n_tiles,
                                std::size_t tile_size)
{
  std::vector<CALC_TYPE> vector;
  vector.resize(n_tiles * tile_size);
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto tile = tiles[k];
    for(std::size_t i = 0; i < tile_size; i++)
    {
      std::size_t i_global = tile_size * k + i;
      vector[i_global] = tile[i];
    }
  }
  return vector;
}
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
      hpx::cout << qL_kk << '\n' << std::flush;
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
////////////////////////////////////////////////////////////////////////////////
// Tiled Cholesky Algorithms
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                                  std::size_t N,
                                  std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&potrf), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}

void left_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                                 std::size_t N,
                                 std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
      for (std::size_t m = k + 1; m < n_tiles; m++)
      {
        // GEMM
        ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + n], ft_tiles[k * n_tiles + n], ft_tiles[m * n_tiles + k], N);
      }
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&potrf), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
  }
}

void top_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                                std::size_t N,
                                std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      for (std::size_t m = 0; m < n; m++)
      {
        // GEMM
        ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm), "cholesky_tiled"), ft_tiles[k * n_tiles + m], ft_tiles[n * n_tiles + m], ft_tiles[k * n_tiles + n], N);
      }
      // TRSM
      ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm), "cholesky_tiled"), ft_tiles[n * n_tiles + n], ft_tiles[k * n_tiles + n], N);
    }
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&potrf), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
  }
}
////////////////////////////////////////////////////////////////////////////////
// Tiled Triangular Solve Algorithms
void forward_solve_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                         std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_rhs,
                         std::size_t N,
                         std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm_l), "triangular_solve_tiled"), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_l), "triangular_solve_tiled"), ft_tiles[m * n_tiles + k], ft_rhs[k], ft_rhs[m], N);
    }
  }
}

void backward_solve_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                          std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_rhs,
                          std::size_t N,
                          std::size_t n_tiles)
{
  for (int k = n_tiles - 1; k >= 0; k--) // int instead of std::size_t for last comparison
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm_u), "triangular_solve_tiled"), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (int m = k - 1; m >= 0; m--) // int instead of std::size_t for last comparison
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_u), "triangular_solve_tiled"), ft_tiles[k * n_tiles + m], ft_rhs[k], ft_rhs[m], N);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
// Tiled Prediction
void prediction_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles,
                      std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_vector,
                      std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_rhs,
                      std::size_t N_row,
                      std::size_t N_col,
                      std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t m = 0; m < n_tiles; m++)
    {
      ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_p), "prediction_tiled"), ft_tiles[k * n_tiles + m], ft_vector[m], ft_rhs[k], N_row, N_col);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
// Main functions
int hpx_main(hpx::program_options::variables_map& vm)
{
  std::string cholesky = vm["cholesky"].as<std::string>();
  // GP parameters
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE    hyperparameters[3];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1;   // noise_variance = small value
  // tiled parameters
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  std::size_t tile_size = n_train / n_tiles;
  std::size_t tile_size_prediction = n_test / n_tiles;
  // HPX structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  //std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> train_output_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
  //hpx::shared_future<std::vector<CALC_TYPE>> ft_alpha;
  hpx::shared_future<std::vector<CALC_TYPE>> ft_prediction;
  hpx::shared_future<CALC_TYPE> ft_error;
  //std::vector<CALC_TYPE> cross_covariance;
  // data holders for assembly
  std::vector<CALC_TYPE>   training_input;
  std::vector<CALC_TYPE>   training_output;
  std::vector<CALC_TYPE>   test_input;
  std::vector<CALC_TYPE>   test_output;
  // data files
  FILE    *training_input_file;
  FILE    *training_output_file;
  FILE    *test_input_file;
  FILE    *test_output_file;
  FILE    *error_file;
  ////////////////////////////////////////////////////////////////////////////
  // Load data
  training_input.resize(n_train);
  training_output.resize(n_train);
  test_input.resize(n_test);
  test_output.resize(n_test);
  training_input_file = fopen("../src/data/training/training_input.txt", "r");
  training_output_file = fopen("../src/data/training/training_output.txt", "r");
  test_input_file = fopen("../src/data/test/test_input_3.txt", "r");
  test_output_file = fopen("../src/data/test/test_output_3.txt", "r");
  if (training_input_file == NULL || training_output_file == NULL || test_input_file == NULL || test_output_file == NULL)
  {
    printf("Files not found!\n");
    return 1;
  }
  // load training data
  std::size_t scanned_elements = 0;
  for (int i = 0; i < n_train; i++)
  {
    scanned_elements += fscanf(training_input_file,TYPE,&training_input[i]);
    scanned_elements += fscanf(training_output_file,TYPE,&training_output[i]);
  }
  if (scanned_elements != 2 * n_train)
  {
    printf("Error in reading training data!\n");
    return 1;
  }
  // load test data
  scanned_elements = 0;
  for (int i = 0; i < n_test; i++)
  {
    scanned_elements += fscanf(test_input_file,TYPE,&test_input[i]);
    scanned_elements += fscanf(test_output_file,TYPE,&test_output[i]);
  }
  if (scanned_elements != 2 * n_test)
  {
    printf("Error in reading test data!\n");
    return 1;
  }
  // close file streams
  fclose(training_input_file);
  fclose(training_output_file);
  fclose(test_input_file);
  fclose(test_output_file);
  //////////////////////////////////////////////////////////////////////////////
  // ASSEMBLE
  // Assemble covariance matrix vector
  K_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j <= i; j++)
     {
        K_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_covariance, "assemble_tiled"), i, j, tile_size, n_regressors, hyperparameters, training_input);
     }
  }
  // Assemble alpha
  alpha_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    alpha_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, tile_size, training_output);
  }
  // Assemble transposed covariance matrix vector
  cross_covariance_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; j++)
     {
        cross_covariance_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_cross_covariance, "assemble_tiled"), i, j, tile_size_prediction, tile_size, n_regressors, hyperparameters, test_input, training_input);
     }
  }
  // Assemble zero prediction
  prediction_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    prediction_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), tile_size_prediction);
  }
  //////////////////////////////////////////////////////////////////////////////
  // CHOLESKY
  // Compute Cholesky decomposition
  if (cholesky.compare("left") == 0)
  {
    left_looking_cholesky_tiled(K_tiles, tile_size, n_tiles);
  }
  else if (cholesky.compare("right") == 0)
  {
    right_looking_cholesky_tiled(K_tiles, tile_size, n_tiles);
  }
  else // set to "top" per default
  {
    top_looking_cholesky_tiled(K_tiles, tile_size, n_tiles);
  }
  //////////////////////////////////////////////////////////////////////////////
  // TRIANGULAR SOLVE
  forward_solve_tiled(K_tiles, alpha_tiles, tile_size, n_tiles);
  backward_solve_tiled(K_tiles, alpha_tiles, tile_size, n_tiles);
  //////////////////////////////////////////////////////////////////////////////
  // PREDICT
  prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, tile_size, tile_size_prediction, n_tiles);
  // assemble and compute error
  ft_prediction = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&assemble), "prediction_tiled"), prediction_tiles, n_tiles, tile_size_prediction);
  ft_error = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&norm_2), "prediction_tiled"), ft_prediction, test_output, n_test);
  //////////////////////////////////////////////////////////////////////////////
  // write error to file
  CALC_TYPE average_error = ft_error.get() / n_test;
  error_file = fopen("error.csv", "w");
  fprintf(error_file, "\"error\",%lf\n", average_error);
  fclose(error_file);
  //////////////////////////////////////////////////////////////////////////////
  return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    // hpx
    hpx::program_options::options_description desc_commandline;
    hpx::local::init_params init_args;
    ////////////////////////////////////////////////////////////////////////////
    // Setup input arguments
    desc_commandline.add_options()
        ("n_train", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of training samples (max 100 000)")
        ("n_test", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of test samples (max 5 000)")
        ("n_regressors", hpx::program_options::value<std::size_t>()->default_value(100),
        "Number of delayed input regressors")
        ("n_tiles", hpx::program_options::value<std::size_t>()->default_value(10),
         "Number of tiles per dimension -> n_tiles * n_tiles total")
        ("cholesky", hpx::program_options::value<std::string>()->default_value("right"),
         "Choose between left- right- or top-looking tiled Cholesky decomposition")
    ;
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
