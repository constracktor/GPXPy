#ifndef TILED_ALGORITHMS_CPU_H_INCLUDED
#define TILED_ALGORITHMS_CPU_H_INCLUDED

#include <hpx/future.hpp>
#include "ublas_adapter.hpp"
#include "mkl_adapter.hpp"
#include "uncertainty.hpp"
#include "gp_functions_grad.hpp"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// Tiled Cholesky Algorithms
template <typename T>
void right_looking_cholesky_tiled_mkl(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                      std::size_t N,
                                      std::size_t n_tiles)

{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_potrf<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_trsm<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_syrk<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_gemm<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + k],
                                                  ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}

template <typename T>
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                  std::size_t N,
                                  std::size_t n_tiles)

{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&(potrf<T>)), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + k],
                                                  ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}

template <typename T>
void left_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                 std::size_t N,
                                 std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
      for (std::size_t m = k + 1; m < n_tiles; m++)
      {
        // GEMM
        ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + n],
                                                  ft_tiles[k * n_tiles + n], ft_tiles[m * n_tiles + k], N);
      }
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&potrf<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
  }
}

template <typename T>
void top_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
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
        ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + m],
                                                  ft_tiles[n * n_tiles + m], ft_tiles[k * n_tiles + n], N);
      }
      // TRSM
      ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm<T>), "cholesky_tiled"), ft_tiles[n * n_tiles + n], ft_tiles[k * n_tiles + n], N);
    }
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&potrf<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
  }
}
////////////////////////////////////////////////////////////////////////////////
// Tiled Triangular Solve Algorithms
template <typename T>
void forward_solve_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                         std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                         std::size_t N,
                         std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm_l<T>), "triangular_solve_tiled"), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_l<T>), "triangular_solve_tiled"), ft_tiles[m * n_tiles + k], ft_rhs[k], ft_rhs[m], N);
    }
  }
}

template <typename T>
void backward_solve_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                          std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                          std::size_t N,
                          std::size_t n_tiles)
{
  for (int k = n_tiles - 1; k >= 0; k--) // int instead of std::size_t for last comparison
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&trsm_u<T>), "triangular_solve_tiled"), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (int m = k - 1; m >= 0; m--) // int instead of std::size_t for last comparison
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_u<T>), "triangular_solve_tiled"), ft_tiles[k * n_tiles + m], ft_rhs[k], ft_rhs[m], N);
    }
  }
}
// // Tiled Triangular Solve Algorithms for Mtrices (K * X = B)
template <typename T>
void forward_solve_tiled_matrix(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                                std::size_t N,
                                std::size_t M,
                                std::size_t n_tiles,
                                std::size_t m_tiles)
{
  for (std::size_t c = 0; c < m_tiles; c++)
  {
    for (std::size_t k = 0; k < n_tiles; k++)
    {
      // TRSM
      ft_rhs[k * m_tiles + c] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_trsm_l_matrix<T>), "triangular_solve_tiled_matrix"), ft_tiles[k * n_tiles + k],
                                              ft_rhs[k * m_tiles + c], N, M);
      for (std::size_t m = k + 1; m < n_tiles; m++)
      {
        // GEMV
        ft_rhs[m * m_tiles + c] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_gemm_l_matrix<T>), "triangular_solve_tiled_matrix"), ft_tiles[m * n_tiles + k],
                                                ft_rhs[k * m_tiles + c], ft_rhs[m * m_tiles + c], N, M);
      }
    }
  }
}

template <typename T>
void backward_solve_tiled_matrix(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                 std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                                 std::size_t N,
                                 std::size_t M,
                                 std::size_t n_tiles,
                                 std::size_t m_tiles)
{
  for (int c = 0; c < m_tiles; c++)
  {
    for (int k = n_tiles - 1; k >= 0; k--) // int instead of std::size_t for last comparison
    {
      // TRSM
      ft_rhs[k * m_tiles + c] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_trsm_u_matrix<T>), "triangular_solve_tiled_matrix"), ft_tiles[k * n_tiles + k],
                                              ft_rhs[k * m_tiles + c], N, M);
      for (int m = k - 1; m >= 0; m--) // int instead of std::size_t for last comparison
      {
        // GEMV
        ft_rhs[m * m_tiles + c] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_gemm_u_matrix<T>), "triangular_solve_tiled_matrix"), ft_tiles[k * n_tiles + m],
                                                ft_rhs[k * m_tiles + c], ft_rhs[m * m_tiles + c], N, M);
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
// Tiled Loss
template <typename T>
void compute_loss_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                        std::vector<hpx::shared_future<std::vector<T>>> &ft_alpha,
                        std::vector<hpx::shared_future<std::vector<T>>> &ft_y,
                        hpx::shared_future<T> &loss,
                        std::size_t N,
                        std::size_t n_tiles)
{
  std::vector<hpx::shared_future<T>> loss_tiled;
  loss_tiled.resize(n_tiles);
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    loss_tiled[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_loss<T>), "loss_tiled"), ft_tiles[k * n_tiles + k], ft_alpha[k], ft_y[k], N);
  }

  loss = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&add_losses<T>), "loss_tiled"), loss_tiled, N, n_tiles);
  // for (std::size_t k = 0; k < n_tiles; k++)
  // {
  // loss += hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_loss<T>), "loss_tiled"), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
  // }
  // loss += (N * n_tiles) * log(2 * M_PI);
  // loss = -0.5 * loss;
  // T l = 1.0;
  // loss = hpx::make_ready_future(l);
  // loss = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_loss<T>), "loss_tiled"), N);
}

// Tiled Prediction
template <typename T>
void prediction_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                      std::vector<hpx::shared_future<std::vector<T>>> &ft_vector,
                      std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                      std::size_t N_row,
                      std::size_t N_col,
                      std::size_t n_tiles,
                      std::size_t m_tiles)
{
  for (std::size_t k = 0; k < m_tiles; k++)
  {
    for (std::size_t m = 0; m < n_tiles; m++)
    {
      ft_rhs[k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemv_p<T>), "prediction_tiled"), ft_tiles[k * n_tiles + m], ft_vector[m], ft_rhs[k], N_row, N_col);
    }
  }
}

// // Tiled Diagonal of Posterior Covariance Matrix
// template <typename T>
// void posterior_covariance_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_CC_tiles,
//                                 std::vector<hpx::shared_future<std::vector<T>>> &ft_tCC_tiles,
//                                 std::vector<hpx::shared_future<std::vector<T>>> &ft_K_tiles,
//                                 std::size_t N,
//                                 std::size_t M,
//                                 std::size_t n_tiles,
//                                 std::size_t m_tiles)
// {
//   for (int i = 0; i < m_tiles; ++i)
//   {
//     for (int j = i; j < i + 1; ++j)
//     { // outer two loops used for diagonal tiles of prior K
//       for (int m = 0; m < n_tiles; ++m)
//       { // Compute inner product to obtain diagonal elements of (K_MxN * (K^-1_NxN * K_NxM))
//         ft_K_tiles[i * m_tiles + j] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_gemm_uncertainty_matrix<T>), "posterior_tiled"), ft_CC_tiles[i * n_tiles + m],
//                                                     ft_tCC_tiles[m * m_tiles + i], ft_K_tiles[i * m_tiles + j], N, M);
//       }
//     }
//   }
// }

// // Tiled Prediction Uncertainty
// template <typename T>
// void prediction_uncertainty_tiled(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
//                                   std::vector<hpx::shared_future<std::vector<T>>> &ft_vector,
//                                   std::size_t M,
//                                   std::size_t m_tiles)
// {
//   for (std::size_t i = 0; i < m_tiles; i++)
//   {
//     ft_vector[i] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&diag<T>), "uncertainty_tiled"), ft_tiles[i * m_tiles + i], M);
//   }
// }

////////////////////////////////////////////////////////////////////////////////
// Fill y*y^T*inv(K)-I Cholesky Algorithms
template <typename T>
void update_grad_K_tiled_mkl(std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                             const std::vector<hpx::shared_future<std::vector<T>>> &ft_v1,
                             const std::vector<hpx::shared_future<std::vector<T>>> &ft_v2,
                             std::size_t N,
                             std::size_t n_tiles)

{
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    for (std::size_t j = 0; j < n_tiles; j++)
    {
      ft_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_sger<T>), "gradient_tiled"), ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j], N);
    }
  }
}

// Perform a gradient scent step for selected hyperparameter
template <typename T>
void update_hyperparameter(const std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                           const std::vector<hpx::shared_future<std::vector<T>>> &ft_rhs,
                           T *hyperparameters,
                           std::size_t N,
                           std::size_t n_tiles,
                           int parameter_idx)
{

  if (parameter_idx == 0 || parameter_idx == 1)
  {
    std::vector<hpx::shared_future<std::vector<T>>> diag_tiles;
    diag_tiles.resize(n_tiles);
    for (std::size_t d = 0; d < n_tiles; d++)
    {
      diag_tiles[d] = hpx::async(hpx::annotated_function(&gen_tile_zeros_diag<T>, "assemble_tiled"), N);
    }

    // Compute diagonal tiles using GEMM
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        diag_tiles[i] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&mkl_gemm_diag<T>), "gradient_tiled"), ft_tiles[i * n_tiles + j],
                                      ft_rhs[j * n_tiles + i], diag_tiles[i], N);
      }
    }
    // compute trace of diag_tiles
    hpx::shared_future<T> trace = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_trace<T>), "gradient_tiled"), diag_tiles, N, n_tiles);
    // update hyperparameter
    hyperparameters[parameter_idx] = hyperparameters[parameter_idx] - hyperparameters[3] * trace.get();
  }
  else
  {
    // Throw an exception for invalid param_idx
    throw std::invalid_argument("Invalid param_idx");
  }
}

// Update noise variance using gradient decent
template <typename T>
void update_noise_variance(const std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                           T *hyperparameters,
                           std::size_t N,
                           std::size_t n_tiles)
{
  // compute trace of diag_tiles
  hpx::shared_future<T> trace = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_trace_noise<T>), "gradient_tiled"), ft_tiles, hyperparameters, N, n_tiles);
  // update hyperparameter
  hyperparameters[2] = hyperparameters[2] - hyperparameters[3] * trace.get();
}
#endif
