#ifndef TILED_ALGORITHMS_CPU_H_INCLUDED
#define TILED_ALGORITHMS_CPU_H_INCLUDED

#include <hpx/local/future.hpp>
#include "ublas_adapter.hpp"
////////////////////////////////////////////////////////////////////////////////
// Tiled Cholesky Algorithms
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
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
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
        ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[m * n_tiles + n], ft_tiles[k * n_tiles + n], ft_tiles[m * n_tiles + k], N);
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
        ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm<T>), "cholesky_tiled"), ft_tiles[k * n_tiles + m], ft_tiles[n * n_tiles + m], ft_tiles[k * n_tiles + n], N);
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
////////////////////////////////////////////////////////////////////////////////
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
#endif
