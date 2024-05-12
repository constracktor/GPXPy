#ifndef GP_FUNCTIONS_GRAD_H_INCLUDED
#define GP_FUNCTIONS_GRAD_H_INCLUDED

#include <cmath>
#include <vector>
#include <tuple>

// compute the distance devided by the lengthscale
template <typename T>
T compute_covariance_dist_func(std::size_t i_global,
                               std::size_t j_global,
                               std::size_t n_regressors,
                               T *hyperparameters,
                               const std::vector<T> &i_input,
                               const std::vector<T> &j_input)
{
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  T z_ik = 0.0;
  T z_jk = 0.0;
  T distance = 0.0;
  for (std::size_t k = 0; k < n_regressors; k++)
  {
    //
    int offset = -n_regressors + 1 + k;
    int i_local = i_global + offset;
    int j_local = j_global + offset;
    //
    if (i_local >= 0)
    {
      z_ik = i_input[i_local];
    }
    if (j_local >= 0)
    {
      z_jk = j_input[j_local];
    }
    distance += pow(z_ik - z_jk, 2);
  }

  return -1.0 / (2.0 * hyperparameters[0] * hyperparameters[0]) * distance;
}

// compute derivative w.r.t. vertical_lengthscale
template <typename T>
std::vector<T> gen_tile_grad_v(std::size_t row,
                               std::size_t col,
                               std::size_t N,
                               std::size_t n_regressors,
                               T *hyperparameters,
                               const std::vector<T> &input)
{
  std::size_t i_global, j_global;
  T cov_dist;
  // Initialize tile
  std::vector<T> tile;
  tile.resize(N * N);
  for (std::size_t i = 0; i < N; i++)
  {
    i_global = N * row + i;
    for (std::size_t j = 0; j < N; j++)
    {
      j_global = N * col + j;
      // compute covariance function
      cov_dist = compute_covariance_dist_func(i_global, j_global, n_regressors, hyperparameters, input, input);
      tile[i * N + j] = 2 * hyperparameters[1] * exp(cov_dist);
    }
  }
  return std::move(tile);
}

// compute derivative w.r.t. lengthscale
template <typename T>
std::vector<T> gen_tile_grad_l(std::size_t row,
                               std::size_t col,
                               std::size_t N,
                               std::size_t n_regressors,
                               T *hyperparameters,
                               const std::vector<T> &input)
{
  std::size_t i_global, j_global;
  T cov_dist;
  // Initialize tile
  std::vector<T> tile;
  tile.resize(N * N);
  for (std::size_t i = 0; i < N; i++)
  {
    i_global = N * row + i;
    for (std::size_t j = 0; j < N; j++)
    {
      j_global = N * col + j;
      // compute covariance function
      cov_dist = compute_covariance_dist_func(i_global, j_global, n_regressors, hyperparameters, input, input);
      tile[i * N + j] = -2 * (hyperparameters[1] * hyperparameters[1] / hyperparameters[0]) * cov_dist * exp(cov_dist);
    }
  }
  return std::move(tile);
}

// generate an identity tile if i==j
template <typename T>
std::vector<T> gen_tile_identity(std::size_t row,
                                 std::size_t col,
                                 std::size_t N)
{
  std::size_t i_global, j_global;
  // Initialize tile
  std::vector<T> tile;
  tile.resize(N * N);
  std::fill(tile.begin(), tile.end(), 0.0);
  for (std::size_t i = 0; i < N; i++)
  {
    i_global = N * row + i;
    for (std::size_t j = i; j <= i; j++)
    {
      j_global = N * col + j;
      if (i_global == j_global)
      {
        tile[i * N + j] = 1.0;
      }
    }
  }
  return std::move(tile);
}

// generate a empty tile NxN
template <typename T>
std::vector<T> gen_tile_zeros_diag(std::size_t N)
{
  // Initialize tile
  std::vector<T> tile;
  tile.resize(N * N);
  std::fill(tile.begin(), tile.end(), 0.0);
  return std::move(tile);
}

// compute negative-log likelihood tiled
template <typename T>
T compute_loss(const std::vector<T> &K_diag_tile,
               const std::vector<T> &alpha_tile,
               const std::vector<T> &y_tile,
               std::size_t N)
{
  T l = 0.0;
  for (std::size_t i = 0; i < N; i++)
  {
    // Add the squared difference to the error
    l += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]) + y_tile[i] * alpha_tile[i];
  }

  return l;
}

// compute negative-log likelihood
template <typename T>
T add_losses(const std::vector<T> &losses,
             std::size_t N,
             std::size_t n)
{
  T l = 0.0;
  for (std::size_t i = 0; i < n; i++)
  {
    // Add the squared difference to the error
    l += losses[i];
  }
  l += N * n * log(2 * M_PI);
  return 0.5 * l / (N * n);
}

// compute trace
template <typename T>
T compute_trace(const std::vector<std::vector<T>> &diag_tiles,
                std::size_t N,
                std::size_t n_tiles)
{
  // Initialize tile
  T trace = 0.0;
  for (std::size_t d = 0; d < n_tiles; d++)
  {
    auto tile = diag_tiles[d];
    for (std::size_t i = 0; i < N; ++i)
    {
      trace += tile[i * N + i];
    }
  }
  return std::move(trace);
}

// compute trace for noise variance
// Same function as compute_trace with the only difference
// that we need to retrieve the diag tiles
template <typename T>
T compute_trace_noise(const std::vector<std::vector<T>> &ft_tiles,
                      T *hyperparameters,
                      std::size_t N,
                      std::size_t n_tiles)
{
  // Initialize tile
  T trace = 0.0;
  for (std::size_t d = 0; d < n_tiles; d++)
  {
    auto tile = ft_tiles[d * n_tiles + d];
    for (std::size_t i = 0; i < N; ++i)
    {
      trace += (tile[i * N + i] * 2 * hyperparameters[2]);
    }
  }
  return std::move(trace);
}
#endif
