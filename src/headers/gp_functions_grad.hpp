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

  return -1.0 / (2.0 * hyperparameters[0]) * distance;
}

// compute the squared exponential kernel of two feature vectors and
// additionally return the derivatives w.r.t. lengthscale and vertical_lengthscale
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
      tile[i * N + j] = exp(cov_dist);
    }
  }
  return std::move(tile);
}

// generate a tile of the covariance matrix
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
      tile[i * N + j] = -(hyperparameters[1] / hyperparameters[0]) * cov_dist * exp(cov_dist);
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
        tile[i * N + j] = -1.0;
      }
    }
  }
  return std::move(tile);
}

// compute negative-log likelihood
template <typename T>
T compute_loss(const std::vector<std::vector<T>> &ft_tiles,
               const std::vector<std::vector<T>> &ft_rhs,
               std::size_t N,
               std::size_t n_tiles)
{
  T loss = 0.0;
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto L = ft_tiles[k * n_tiles + k];
    auto beta = ft_rhs[k];
    for (std::size_t l = 0; l < N; l++)
    {
      loss += log(pow(L[l * N + l], 2)) + pow(beta[l], 2);
    }
  }

  loss += (N * n_tiles) * log(2 * M_PI);
  return -0.5 * loss;
}
#endif
