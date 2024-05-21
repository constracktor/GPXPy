#ifndef GP_FUNCTIONS_GRAD_H_INCLUDED
#define GP_FUNCTIONS_GRAD_H_INCLUDED

#include <cmath>
#include <vector>
#include <numeric>
#include <tuple>
#include <iostream>

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
      tile[i * N + j] = 2.0 * sqrt(hyperparameters[1]) * exp(cov_dist);
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
      tile[i * N + j] = -2.0 * (hyperparameters[1] / sqrt(hyperparameters[0])) * cov_dist * exp(cov_dist);
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

// return zero - used to initialize moment vectors
template <typename T>
T gen_zero()
{
  T z = 0.0;
  return z;
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
  l += N * n * log(2.0 * M_PI);
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
  trace = 1.0 / (2.0 * N * n_tiles) * trace;
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
      trace += (tile[i * N + i] * 2.0 * sqrt(hyperparameters[2]));
    }
  }
  trace = 1.0 / (2.0 * N * n_tiles) * trace;
  return std::move(trace);
}

// update first moment
template <typename T>
T update_fist_moment(const T &gradient,
                     T m_T,
                     const std::vector<T> &beta1_T,
                     int iter)
{
  return beta1_T[iter] * m_T + (1.0 - beta1_T[iter]) * gradient;
}

// update first moment
template <typename T>
T update_second_moment(const T &gradient,
                       T v_T,
                       const std::vector<T> &beta2_T,
                       int iter)
{
  return beta2_T[iter] * v_T + (1.0 - beta2_T[iter]) * gradient * gradient;
}

// update hyperparameter using gradient decent
template <typename T>
T update_param(const T &unconstrained_hyperparam,
               T *hyperparameters,
               const T &gradient,
               T m_T,
               T v_T,
               const std::vector<T> &beta1_T,
               const std::vector<T> &beta2_T,
               int iter)
{
  T alpha_T = hyperparameters[3] * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
  // T mhat = m_T / (1.0 - beta1_T[iter]);
  // T vhat = v_T / (1.0 - beta2_T[iter]);
  // return unconstrained_hyperparam - hyperparameters[3] * mhat / (sqrt(vhat) + hyperparameters[6]));
  return unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + hyperparameters[6]);
}

// transform hyperparameter to enforce constraints using softplus
template <typename T>
T to_constrained(const T &parameter,
                 bool noise)

{
  if (noise)
  {
    return log(1.0 + exp(parameter)) + 1e-6;
  }
  else
  {
    return log(1.0 + exp(parameter));
  }
}

// transform hyperparmeter to entire real line using inverse
// of sofplus. Optimizer, such as gradient decent or Adam,
//  work better with unconstrained parameters
template <typename T>
T to_unconstrained(const T &parameter,
                   bool noise)
{
  if (noise)
  {
    return log(exp(parameter - 1e-6) - 1.0);
  }
  else
  {
    return log(exp(parameter) - 1.0);
  }
}

// compute hyper-parameter beta_1 or beta_2 to power t
template <typename T>
T gen_beta_t(int t,
             T *hyperparameters,
             int param_idx)
{
  return pow(hyperparameters[param_idx], t);
}
#endif
