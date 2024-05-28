#ifndef GP_FUNCTIONS_GRAD_H_INCLUDED
#define GP_FUNCTIONS_GRAD_H_INCLUDED

#include <cmath>
#include <vector>
#include <numeric>
#include <tuple>
#include <iostream>

// compute the distance devided by the lengthscale
float compute_covariance_dist_func(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   float *hyperparameters,
                                   const std::vector<float> &i_input,
                                   const std::vector<float> &j_input)
{
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  float z_ik = 0.0;
  float z_jk = 0.0;
  float distance = 0.0;
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
std::vector<float> gen_tile_grad_v(std::size_t row,
                                   std::size_t col,
                                   std::size_t N,
                                   std::size_t n_regressors,
                                   float *hyperparameters,
                                   const std::vector<float> &input)
{
  std::size_t i_global, j_global;
  float cov_dist;
  // Initialize tile
  std::vector<float> tile;
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
std::vector<float> gen_tile_grad_l(std::size_t row,
                                   std::size_t col,
                                   std::size_t N,
                                   std::size_t n_regressors,
                                   float *hyperparameters,
                                   const std::vector<float> &input)
{
  std::size_t i_global, j_global;
  float cov_dist;
  // Initialize tile
  std::vector<float> tile;
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

// compute negative-log likelihood tiled
float compute_loss(const std::vector<float> &K_diag_tile,
                   const std::vector<float> &alpha_tile,
                   const std::vector<float> &y_tile,
                   std::size_t N)
{
  float l = 0.0;
  for (std::size_t i = 0; i < N; i++)
  {
    // Add the squared difference to the error
    l += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]) + y_tile[i] * alpha_tile[i];
  }
  return l;
}

// compute negative-log likelihood
float add_losses(const std::vector<float> &losses,
                 std::size_t N,
                 std::size_t n)
{
  float l = 0.0;
  for (std::size_t i = 0; i < n; i++)
  {
    // Add the squared difference to the error
    l += losses[i];
  }
  l += N * n * log(2.0 * M_PI);
  return 0.5 * l / (N * n);
}

// compute trace
float compute_trace(const std::vector<std::vector<float>> &diag_tiles,
                    std::size_t N,
                    std::size_t n_tiles)
{
  // Initialize tile
  float trace = 0.0;
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
float update_fist_moment(const float &gradient,
                         float m_T,
                         const std::vector<float> &beta1_T,
                         int iter)
{
  return beta1_T[iter] * m_T + (1.0 - beta1_T[iter]) * gradient;
}

// update first moment
float update_second_moment(const float &gradient,
                           float v_T,
                           const std::vector<float> &beta2_T,
                           int iter)
{
  return beta2_T[iter] * v_T + (1.0 - beta2_T[iter]) * gradient * gradient;
}

// update hyperparameter using gradient decent
float update_param(const float &unconstrained_hyperparam,
                   float *hyperparameters,
                   const float &gradient,
                   float m_T,
                   float v_T,
                   const std::vector<float> &beta1_T,
                   const std::vector<float> &beta2_T,
                   int iter)
{
  float alpha_T = hyperparameters[3] * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
  // T mhat = m_T / (1.0 - beta1_T[iter]);
  // T vhat = v_T / (1.0 - beta2_T[iter]);
  // return unconstrained_hyperparam - hyperparameters[3] * mhat / (sqrt(vhat) + hyperparameters[6]));
  return unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + hyperparameters[6]);
}

// transform hyperparameter to enforce constraints using softplus
float to_constrained(const float &parameter,
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
float to_unconstrained(const float &parameter,
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
float gen_beta_t(int t,
                 float *hyperparameters,
                 int param_idx)
{
  return pow(hyperparameters[param_idx], t);
}

//////////////////////////////////////////////////////////
/////////  functions with double precision  //////////////
//////////////////////////////////////////////////////////

// compute the distance devided by the lengthscale
double compute_covariance_dist_func(std::size_t i_global,
                                    std::size_t j_global,
                                    std::size_t n_regressors,
                                    double *hyperparameters,
                                    const std::vector<double> &i_input,
                                    const std::vector<double> &j_input)
{
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  double z_ik = 0.0;
  double z_jk = 0.0;
  double distance = 0.0;
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
std::vector<double> gen_tile_grad_v(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    double *hyperparameters,
                                    const std::vector<double> &input)
{
  std::size_t i_global, j_global;
  double cov_dist;
  // Initialize tile
  std::vector<double> tile;
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
std::vector<double> gen_tile_grad_l(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    double *hyperparameters,
                                    const std::vector<double> &input)
{
  std::size_t i_global, j_global;
  double cov_dist;
  // Initialize tile
  std::vector<double> tile;
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

// compute hyper-parameter beta_1 or beta_2 to power t
double gen_beta_t(int t,
                  double *hyperparameters,
                  int param_idx)
{
  return pow(hyperparameters[param_idx], t);
}

// compute negative-log likelihood tiled
double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N)
{
  double l = 0.0;
  for (std::size_t i = 0; i < N; i++)
  {
    // Add the squared difference to the error
    l += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]) + y_tile[i] * alpha_tile[i];
  }
  return l;
}

// compute negative-log likelihood
double add_losses(const std::vector<double> &losses,
                  std::size_t N,
                  std::size_t n)
{
  double l = 0.0;
  for (std::size_t i = 0; i < n; i++)
  {
    // Add the squared difference to the error
    l += losses[i];
  }
  l += N * n * log(2.0 * M_PI);
  return 0.5 * l / (N * n);
}

// compute trace
double compute_trace(const std::vector<std::vector<double>> &diag_tiles,
                     std::size_t N,
                     std::size_t n_tiles)
{
  // Initialize tile
  double trace = 0.0;
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

// transform hyperparameter to enforce constraints using softplus
double to_constrained(const double &parameter,
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
double to_unconstrained(const double &parameter,
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

// update first moment
double update_fist_moment(const double &gradient,
                          double m_T,
                          const std::vector<double> &beta1_T,
                          int iter)
{
  return beta1_T[iter] * m_T + (1.0 - beta1_T[iter]) * gradient;
}

// update first moment
double update_second_moment(const double &gradient,
                            double v_T,
                            const std::vector<double> &beta2_T,
                            int iter)
{
  return beta2_T[iter] * v_T + (1.0 - beta2_T[iter]) * gradient * gradient;
}

// update hyperparameter using gradient decent
double update_param(const double &unconstrained_hyperparam,
                    double *hyperparameters,
                    const double &gradient,
                    double m_T,
                    double v_T,
                    const std::vector<double> &beta1_T,
                    const std::vector<double> &beta2_T,
                    int iter)
{
  double alpha_T = hyperparameters[3] * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
  // T mhat = m_T / (1.0 - beta1_T[iter]);
  // T vhat = v_T / (1.0 - beta2_T[iter]);
  // return unconstrained_hyperparam - hyperparameters[3] * mhat / (sqrt(vhat) + hyperparameters[6]));
  return unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + hyperparameters[6]);
}

///////////////////////////////////////////////////////////////////////////////////
/////////  functions with template because only vary by return type  //////////////
///////////////////////////////////////////////////////////////////////////////////

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

#endif
