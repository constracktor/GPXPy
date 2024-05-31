#ifndef GP_FUNCTIONS_GRAD_H_INCLUDED
#define GP_FUNCTIONS_GRAD_H_INCLUDED

#include <cmath>
#include <vector>
#include <numeric>
#include <tuple>
#include <iostream>

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

// evaluate sigmoid function for given value
double compute_sigmoid(const double &parameter)
{
  return 1.0 / (1.0 + exp(-parameter));
}

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

  return -1.0 / (2.0 * pow(hyperparameters[0], 2.0)) * distance;
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
      tile[i * N + j] = exp(cov_dist) * compute_sigmoid(to_unconstrained(hyperparameters[1], false));
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
      tile[i * N + j] = -2.0 * (hyperparameters[1] / hyperparameters[0]) * cov_dist * exp(cov_dist) * compute_sigmoid(to_unconstrained(hyperparameters[0], false));
    }
  }
  return std::move(tile);
}

// compute hyper-parameter beta_1 or beta_2 to power t
double gen_beta_T(int t, double *hyperparameters, int param_idx)
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
double compute_gradient(const std::vector<std::vector<double>> &diag_tiles,
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
  // printf("gradient: %.12lf\n", trace);
  return std::move(trace);
}

// compute trace for noise variance
// Same function as compute_trace with the only difference
// that we need to retrieve the diag tiles
double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              double *hyperparameters,
                              std::size_t N,
                              std::size_t n_tiles)
{
  // Initialize tile
  double trace = 0.0;
  for (std::size_t d = 0; d < n_tiles; d++)
  {
    auto tile = ft_tiles[d * n_tiles + d];
    for (std::size_t i = 0; i < N; ++i)
    {
      trace += (tile[i * N + i] * compute_sigmoid(to_unconstrained(hyperparameters[2], true)));
    }
  }
  trace = 1.0 / (2.0 * N * n_tiles) * trace;
  return std::move(trace);
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

// generate an identity tile if i==j
std::vector<double> gen_tile_identity(std::size_t row,
                                      std::size_t col,
                                      std::size_t N)
{
  std::size_t i_global, j_global;
  // Initialize tile
  std::vector<double> tile;
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
std::vector<double> gen_tile_zeros_diag(std::size_t N)
{
  // Initialize tile
  std::vector<double> tile;
  tile.resize(N * N);
  std::fill(tile.begin(), tile.end(), 0.0);
  return std::move(tile);
}

// return zero - used to initialize moment vectors
double gen_zero()
{
  double z = 0.0;
  return z;
}

#endif
