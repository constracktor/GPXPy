#ifndef GP_FUNCTIONS_H_INCLUDED
#define GP_FUNCTIONS_H_INCLUDED
// #define _USE_MATH_DEFINES

#include <cmath>
#include <vector>
#include <numeric>

// compute the squared exponential kernel of two feature vectors
float compute_covariance_function(std::size_t i_global,
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
  return hyperparameters[1] * exp(-1.0 / (2.0 * hyperparameters[0]) * distance);
}

// generate a tile of the covariance matrix
std::vector<float> gen_tile_covariance(std::size_t row,
                                       std::size_t col,
                                       std::size_t N,
                                       std::size_t n_regressors,
                                       float *hyperparameters,
                                       const std::vector<float> &input)
{
  std::size_t i_global, j_global;
  float covariance_function;
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
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
      if (i_global == j_global)
      {
        // noise variance on diagonal
        covariance_function += (hyperparameters[2]);
      }
      tile[i * N + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile of the prior covariance matrix
std::vector<float> gen_tile_prior_covariance(std::size_t row,
                                             std::size_t col,
                                             std::size_t N,
                                             std::size_t n_regressors,
                                             float *hyperparameters,
                                             const std::vector<float> &input)
{
  std::size_t i_global, j_global;
  float covariance_function;
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
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
      tile[i * N + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile of the cross-covariance matrix
std::vector<float> gen_tile_cross_covariance(std::size_t row,
                                             std::size_t col,
                                             std::size_t N_row,
                                             std::size_t N_col,
                                             std::size_t n_regressors,
                                             float *hyperparameters,
                                             const std::vector<float> &row_input,
                                             const std::vector<float> &col_input)
{
  std::size_t i_global, j_global;
  float covariance_function;
  // Initialize tile
  std::vector<float> tile;
  tile.resize(N_row * N_col);
  for (std::size_t i = 0; i < N_row; i++)
  {
    i_global = N_row * row + i;
    for (std::size_t j = 0; j < N_col; j++)
    {
      j_global = N_col * col + j;
      // compute covariance function
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, row_input, col_input);
      tile[i * N_col + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile containing the output observations
std::vector<float> gen_tile_output(std::size_t row,
                                   std::size_t N,
                                   const std::vector<float> &output)
{
  std::size_t i_global;
  // Initialize tile
  std::vector<float> tile;
  tile.resize(N);
  for (std::size_t i = 0; i < N; i++)
  {
    i_global = N * row + i;
    tile[i] = output[i_global];
  }
  return std::move(tile);
}

// compute the total 2-norm error
float compute_error_norm(std::size_t n_tiles,
                         std::size_t tile_size,
                         const std::vector<float> &b,
                         const std::vector<std::vector<float>> &tiles)
{
  float error = 0.0;
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto a = tiles[k];
    for (std::size_t i = 0; i < tile_size; i++)
    {
      std::size_t i_global = tile_size * k + i;
      // ||a - b||_2
      error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
    }
  }
  return std::move(sqrt(error));
}

//////////////////////////////////////////////////////////
/////////  functions with double precision  //////////////
//////////////////////////////////////////////////////////

// compute the squared exponential kernel of two feature vectors
double compute_covariance_function(std::size_t i_global,
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
  return hyperparameters[1] * exp(-1.0 / (2.0 * hyperparameters[0]) * distance);
}

// generate a tile of the covariance matrix
std::vector<double> gen_tile_covariance(std::size_t row,
                                        std::size_t col,
                                        std::size_t N,
                                        std::size_t n_regressors,
                                        double *hyperparameters,
                                        const std::vector<double> &input)
{
  std::size_t i_global, j_global;
  double covariance_function;
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
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
      if (i_global == j_global)
      {
        // noise variance on diagonal
        covariance_function += (hyperparameters[2]);
      }
      tile[i * N + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile of the prior covariance matrix
std::vector<double> gen_tile_prior_covariance(std::size_t row,
                                              std::size_t col,
                                              std::size_t N,
                                              std::size_t n_regressors,
                                              double *hyperparameters,
                                              const std::vector<double> &input)
{
  std::size_t i_global, j_global;
  double covariance_function;
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
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
      tile[i * N + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile of the cross-covariance matrix
std::vector<double> gen_tile_cross_covariance(std::size_t row,
                                              std::size_t col,
                                              std::size_t N_row,
                                              std::size_t N_col,
                                              std::size_t n_regressors,
                                              double *hyperparameters,
                                              const std::vector<double> &row_input,
                                              const std::vector<double> &col_input)
{
  std::size_t i_global, j_global;
  double covariance_function;
  // Initialize tile
  std::vector<double> tile;
  tile.resize(N_row * N_col);
  for (std::size_t i = 0; i < N_row; i++)
  {
    i_global = N_row * row + i;
    for (std::size_t j = 0; j < N_col; j++)
    {
      j_global = N_col * col + j;
      // compute covariance function
      covariance_function = compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, row_input, col_input);
      tile[i * N_col + j] = covariance_function;
    }
  }
  return std::move(tile);
}

// generate a tile containing the output observations
std::vector<double> gen_tile_output(std::size_t row,
                                    std::size_t N,
                                    const std::vector<double> &output)
{
  std::size_t i_global;
  // Initialize tile
  std::vector<double> tile;
  tile.resize(N);
  for (std::size_t i = 0; i < N; i++)
  {
    i_global = N * row + i;
    tile[i] = output[i_global];
  }
  return std::move(tile);
}

// compute the total 2-norm error
double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
  double error = 0.0;
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto a = tiles[k];
    for (std::size_t i = 0; i < tile_size; i++)
    {
      std::size_t i_global = tile_size * k + i;
      // ||a - b||_2
      error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
    }
  }
  return std::move(sqrt(error));
}

///////////////////////////////////////////////////////////////////////////////////
/////////  functions with template because only vary by return type  //////////////
///////////////////////////////////////////////////////////////////////////////////

// generate a empty tile
template <typename T>
std::vector<T> gen_tile_zeros(std::size_t N)
{
  // Initialize tile
  std::vector<T> tile;
  tile.resize(N);
  std::fill(tile.begin(), tile.end(), 0.0);
  return std::move(tile);
}
#endif
