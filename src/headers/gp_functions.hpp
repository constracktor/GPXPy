#ifndef GP_FUNCTIONS_H_INCLUDED
#define GP_FUNCTIONS_H_INCLUDED

#include <cmath>
#include <vector>

// compute feature vector z_i
template <typename T>
std::vector<T> compute_regressor_vector(std::size_t row,
                                        std::size_t n_regressors,
                                        std::vector<T> input)
{
  std::vector<T> regressor_vector;
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

// compute the squared exponential kernel of two feature vectors
template <typename T>
T compute_covariance_function(std::size_t n_regressors,
                              T* hyperparameters,
                              std::vector<T> z_i,
                              std::vector<T> z_j)
{
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  T distance = 0.0;
  for (std::size_t i = 0; i < n_regressors; i++)
  {
    distance += pow(z_i[i] - z_j[i],2);
  }
  return hyperparameters[1] * exp(-0.5 * hyperparameters[0] * distance);
}

// generate a tile of the covariance matrix
template <typename T>
std::vector<T> gen_tile_covariance(std::size_t row,
                                   std::size_t col,
                                   std::size_t N,
                                   std::size_t n_regressors,
                                   T *hyperparameters,
                                   std::vector<T> input)
{
   std::size_t i_global,j_global;
   T covariance_function;
   std::vector<std::vector<T>> z_row, z_col;
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
   std::vector<T> tile;
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
          // noise variance on diagonal
          covariance_function += hyperparameters[2];
        }
        tile[i * N + j] = covariance_function;
      }
   }
   return tile;
}

// generate a tile containing the output observations
template <typename T>
std::vector<T> gen_tile_output(std::size_t row,
                               std::size_t N,
                               std::vector<T> output)
{
   std::size_t i_global;
   // Initialize tile
   std::vector<T> tile;
   tile.resize(N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      tile[i] = output[i_global];
   }
   return tile;
}

// generate a tile of the cross-covariance matrix
template <typename T>
std::vector<T> gen_tile_cross_covariance(std::size_t row,
                                         std::size_t col,
                                         std::size_t N_row,
                                         std::size_t N_col,
                                         std::size_t n_regressors,
                                         T *hyperparameters,
                                         std::vector<T> row_input,
                                         std::vector<T> col_input)
{
   std::size_t i_global,j_global;
   T covariance_function;
   std::vector<std::vector<T>> z_row, z_col;
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
   std::vector<T> tile;
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

// generate a empty tile
template <typename T>
std::vector<T> gen_tile_zeros(std::size_t N)
{
   // Initialize tile
   std::vector<T> tile;
   tile.resize(N);
   std::fill(tile.begin(),tile.end(),0.0);
   return tile;
}

// compute the total 2-norm error
template <typename T>
T compute_error_norm(std::vector<std::vector<T>> tiles,
                     std::vector<T> b,
                     std::size_t n_tiles,
                     std::size_t tile_size)
{
  std::vector<T> vector;
  vector.resize(n_tiles);
  T error = 0.0;
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto a = tiles[k];
    for(std::size_t i = 0; i < tile_size; i++)
    {
      std::size_t i_global = tile_size * k + i;
      // ||a - b||_2
      error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
    }
  }
  return sqrt(error);
}
#endif
