#ifndef GP_FUNCTIONS_H
#define GP_FUNCTIONS_H
// #define _USE_MATH_DEFINES

#include <vector>
#include <hpx/future.hpp>
// generate a empty tile
hpx::shared_future<std::vector<std::vector<double>>> preditc_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                                                 const std::vector<double> &test_data, int n_tiles, int n_tile_size,
                                                                 int m_tiles, int m_tile_size, double lengthscale, double vertical_lengthscale,
                                                                 double noise_variance, int n_regressors);

#endif