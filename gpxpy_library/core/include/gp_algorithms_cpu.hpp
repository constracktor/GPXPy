#ifndef GP_HELPER_FUNCTIONS_H
#define GP_HELPER_FUNCTIONS_H

#include <cmath>
#include <vector>

// compute the squared exponential kernel of two feature vectors
double compute_covariance_function(std::size_t i_global, std::size_t j_global,
                                   std::size_t n_regressors,
                                   double* hyperparameters,
                                   const std::vector<double>& i_input,
                                   const std::vector<double>& j_input);

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param row row index of the tile
 * @param col column index of the tile
 * @param N size of the tile
 * @param n_regressors number of regressors
 * @param hyperparameters hyperparameters of the covariance function
 * @param input input data
 */
std::vector<double> gen_tile_covariance(std::size_t row, std::size_t col,
                                        std::size_t N, std::size_t n_regressors,
                                        double* hyperparameters,
                                        const std::vector<double>& input);

// generate a tile of the prior covariance matrix
std::vector<double> gen_tile_full_prior_covariance(
    std::size_t row, std::size_t col, std::size_t N, std::size_t n_regressors,
    double* hyperparameters, const std::vector<double>& input);

// generate a tile of the prior covariance matrix
std::vector<double> gen_tile_prior_covariance(std::size_t row, std::size_t col,
                                              std::size_t N,
                                              std::size_t n_regressors,
                                              double* hyperparameters,
                                              const std::vector<double>& input);

// generate a tile of the cross-covariance matrix
std::vector<double> gen_tile_cross_covariance(
    std::size_t row, std::size_t col, std::size_t N_row, std::size_t N_col,
    std::size_t n_regressors, double* hyperparameters,
    const std::vector<double>& row_input, const std::vector<double>& col_input);

// generate a tile of the cross-covariance matrix
std::vector<double>
gen_tile_cross_cov_T(std::size_t N_row, std::size_t N_col,
                     const std::vector<double>& cross_covariance_tile);

// generate a tile containing the output observations
std::vector<double> gen_tile_output(std::size_t row, std::size_t N,
                                    const std::vector<double>& output);

// compute the total 2-norm error
double compute_error_norm(std::size_t n_tiles, std::size_t tile_size,
                          const std::vector<double>& b,
                          const std::vector<std::vector<double>>& tiles);

// generate an empty tile
std::vector<double> gen_tile_zeros(std::size_t N);

void assemble_tiled_K();

#endif // end of GP_HELPER_FUNCTIONS_H
