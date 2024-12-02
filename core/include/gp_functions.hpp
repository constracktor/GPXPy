#ifndef GP_FUNCTIONS_H
#define GP_FUNCTIONS_H

#include "gp_kernels.hpp"
#include "gp_optimizer.hpp"
#include "target.hpp"
#include <hpx/future.hpp>
#include <vector>

/**
 * @brief Compute the predictions
 */
hpx::shared_future<std::vector<double>>
predict_on_target(const std::vector<double> &training_input,
                  const std::vector<double> &training_output,
                  const std::vector<double> &test_data,
                  int n_tiles,
                  int n_tile_size,
                  int m_tiles,
                  int m_tile_size,
                  int n_regressors,
                  gpxpy_hyper::SEKParams sek_params,
                  gpxpy::Target &target);

/**
 * @brief Compute the predictions and uncertainties
 */
hpx::shared_future<std::vector<std::vector<double>>>
predict_with_uncertainty_on_target(const std::vector<double> &training_input,
                                   const std::vector<double> &training_output,
                                   const std::vector<double> &test_input,
                                   int n_tiles,
                                   int n_tile_size,
                                   int m_tiles,
                                   int m_tile_size,
                                   int n_regressors,
                                   gpxpy_hyper::SEKParams sek_params,
                                   gpxpy::Target &target);

/**
 * @brief Compute the predictions and full covariance matrix
 */
hpx::shared_future<std::vector<std::vector<double>>>
predict_with_full_cov_on_target(const std::vector<double> &training_input,
                                const std::vector<double> &training_output,
                                const std::vector<double> &test_data,
                                int n_tiles,
                                int n_tile_size,
                                int m_tiles,
                                int m_tile_size,
                                int n_regressors,
                                gpxpy_hyper::SEKParams sek_params,
                                gpxpy::Target &target);

/**
 * @brief Compute loss for given data and Gaussian process model
 */
hpx::shared_future<double>
compute_loss_on_target(const std::vector<double> &training_input,
                       const std::vector<double> &training_output,
                       int n_tiles,
                       int n_tile_size,
                       int n_regressors,
                       gpxpy_hyper::SEKParams sek_params,
                       gpxpy::Target &target);

/**
 * @brief Perform optimization for a given number of iterations
 */
hpx::shared_future<std::vector<double>>
optimize_on_target(const std::vector<double> &training_input,
                   const std::vector<double> &training_output,
                   int n_tiles,
                   int n_tile_size,
                   int n_regressors,
                   gpxpy_hyper::SEKParams &sek_params,
                   std::vector<bool> trainable_params,
                   const gpxpy_hyper::AdamParams &adam_params,
                   gpxpy::Target &target);

/**
 * @brief Perform a single optimization step
 */
hpx::shared_future<double>
optimize_step_on_target(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors,
                        int iter,
                        gpxpy_hyper::SEKParams &sek_params,
                        std::vector<bool> trainable_params,
                        gpxpy_hyper::AdamParams &adam_params,
                        gpxpy::Target &target);

/**
 * @brief Compute Cholesky decomposition
 */
hpx::shared_future<std::vector<std::vector<double>>>
cholesky_on_target(const std::vector<double> &training_input,
                   const std::vector<double> &training_output,
                   int n_tiles,
                   int n_tile_size,
                   int n_regressors,
                   gpxpy_hyper::SEKParams sek_params,
                   gpxpy::Target &target);

#endif  // end of GP_FUNCTIONS_H
