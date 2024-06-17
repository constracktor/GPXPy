#include "../include/automobile_bits/gp_functions.hpp"
#include "../include/automobile_bits/functions.hpp"
#include "../include/automobile_bits/tiled_algorithms_cpu.hpp"

#include <cmath>
#include <vector>
#include <numeric>

hpx::shared_future<std::vector<std::vector<double>>> preditc_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                                                 const std::vector<double> &test_input, int n_tiles, int n_tile_size,
                                                                 int m_tiles, int m_tile_size, double lengthscale, double vertical_lengthscale,
                                                                 double noise_variance, int n_regressors)
{

    double hyperparameters[3];
    hyperparameters[0] = lengthscale;          // lengthscale = variance of training_output
    hyperparameters[1] = vertical_lengthscale; // vertical_lengthscale = standard deviation of training_input
    hyperparameters[2] = noise_variance;       // noise_variance = small value
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prior_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> t_cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prediction_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prediction_uncertainty_tiles;

    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_covariance, "assemble_tiled"), i, j,
                                                  n_tile_size, n_regressors, hyperparameters, training_input);
        }
    }

    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }

    // Assemble prior covariance matrix vector
    prior_K_tiles.resize(m_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prior_K_tiles[i * m_tiles + i] = hpx::async(hpx::annotated_function(&gen_tile_prior_covariance, "assemble_tiled"), i, i,
                                                    m_tile_size, n_regressors, hyperparameters, test_input);
    }
    // Assemble MxN cross-covariance matrix vector
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance, "assemble_tiled"), i, j,
                                                                 m_tile_size, n_tile_size, n_regressors, hyperparameters, test_input, training_input);
        }
    }
    // Assemble NxM (transpose) cross-covariance matrix vector
    t_cross_covariance_tiles.resize(n_tiles * m_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < m_tiles; j++)
        {
            t_cross_covariance_tiles[i * m_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance, "assemble_tiled"), i, j,
                                                                   n_tile_size, m_tile_size, n_regressors, hyperparameters, training_input, test_input);
        }
    }
    // Assemble zero prediction
    prediction_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), m_tile_size);
    }
    // Assemble zero prediction
    prediction_uncertainty_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_uncertainty_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), m_tile_size);
    }

    right_looking_cholesky_tiled_mkl(K_tiles, n_tile_size, n_tiles);
    // Triangular solve K_NxN * alpha = y
    // forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);
    // backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);

    /////////////////

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KK_tiled(K_tiles, cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
    backward_solve_KK_tiled(K_tiles, cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    //////////////////////////////////////////////////////////////////////////////
    // PART 3: PREDICTION
    prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    posterior_covariance_tiled(cross_covariance_tiles, t_cross_covariance_tiles, prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
    // predicition uncertainty
    prediction_uncertainty_tiled(prior_K_tiles, prediction_uncertainty_tiles, m_tile_size, m_tiles);

    std::vector<double> pred;
    std::vector<double> pred_var;
    pred.reserve(test_input.size());     // preallocate memory
    pred_var.reserve(test_input.size()); // preallocate memory
    for (std::size_t i; i < m_tiles; i++)
    {
        pred.insert(pred.end(), prediction_tiles[i].get().begin(), prediction_tiles[i].get().end());
        pred_var.insert(pred_var.end(), prediction_uncertainty_tiles[i].get().begin(), prediction_uncertainty_tiles[i].get().end());
    }

    return hpx::async([pred, pred_var]()
                      {
        std::vector<std::vector<double>> result(2);
        result[0] = pred;
        result[1] = pred_var;
        return result; });
}