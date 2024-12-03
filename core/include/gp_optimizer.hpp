#ifndef GP_OPTIMIZER_H
#define GP_OPTIMIZER_H

#include "gp_kernels.hpp"
#include <cmath>
#include <hpx/future.hpp>
#include <string>
#include <vector>

namespace gpxpy_hyper
{
/**
 * @brief Hyperparameters for the Adam optimizer
 */
struct AdamParams
{
    /** @brief TODO: documentation */
    double learning_rate;

    /** @brief TODO: documentation */
    double beta1;

    /** @brief TODO: documentation */
    double beta2;

    /** @brief TODO: documentation */
    double epsilon;

    /** @brief TODO: documentation */
    int opt_iter;

    /** @brief TODO: documentation */
    std::vector<double> M_T;

    /** @brief TODO: documentation */
    std::vector<double> V_T;

    /**
     * @brief Initialize hyperparameters
     *
     * @param lr learning rate
     * @param b1 beta1
     * @param b2 beta2
     * @param eps epsilon
     * @param opt_i number of optimization iterations
     * @param M_T_init initial values for first moment vector
     * @param V_T_init initial values for second moment vector
     */
    AdamParams(double lr = 0.001,
               double b1 = 0.9,
               double b2 = 0.999,
               double eps = 1e-8,
               int opt_i = 0,
               std::vector<double> M_T = { 0.0, 0.0, 0.0 },
               std::vector<double> V_T = { 0.0, 0.0, 0.0 });

    /**
     * @brief Returns a string representation of the hyperparameters
     */
    std::string repr() const;
};

}  // namespace gpxpy_hyper

/**
 * @brief Transform hyperparameter to enforce constraints using softplus.
 */
double to_constrained(const double parameter, bool noise);

/**
 * @brief Transform hyperparmeter to entire real line using inverse of softplus.
 *        Optimizers, such as gradient decent or Adam, work better with
 *        unconstrained parameters.
 */
double to_unconstrained(const double parameter, bool noise);

/**
 * @brief Calculates & returns sigmoid function for `parameter` as input.
 */
double compute_sigmoid(const double parameter);

/**
 * @brief Compute the distance divided by the lengthscale.
 */
double compute_covariance_dist_func(std::size_t i_global,
                                    std::size_t j_global,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams sek_params,
                                    const std::vector<double> &i_input,
                                    const std::vector<double> &j_input);

/**
 * @brief Compute vector of distances devided by the lengthscale.
 */
std::vector<double> compute_cov_dist_vec(std::size_t row,
                                         std::size_t col,
                                         std::size_t N,
                                         std::size_t n_regressors,
                                         gpxpy_hyper::SEKParams sek_params,
                                         const std::vector<double> &input);

/**
 * @brief Generate a tile of the covariance matrix.
 */
std::vector<double>
gen_tile_covariance_opt(std::size_t row,
                        std::size_t col,
                        std::size_t N,
                        std::size_t n_regressors,
                        gpxpy_hyper::SEKParams sek_params,
                        const std::vector<double> &cov_dists);

/**
 * @brief Generate a derivative tile w.r.t. vertical_lengthscale.
 */
std::vector<double> gen_tile_grad_v(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams sek_params,
                                    const std::vector<double> &cov_dists);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 */
std::vector<double> gen_tile_grad_l(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams sek_params,
                                    const std::vector<double> &cov_dists);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 */
std::vector<double>
gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 */
std::vector<double>
gen_tile_grad_l_trans(std::size_t N, const std::vector<double> &grad_l_tile);

/**
 * @brief Compute hyper-parameter beta_1 or beta_2 to power t.
 */
double gen_beta_T(int t, double beta);

/**
 * @brief Compute negative-log likelihood tiled.
 */
double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N);

/**
 * @brief Compute negative-log likelihood.
 */
double
add_losses(const std::vector<double> &losses, std::size_t N, std::size_t n);

/**
 * @brief Compute trace of (K^-1 - K^-1*y*y^T*K^-1)* del(K)/del(hyperparam) =
 *        gradient(K) w.r.t. hyperparam.
 */
double compute_gradient(const double &grad_l,
                        const double &grad_r,
                        std::size_t N,
                        std::size_t n_tiles);

/**
 * @brief Compute trace for noise variance.
 *
 * Same function as compute_trace with() the only difference that we only use
 * diag tiles multiplied by derivative of noise_variance.
 */
double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              double noise_variance,
                              std::size_t N,
                              std::size_t n_tiles);

/**
 * @brief Update biased first raw moment estimate.
 */
double
update_first_moment(const double &gradient, double m_T, const double &beta_1);

/**
 * @brief Update biased second raw moment estimate.
 */
double
update_second_moment(const double &gradient, double v_T, const double &beta_2);

/**
 * @brief Update hyperparameter using gradient decent.
 */
hpx::shared_future<double> update_param(const double unconstrained_hyperparam,
                                        gpxpy_hyper::SEKParams sek_params,
                                        gpxpy_hyper::AdamParams adam_params,
                                        double m_T,
                                        double v_T,
                                        const std::vector<double> beta1_T,
                                        const std::vector<double> beta2_T,
                                        int iter);

/**
 * @brief Generate an identity tile if i==j.
 */
std::vector<double>
gen_tile_identity(std::size_t row, std::size_t col, std::size_t N);

/**
 * @brief Generate an empty tile NxN.
 */
std::vector<double> gen_tile_zeros_diag(std::size_t N);

/**
 * @brief return zero - used to initialize moment vectors
 */
double gen_moment();

double sum_gradleft(const std::vector<double> &diagonal, double grad);

double sum_gradright(const std::vector<double> &inter_alpha,
                     const std::vector<double> &alpha,
                     double grad,
                     std::size_t N);

double sum_noise_gradleft(const std::vector<double> &ft_invK,
                          double grad,
                          gpxpy_hyper::SEKParams sek_params,
                          std::size_t N,
                          std::size_t n_tiles);

double sum_noise_gradright(const std::vector<double> &alpha,
                           double grad,
                           gpxpy_hyper::SEKParams sek_params,
                           std::size_t N);

#endif  // end of GP_OPTIMIZER_H
