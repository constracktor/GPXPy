#ifndef GP_FUNCTIONS_GRAD_H_INCLUDED
#define GP_FUNCTIONS_GRAD_H_INCLUDED

#include <cmath>
#include <vector>

// transform hyperparameter to enforce constraints using softplus
double to_constrained(const double& parameter, bool noise);

// transform hyperparmeter to entire real line using inverse
// of sofplus. Optimizers, such as gradient decent or Adam,
// work better with unconstrained parameters
double to_unconstrained(const double& parameter, bool noise);

// evaluate sigmoid function for given value
double compute_sigmoid(const double& parameter);

// compute the distance devided by the lengthscale
double compute_covariance_dist_func(std::size_t i_global, std::size_t j_global,
                                    std::size_t n_regressors,
                                    double* hyperparameters,
                                    const std::vector<double>& i_input,
                                    const std::vector<double>& j_input);

// compute vector of distances devided by the lengthscale
std::vector<double> compute_cov_dist_vec(std::size_t row, std::size_t col,
                                         std::size_t N,
                                         std::size_t n_regressors,
                                         double* hyperparameters,
                                         const std::vector<double>& input);

// generate a tile of the covariance matrix
std::vector<double>
gen_tile_covariance_opt(std::size_t row, std::size_t col, std::size_t N,
                        std::size_t n_regressors, double* hyperparameters,
                        const std::vector<double>& cov_dists);

// generate a derivative tile w.r.t. vertical_lengthscale
std::vector<double> gen_tile_grad_v(std::size_t row, std::size_t col,
                                    std::size_t N, std::size_t n_regressors,
                                    double* hyperparameters,
                                    const std::vector<double>& cov_dists);

// generate a derivative tile w.r.t. lengthscale
std::vector<double> gen_tile_grad_l(std::size_t row, std::size_t col,
                                    std::size_t N, std::size_t n_regressors,
                                    double* hyperparameters,
                                    const std::vector<double>& cov_dists);

// generate a derivative tile w.r.t. lengthscale
std::vector<double>
gen_tile_grad_v_trans(std::size_t N, const std::vector<double>& grad_l_tile);

// generate a derivative tile w.r.t. lengthscale
std::vector<double>
gen_tile_grad_l_trans(std::size_t N, const std::vector<double>& grad_l_tile);

// compute hyper-parameter beta_1 or beta_2 to power t
double gen_beta_T(int t, double* hyperparameters, int param_idx);

// compute negative-log likelihood tiled
double compute_loss(const std::vector<double>& K_diag_tile,
                    const std::vector<double>& alpha_tile,
                    const std::vector<double>& y_tile, std::size_t N);

// compute negative-log likelihood
double add_losses(const std::vector<double>& losses, std::size_t N,
                  std::size_t n);

// compute trace of (K^-1 - K^-1*y*y^T*K^-1)* del(K)/del(hyperparam) =
// gradient(K) w.r.t. hyperparam
double compute_gradient(const double& grad_l, const double& grad_r,
                        std::size_t N, std::size_t n_tiles);

// compute trace for noise variance
// Same function as compute_trace with() the only difference
// that we only use diag tiles multiplied by derivative of noise_variance
double compute_gradient_noise(const std::vector<std::vector<double>>& ft_tiles,
                              double* hyperparameters, std::size_t N,
                              std::size_t n_tiles);

// Update biased first raw moment estimate
double update_fist_moment(const double& gradient, double m_T,
                          const double& beta_1);

// Update biased second raw moment estimate
double update_second_moment(const double& gradient, double v_T,
                            const double& beta_2);

// update hyperparameter using gradient decent
double update_param(const double& unconstrained_hyperparam,
                    double* hyperparameters, const double& gradient, double m_T,
                    double v_T, const std::vector<double>& beta1_T,
                    const std::vector<double>& beta2_T, int iter);

// generate an identity tile if i==j
std::vector<double> gen_tile_identity(std::size_t row, std::size_t col,
                                      std::size_t N);

// generate an empty tile NxN
std::vector<double> gen_tile_zeros_diag(std::size_t N);

// return zero - used to initialize moment vectors
double gen_zero();

double sum_gradleft(const std::vector<double>& diagonal, double grad);

double sum_gradright(const std::vector<double>& inter_alpha,
                     const std::vector<double>& alpha, double grad,
                     std::size_t N);

double sum_noise_gradleft(const std::vector<double>& ft_invK, double grad,
                          double* hyperparameters, std::size_t N,
                          std::size_t n_tiles);


double sum_noise_gradright(const std::vector<double>& alpha, double grad,
                           double* hyperparameters, std::size_t N);

#endif
