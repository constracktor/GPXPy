#ifndef GP_FUNCTIONS_H
#define GP_FUNCTIONS_H
// #define _USE_MATH_DEFINES

#include <vector>
#include <hpx/future.hpp>

namespace gpppy_hyper
{
    struct Hyperparameters
    {
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        int opt_iter;
        std::vector<double> M_T;
        std::vector<double> V_T;

        Hyperparameters(double lr = 0.001, double b1 = 0.9,
                        double b2 = 0.999, double eps = 1e-8, int opt_i = 0,
                        std::vector<double> M_T = {0.0, 0.0, 0.0}, std::vector<double> V_T = {0.0, 0.0, 0.0});

        std::string repr() const;
    };
}

hpx::shared_future<std::vector<std::vector<double>>> predict_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                                                 const std::vector<double> &test_data, int n_tiles, int n_tile_size,
                                                                 int m_tiles, int m_tile_size, double lengthscale, double vertical_lengthscale,
                                                                 double noise_variance, int n_regressors);

hpx::shared_future<double> compute_loss_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                            int n_tiles, int n_tile_size, int n_regressors, double *hyperparameters);

hpx::shared_future<std::vector<double>> optimize_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                                     int n_tiles, int n_tile_size, double &lengthscale, double &vertical_lengthscale,
                                                     double &noise_variance, int n_regressors, const gpppy_hyper::Hyperparameters &hyperparams,
                                                     std::vector<bool> trainable_params);

hpx::shared_future<double> optimize_step_hpx(const std::vector<double> &training_input, const std::vector<double> &training_output,
                                             int n_tiles, int n_tile_size, double &lengthscale, double &vertical_lengthscale,
                                             double &noise_variance, int n_regressors, gpppy_hyper::Hyperparameters &hyperparams,
                                             std::vector<bool> trainable_params, int iter);

#endif