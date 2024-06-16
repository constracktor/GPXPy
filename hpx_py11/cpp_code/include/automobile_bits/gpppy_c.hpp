#include <vector>
#include <string>

// #include <iostream>

// #include <hpx/hpx_init.hpp>
// #include <hpx/future.hpp>

#ifndef GPPPY_C_H
#define GPPPY_C_H

namespace gpppy
{

    struct GP_data
    {
        int n_samples;
        std::string file_path;
        std::vector<double> data;

        GP_data(const std::string &f_path, int n);
    };

    struct Kernel_Params
    {
        double lengthscale;
        double vertical_lengthscale;
        double noise_variance;
        int n_regressors;

        Kernel_Params(double l = 1.0, double v = 1.0, double n = 0.1, int n_r = 100);

        std::string repr() const;
    };

    struct Hyperparameters
    {
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        int opt_iter;

        Hyperparameters(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, int opt_i = 0);

        std::string repr() const;
    };

    class GP
    {

    private:
        std::vector<double> _training_input;
        std::vector<double> _training_output;

        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_v_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_l_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prior_K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> y_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> t_cross_covariance_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_uncertainty_tiles;

    public:
        /// Constructor
        GP(std::vector<double> input, std::vector<double> output);

        std::vector<double> get_training_input() const;

        std::vector<double> get_training_output() const;

        //     void set(int s);

        //     /// Get name
        //     /// @return Name
        //     std::string get_stud_id() const;

        //     void do_fut() const;

        //     int add(int i, int j);
    };

}

#endif
