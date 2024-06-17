#include "../include/automobile_bits/gpppy_c.hpp"
#include "../include/automobile_bits/utils_c.hpp"
#include "../include/automobile_bits/gp_functions.hpp"

#include <stdexcept>
#include <iomanip>
#include <cstdio>
#include <sstream>

namespace gpppy
{
    // Implementation of the Car constructor
    GP_data::GP_data(const std::string &f_path, int n)
    {
        n_samples = n;
        file_path = f_path;
        data = utils::load_data(f_path, n);
    }

    Kernel_Params::Kernel_Params(double l, double v, double n, int n_r)
        : lengthscale(l), vertical_lengthscale(v), noise_variance(n), n_regressors(n_r) {}

    std::string Kernel_Params::repr() const
    {
        std::ostringstream oss;
        oss << "Kernel_Params: [lengthscale=" << lengthscale
            << ", vertical_lengthscale=" << vertical_lengthscale
            << ", noise_variance=" << noise_variance
            << ", n_regressors=" << n_regressors << "]";
        return oss.str();
    }

    Hyperparameters::Hyperparameters(double lr, double b1, double b2, double eps, int opt_i)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), opt_iter(opt_i) {}

    std::string Hyperparameters::repr() const
    {
        std::ostringstream oss;
        oss << "Hyperparameters: [learning_rate=" << learning_rate
            << ", beta1=" << beta1
            << ", beta2=" << beta2
            << ", epsilon=" << epsilon
            << ", opt_iter=" << opt_iter << "]";
        return oss.str();
    }

    GP::GP(std::vector<double> input, std::vector<double> output, int n_tiles, int n_tile_size, double l, double v, double n, int n_r)
    {
        _training_input = input;
        _training_output = output;
        _n_tiles = n_tiles;
        _n_tile_size = n_tile_size;
        lengthscale = l;
        vertical_lengthscale = v;
        noise_variance = n;
        n_regressors = n_r;
    }

    std::vector<double> GP::get_training_input() const
    {
        return _training_input;
    }

    std::vector<double> GP::get_training_output() const
    {
        return _training_output;
    }

    std::string GP::repr() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(5);
        oss << "Kernel_Params: [lengthscale=" << lengthscale
            << ", vertical_lengthscale=" << vertical_lengthscale
            << ", noise_variance=" << noise_variance
            << ", n_regressors=" << n_regressors << "]";
        return oss.str();
    }

    std::vector<std::vector<double>> GP::predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size)
    {
        hpx::shared_future<std::vector<std::vector<double>>> fut = preditc_hpx(_training_input, _training_output, test_data,
                                                                               _n_tiles, _n_tile_size, m_tiles, m_tile_size,
                                                                               lengthscale, vertical_lengthscale, noise_variance, n_regressors);
    
        // hpx::async([input, output]()
        //  { return add_vectors(input, output); });

        std::vector<std::vector<double>> result;
        hpx::threads::run_as_hpx_thread([&result, &fut]()
                                        {
                                            result = fut.get(); // Wait for and get the result from the future
                                        });
        return result;
    }

}
