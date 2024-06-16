#include "../include/automobile_bits/gpppy_c.hpp"

#include <stdexcept>
#include <cstdio>
#include <sstream>

namespace gpppy
{
    // Implementation of the Car constructor
    GP_data::GP_data(const std::string &f_path, int n)
    {
        n_samples = n;
        file_path = f_path;
        data = load_data(f_path, n);
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

    std::vector<double> load_data(const std::string &file_path, int n_samples)
    {
        std::vector<double> _data;
        _data.resize(n_samples);

        FILE *input_file = fopen(file_path.c_str(), "r");
        if (input_file == NULL)
        {
            throw std::runtime_error("Error: File not found: " + file_path);
        }

        // load data
        std::size_t scanned_elements = 0;
        for (int i = 0; i < n_samples; i++)
        {
            // if (fscanf(input_file, "%lf", &data[i]) != 1)
            // {
            //     fclose(training_input_file);
            //     throw std::runtime_error("Error: Failed to read data element at index " + std::to_string(i));
            // }
            scanned_elements += fscanf(input_file, "%lf", &_data[i]); // scanned_elements++;
        }

        fclose(input_file);

        if (scanned_elements != n_samples)
        {
            throw std::runtime_error("Error: Data not correctly read. Expected " + std::to_string(n_samples) + " elements, but read " + std::to_string(scanned_elements));
        }

        return std::move(_data);
    }
}
