#ifndef GPPPY_C_H
#define GPPPY_C_H

#include <vector>
#include <string>
#include "gp_functions.hpp"

namespace gpppy
{

    struct GP_data
    {
        int n_samples;
        std::string file_path;
        std::vector<double> data;

        // Initialize of the Gaussian process data constructor
        GP_data(const std::string &f_path, int n);
    };

    class GP
    {
    private:
        std::vector<double> _training_input;
        std::vector<double> _training_output;
        int _n_tiles;
        int _n_tile_size;

    public:
        double lengthscale;
        double vertical_lengthscale;
        double noise_variance;
        int n_regressors;
        std::vector<bool> trainable_params;

        // Initialize of the Gaussian process constructor
        GP(std::vector<double> input, std::vector<double> output, int n_tiles, int n_tile_size, double l, double v, double n, int n_r, std::vector<bool> trainable_bool);

        // Print Gausian process attributes
        std::string repr() const;

        // Return training input data
        std::vector<double> get_training_input() const;

        // Return training output data
        std::vector<double> get_training_output() const;

        // Predict output for test input
        std::vector<double> predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Predict output for test input and additionally provide uncertainty for the predictions
        std::vector<std::vector<double>> predict_with_uncertainty(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Predict output for test input and additionally compute full posterior covariance matrix
        std::vector<std::vector<double>> predict_with_full_cov(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Optimize hyperparameters for a specified number of iterations
        std::vector<double> optimize(const gpppy_hyper::Hyperparameters &hyperparams);

        // Perform a single optimization step
        double optimize_step(gpppy_hyper::Hyperparameters &hyperparams, int iter);

        // Calculate loss for given data and Gaussian process model
        double calculate_loss();

        // Compute Cholesky decomposition (Returns L) <- not usuable yet. Only purpose right now
        // is to measure performance to compare against PyTorch torch.linalg.cholesky()
        std::vector<std::vector<double>> cholesky();
    };

}

#endif