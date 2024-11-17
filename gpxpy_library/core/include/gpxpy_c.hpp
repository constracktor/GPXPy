#ifndef gpxpy_C_H
#define gpxpy_C_H

#include <vector>
#include <string>
#include "gp_functions.hpp"

namespace gpxpy
{

    /**
     * @brief Data structure for Gaussian process data
     *
     * It includes the file path to the data, the number of samples, and the
     * data itself.
     */
    struct GP_data
    {
        std::string file_path; ///< Path to the file containing the data
        int n_samples; ///< Number of samples in the data
        std::vector<double> data; ///< Vector containing the data

        /**
        * @brief Construct a new GP_data object
        *
        * @param file_path Path to the file containing the data
        * @param n Number of samples to read from the file
        */
        GP_data(const std::string& file_path, int n);
    };


    /**
     * @brief Gaussian Process class for regression tasks
     *
     * This class provides methods for training a Gaussian Process model, making
     * predictions, optimizing hyperparameters, and calculating loss. It also
     * includes methods for computing the Cholesky decomposition.
     */
    class GP
    {
    private:
        std::vector<double> _training_input; ///< Input data for training
        std::vector<double> _training_output; ///< Output data for given input data
        int _n_tiles; ///< Number of tiles
        int _n_tile_size; ///< Size of each square tile in each dimension

    public:
        double lengthscale; ///< "l" parameter of squared exponential kernel
        double vertical_lengthscale; ///< "v" parameter of squared exponential kernel

        /**
         * "sigma" parameter of squared exponential kernel, also referred to
         * as `n`
         */
        double noise_variance;

        int n_regressors; ///< Number of regressors

        ///< True bools indicate trainable parameters, else not trainable
        std::vector<bool> trainable_params;


        /**
         * @brief Constructs a Gaussian Process (GP)
         *
         * @param input Input data for training of the GP
         * @param output Expected output data for training of the GP
         * @param n_tiles Number of tiles
         * @param n_tile_size Size of each tile in each dimension
         * @param l Lengthscale Parameter of squared exponential kernel: l
         * @param v Vertical Lengthscale parameter of squared exponential kernel: v
         * @param n Noise Variance parameter of squared exponential kernel: n
         * @param n_regressors Number of regressors
         * @param trainable_bool Vector indicating which parameters are trainable
         */
        GP(std::vector<double> input, std::vector<double> output, int n_tiles,
           int n_tile_size, double l, double v, double n, int n_regressors,
           std::vector<bool> trainable_bool);

        /**
         * Print Gausian process attributes
         */
        std::string repr() const;

        /**
         * Returns size of input data for training
         */
        std::vector<double> get_training_input() const;

        /**
         * Returns size of output data for training
         */
        std::vector<double> get_training_output() const;

        // Predict output for test input
        std::vector<double> predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Predict output for test input and additionally provide uncertainty for the predictions
        std::vector<std::vector<double>> predict_with_uncertainty(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Predict output for test input and additionally compute full posterior covariance matrix
        std::vector<std::vector<double>> predict_with_full_cov(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        // Optimize hyperparameters for a specified number of iterations
        std::vector<double> optimize(const gpxpy_hyper::Hyperparameters &hyperparams);

        // Perform a single optimization step
        double optimize_step(gpxpy_hyper::Hyperparameters &hyperparams, int iter);

        // Calculate loss for given data and Gaussian process model
        double calculate_loss();

        // Compute Cholesky decomposition (Returns L) <- not usable yet. Only purpose right now
        // is to measure performance to compare against PyTorch torch.linalg.cholesky()
        std::vector<std::vector<double>> cholesky();
    };
}

#endif
