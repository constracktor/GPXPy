#ifndef gpxpy_C_H
#define gpxpy_C_H

#include "gp_functions.hpp"
#include <string>
#include <vector>

// namespace for GPXPy library entities
namespace gpxpy
{

    /**
     * @brief Data structure for Gaussian process data
     *
     * It includes the file path to the data, the number of samples, and the
     * data itself which contains this many samples.
     */
    struct GP_data
    {
        /** @brief Path to the file containing the data */
        std::string file_path;

        /** @brief Number of samples in the data */
        int n_samples;

        /** @brief Vector containing the data */
        std::vector<double> data;

        /**
         * @brief Initialize of Gaussian process data by loading data from a
         * file.
         *
         * The file specified by `f_path` must contain `n` samples.
         *
         * @param f_path Path to the file
         * @param n Number of samples
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

        /** @brief Input data for training */
        std::vector<double> _training_input;

        /** @brief Output data for given input data */
        std::vector<double> _training_output;

        /** @brief Number of tiles */
        int _n_tiles;

        /** @brief Size of each tile in each dimension */
        int _n_tile_size;

    public:

        /** @brief Lengthscale parameter `l` of squared exponential kernel */
        double lengthscale;

        /**
         * @brief Vertical lengthscale parameter `v` of squared exponential
         * kernel
         */
        double vertical_lengthscale;

        /**
         * @brief Noise variance parameter `sigma` / `n` of squared exponential
         * kernel
         */
        double noise_variance;

        /** @brief Number of regressors */
        int n_regressors;

        /**
         * @brief List of bools indicating trainable parameters: lengthscale,
         * vertical lengthscale, noise variance
         */
        std::vector<bool> trainable_params;

        /**
         * @brief Flag to indicate whether to use GPU for computations
         *
         * May only be enabled on initialization, and is disabled by
         * default.
         */
        bool use_gpu;

        /**
         * @brief Constructs a Gaussian Process (GP)
         *
         * @param input Input data for training of the GP
         * @param output Expected output data for training of the GP
         * @param n_tiles Number of tiles
         * @param n_tile_size Size of each tile in each dimension
         * @param l Lengthscale Parameter of squared exponential kernel: l
         * @param v Vertical Lengthscale parameter of squared exponential
         *     kernel: v
         * @param n Noise Variance parameter of squared exponential kernel: n
         * @param n_regressors Number of regressors
         * @param trainable_bool Vector indicating which parameters are
         *     trainable
         */
        GP(std::vector<double> input,
           std::vector<double> output,
           int n_tiles,
           int n_tile_size,
           double l,
           double v,
           double n,
           int n_regressors,
           std::vector<bool> trainable_bool,
           bool use_gpu);

        /**
         * Returns Gaussian process attributes as string.
         */
        std::string repr() const;

        /**
         * @brief Returns training input data
         */
        std::vector<double> get_training_input() const;

        /**
         * @brief Returns training output data
         */
        std::vector<double> get_training_output() const;

        /**
         * @brief Predict output for test input
         */
        std::vector<double> predict(const std::vector<double>& test_data,
                                    int m_tiles,
                                    int m_tile_size);

        /**
         * @brief Predict output for test input and additionally provide
         * uncertainty for the predictions.
         */
        std::vector<std::vector<double>> predict_with_uncertainty(
            const std::vector<double>& test_data, int m_tiles, int m_tile_size);

        /**
         * @brief Predict output for test input and additionally compute full
         * posterior covariance matrix.
         *
         * @param test_input Test input data
         * @param m_tiles Number of tiles
         * @param m_tile_size Size of each tile
         *
         * @return Full covariance matrix
         */
        std::vector<std::vector<double>> predict_with_full_cov(
            const std::vector<double>& test_data, int m_tiles, int m_tile_size);

        /**
         * @brief Optimize hyperparameters
         *
         * @param hyperparams Hyperparameters of squared exponential kernel:
         *        lengthscale, vertical_lengthscale, noise_variance
         *
         * @return losses
         */
        std::vector<double>
        optimize(const gpxpy_hyper::Hyperparameters& hyperparams);

        /**
         * @brief Perform a single optimization step
         *
         * @param hyperparams Hyperparameters of squared exponential kernel:
         *        lengthscale, vertical_lengthscale, noise_variance
         * @param iter number of iterations
         *
         * @return loss
         */
        double optimize_step(gpxpy_hyper::Hyperparameters& hyperparams,
                             int iter);

        /**
         * @brief Calculate loss for given data and Gaussian process model
         */
        double calculate_loss();

        /**
         * @brief Computes & returns cholesky decomposition
         */
        std::vector<std::vector<double>> cholesky();
    };
} // namespace gpxpy

#endif
