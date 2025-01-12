#include "../install_cpp/include/gpxpy_c.hpp"
#include "../install_cpp/include/utils_c.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <hpx/algorithm.hpp>
#include <iostream>

auto now = std::chrono::high_resolution_clock::now;

// NOTE: my vec, to remove later
std::vector<double> cholesky_times;

int main(int argc, char *argv[])
{
    // number of training points, number of rows/columns in the kernel matrix
    const int N_TRAIN_START = 512;  // 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    const int N_TRAIN_END = 1024;
    const int N_TRAIN_STEP = 512;

    const int N_TEST = 1024;

    const int LOOPS = 8;
    const int OPTIMIZE_ITERATIONS = 1;

    // 2^NUM_CORES_EXPONENT CPU cores are used by HPX
    const std::size_t NUM_CORES_EXPONENT = 2;

    // number of tiles per dimension
    const int n_train_tiles = 1;

    // number of regressors, i.e. number of previous points incl. current point
    // considered for each entry in the kernel matrix
    const int n_reg = 128;

    // path to training input & output data
    std::string train_in_path = "../../../data/training/training_input.txt";
    std::string train_out_path = "../../../data/training/training_output.txt";

    // path to test input data, output data is predicted
    std::string test_in_path = "../../../data/test/test_input.txt";

    for (std::size_t cores = 4; cores <= pow(2, NUM_CORES_EXPONENT); cores *= 2) // NOTE: currently all cores
    {
        // Add number of threads to arguments
        std::vector<std::string> args(argv, argv + argc);
        args.push_back("--hpx:threads=" + std::to_string(cores));

        // Convert arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }
        int hpx_argc = static_cast<int>(cstr_args.size());
        char **hpx_argv = cstr_args.data();

        for (std::size_t n_train = N_TRAIN_START; n_train <= N_TRAIN_END; n_train += N_TRAIN_STEP)
        {
            for (std::size_t loop = 0; loop < LOOPS; loop++)
            {
                // Total time ---------------------------------------------- {{{
                auto start_total = now();

                // Compute tile sizes and number of predict tiles
                int n_train_tile_size = utils::compute_train_tile_size(n_train, n_train_tiles);
                auto [n_test_tiles, n_test_tile_size] = utils::compute_test_tiles(N_TEST, n_train_tiles, n_train_tile_size);

                // Hyperparameters for Adam optimizer
                std::vector<double> M = { 0.0, 0.0, 0.0 };
                gpxpy_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPTIMIZE_ITERATIONS, M };

                // Load data from files
                gpxpy::GP_data training_input(train_in_path, n_train);
                gpxpy::GP_data training_output(train_out_path, n_train);
                gpxpy::GP_data test_input(test_in_path, N_TEST);

                // GP construct time --------------------------------------- {{{
                auto start_init_gp = now();

                std::vector<bool> trainable = { true, true, true };

                // GP for CPU computation
                // std::string target = "cpu";
                // gpxpy::GP gp_cpu(training_input.data, training_output.data, n_train_tiles, n_train_tile_size, 1.0, 1.0, 0.1, n_reg, trainable);

                // GP for GPU computation
                std::string target = "gpu";
                int device = 0;
                int n_streams = 1;
                gpxpy::GP gp_gpu(training_input.data, training_output.data, n_train_tiles, n_train_tile_size, 1.0, 1.0, 0.1, n_reg, trainable, device, n_streams);

                auto init_time = now() - start_init_gp;  // ----------------- }}}

                // Start HPX runtime with arguments
                utils::start_hpx_runtime(hpx_argc, hpx_argv);

                // Cholesky factorization time ----------------------------- {{{
                auto start_cholesky = now();
                auto choleksy_gpu = gp_gpu.cholesky();
                auto cholesky_time = now() - start_cholesky;
                cholesky_times.push_back(cholesky_time.count()); // NOTE: to remove later
                // ------------ }}}

                /*

                // Optimize time (for OPTIMIZE_ITERATIONS) ----------------- {{{
                auto start_opt = now();
                std::vector<double> losses = gp.optimize(hpar);
                auto opt_time = now() - start_opt; // ---------------------- }}}


                // Predict & Uncertainty time  ----------------------------- {{{
                auto start_pred_uncer = now();
                std::vector<std::vector<double>> sum = gp.predict_with_uncertainty(test_input.data, result.first, result.second);
                auto pred_uncer_time = now() - start_pred_uncer; // -------- }}}


                // Predictions with full covariance time ------------------- {{{
                auto start_pred_full_cov = now();
                std::vector<std::vector<double>> full = gp.predict_with_full_cov(test_input.data, result.first, result.second);
                auto pred_full_cov_time = now() - start_pred_full_cov; // -- }}}


                // Predict time -------------------------------------------- {{{
                auto start_pred = now();
                std::vector<double> pred = gp.predict(test_input.data, result.first, result.second);
                auto pred_time = end_pred - start_pred; // ----------------- }}}

                */

                // Stop HPX runtime
                utils::stop_hpx_runtime();

                auto total_time = now() - start_total;  // ------------------ }}}

                // Append parameters & times as CSV
                std::ofstream outfile("../output.csv", std::ios::app);

                // If file is empty, write the header
                if (outfile.tellp() == 0)
                {
                    outfile << "target,cores,n_train,n_test,n_tiles,n_regressor,opt_iter,";
                    outfile << "total_time,init_time,cholesky_time,";
                    // outfile << "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,"
                    outfile << "n_loop\n";
                }
                outfile << target << ","
                        << cores << ","
                        << n_train << ","
                        << N_TEST << ","
                        << n_train_tiles << ","
                        << n_reg << ","
                        << OPTIMIZE_ITERATIONS << ","
                        << total_time.count() << ","
                        << init_time.count() << ","
                        << cholesky_time.count() << ","
                        // << opt_time.count() << "," << pred_uncer_time.count() << "," << pred_full_cov_time.count() << "," << pred_time.count() << ","
                        << loop << "\n";
                outfile.close();
            }
        }
    }

    for (auto &time : cholesky_times) // NOTE: to remove later
    {
        std::cout << time << ",";
    }
    std::cout << std::endl;

    return 0;
}
