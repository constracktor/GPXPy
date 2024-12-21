#include <chrono>
#include <fstream>
#include <iostream>
// #include <boost/program_options.hpp>
#include "../install_cpp/include/gpxpy_c.hpp"
#include "../install_cpp/include/utils_c.hpp"
#include <cmath>
#include <hpx/algorithm.hpp>

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    int START = 1024;  // 8192; // number of training points (also number of rows/columns in the kernel matrix)
    int END = 1024;    // 8192;
    int STEP = 128;
    int LOOP = 1;
    const int OPT_ITER = 1;

    int n_test = 1024;
    const std::size_t N_CORES = 2;  // Set this to the number of threads, maybe higher previously
    const int n_tiles = 32;          // 32; // number of tiles per dimension
    const int n_reg = 128;

    std::string train_path = "../../../data/training/training_input.txt";
    std::string out_path = "../../../data/training/training_output.txt";
    std::string test_path = "../../../data/test/test_input.txt";

    for (std::size_t core = 2; core <= pow(2, N_CORES); core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        for (std::size_t start = START; start <= END; start = start + STEP)
        {
            int n_train = start;
            for (std::size_t l = 0; l < LOOP; l++)
            {
                auto start_total = std::chrono::high_resolution_clock::now();

                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                std::vector<double> M = { 0.0, 0.0, 0.0 };
                gpxpy_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER, M };

                /////////////////////
                ////// data loading
                gpxpy::GP_data training_input(train_path, n_train);
                gpxpy::GP_data training_output(out_path, n_train);
                gpxpy::GP_data test_input(test_path, n_test);

                /////////////////////
                ///// GP
                auto start_init = std::chrono::high_resolution_clock::now();
                std::vector<bool> trainable = { false, false, true };
                gpxpy::GP gp_cpu(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable);
                int device = 0;
                int n_streams = 1;
                gpxpy::GP gp_gpu(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable, device, n_streams);
                auto end_init = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> init_time = end_init - start_init;

                // Initialize HPX with the new arguments, don't run hpx_main
                utils::start_hpx_runtime(new_argc, new_argv);

                // Measure the time taken to execute gp.cholesky();
                auto start_cholesky = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> choleksy_cpu = gp_cpu.cholesky();
                std::vector<std::vector<double>> choleksy_gpu = gp_gpu.cholesky();

                auto end_cholesky = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> cholesky_time = end_cholesky - end_cholesky;

                /* // Measure the time taken to execute gp.optimize(hpar);
                auto start_opt = std::chrono::high_resolution_clock::now();
                // std::vector<double> losses = gp.optimize(hpar);
                auto end_opt = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> opt_time = end_opt - start_opt;

                auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> sum = gp.predict_with_uncertainty(test_input.data, result.first, result.second);
                auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_uncer_time = end_pred_uncer - start_pred_uncer;

                auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> full = gp.predict_with_full_cov(test_input.data, result.first, result.second);
                auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                auto start_pred = std::chrono::high_resolution_clock::now();
                std::vector<double> pred = gp.predict(test_input.data, result.first, result.second);
                auto end_pred = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_time = end_pred - start_pred; */

                // Stop the HPX runtime
                utils::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("../output.csv", std::ios::app);  // Append mode
                if (outfile.tellp() == 0)
                {
                    // If file is empty, write the header
                    outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_time,Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
                }
                outfile << core << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg << ","
                        << OPT_ITER << "," << total_time.count() << "," << init_time.count() << "," << cholesky_time.count() << ","
                        // << opt_time.count() << "," << pred_uncer_time.count() << "," << pred_full_cov_time.count() << "," << pred_time.count()
                        << "," << l << "\n";
                outfile.close();
            }
        }
    }
    return 0;
}
