#include <automobile>

// #include <hpx/include/run_as.hpp>
// #include <hpx/hpx_start.hpp>
// #include <hpx/future.hpp>
// #include <hpx/include/post.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    int n_train = 10;
    int n_test = 10;
    const int N_CORES = 2; // Set this to the number of threads
    const int tile_size = 10;
    const int n_reg = 100;

    int n_tiles = utils::compute_train_tiles(n_train, tile_size);
    std::cout << "n_tiles: " << n_tiles << std::endl;
    auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
    std::cout << "m_tiles: " << result.first << ", m_tile_size: " << result.second << std::endl;

    // Create new argc and argv to include the --hpx:threads argument
    std::vector<std::string> args(argv, argv + argc);
    args.push_back("--hpx:threads=" + std::to_string(N_CORES));

    // Convert the arguments to char* array
    std::vector<char *> cstr_args;
    for (auto &arg : args)
    {
        cstr_args.push_back(const_cast<char *>(arg.c_str()));
    }

    int new_argc = static_cast<int>(cstr_args.size());
    char **new_argv = cstr_args.data();

    /////////////////////
    ///// params
    gpppy::Kernel_Params kpar;
    std::cout << "lengthscale: " << kpar.lengthscale << std::endl;
    std::cout << kpar.repr() << std::endl;

    gpppy::Hyperparameters hpar;
    std::cout << "lr: " << hpar.learning_rate << std::endl;
    std::cout << hpar.repr() << std::endl;

    /////////////////////
    ////// data loading
    std::string file_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt";
    gpppy::GP_data training_input(file_path, n_train);
    utils::print(training_input.data, 0, 10, ", ");

    std::string out_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt";
    gpppy::GP_data training_output(out_path, n_train);
    utils::print(training_output.data, 0, 10, ", ");

    std::string test_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt";
    gpppy::GP_data test_input(test_path, n_test);
    utils::print(test_input.data, 0, 10, ", ");

    /////////////////////
    ///// GP
    gpppy::GP gp(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg);
    std::vector<double> training_data = gp.get_training_output();
    utils::print(training_data, 0, 2, ", ");

    std::cout << gp.repr() << std::endl;

    // Initialize HPX with the new arguments
    utils::start_hpx_runtime(new_argc, new_argv);

    // // Initialize HPX, don't run hpx_main
    // // hpx::start(nullptr, argc, argv);

    // Schedule some functions on the HPX runtime
    // NOTE: run_as_hpx_thread blocks until completion.
    // hpx::threads::run_as_hpx_thread(&m.do_fut);
    // hpx::threads::run_as_hpx_thread([&s]()
    //                                 { s.do_fut(); });

    // hpx::finalize has to be called from the HPX runtime before hpx::stop
    // hpx::post([]()
    //           { hpx::finalize(); });

    std::vector<std::vector<double>> sum;
    sum = gp.predict(test_input.data, result.first, result.second);
    utils::print(sum[0], 0, 10, ", ");
    utils::print(sum[1], 0, 10, ", ");
    // hpx::stop();
    utils::stop_hpx_runtime();

    std::cout << gp.repr() << std::endl;

    std::cout << "Ende" << std::endl;

    return 0;
}
