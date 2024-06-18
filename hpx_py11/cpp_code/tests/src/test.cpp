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
    int n_train = 300;
    int n_test = 700;
    const int N_CORES = 2; // Set this to the number of threads
    const int tile_size = 100;
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

    gpppy_hyper::Hyperparameters hpar = {0.1, 0.9, 0.999, 1e-8, 2};
    std::cout << "lr: " << hpar.learning_rate << std::endl;
    std::cout << hpar.repr() << std::endl;

    /////////////////////
    ////// data loading
    std::string train_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt";
    gpppy::GP_data training_input(train_path, n_train);
    std::cout << "training input" << std::endl;
    utils::print(training_input.data, 0, 10, ", ");

    std::string out_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt";
    gpppy::GP_data training_output(out_path, n_train);
    std::cout << "training output" << std::endl;
    utils::print(training_output.data, 0, 10, ", ");

    std::string test_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt";
    gpppy::GP_data test_input(test_path, n_test);
    utils::print(test_input.data, 0, 10, ", ");

    /////////////////////
    ///// GP
    std::vector<bool> trainable = {false, false, true};
    gpppy::GP gp(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable);
    std::vector<double> training_data = gp.get_training_input();
    std::cout << "training input" << std::endl;
    utils::print(training_data, 0, 5, ", ");

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

    std::vector<double> losses;
    losses = gp.optimize(hpar);
    std::cout << "Loss" << std::endl;
    utils::print(losses, 0, 5);

    std::vector<std::vector<double>> sum;
    sum = gp.predict(test_input.data, result.first, result.second);
    std::cout << "Prediction" << std::endl;
    utils::print(sum[0], 0, 10, ", ");
    std::cout << "Uncertainty" << std::endl;
    utils::print(sum[1], 0, 10, ", ");
    // hpx::stop();
    utils::stop_hpx_runtime();

    std::cout << gp.repr() << std::endl;

    std::cout << hpar.repr() << std::endl;
    std::cout << "Ende" << std::endl;

    return 0;
}
