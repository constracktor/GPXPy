#include <gaussian_process>

#include <iostream>

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    int n_train = 1000;
    int n_test = 500;
    const int N_CORES = 1; // Set this to the number of threads
    // const int tile_size = 100;
    const int n_tiles = 1;
    const int n_reg = 100;
    const int opt_iter = 1;

    // int n_tiles = utils::compute_train_tiles(n_train, tile_size);
    // std::cout << "n_tiles: " << n_tiles << std::endl;
    int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
    std::cout << "n_tile_size: " << tile_size << std::endl;
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
    ///// hyperparams
    std::vector<double> M = {0.0, 0.0, 0.0};
    gpppy_hyper::Hyperparameters hpar = {0.1, 0.9, 0.999, 1e-8, opt_iter, M};
    std::cout << "lr: " << hpar.learning_rate << std::endl;
    std::cout << hpar.repr() << std::endl;

    /////////////////////
    ////// data loading
    std::string train_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/interim_data/training_input.txt";
    gpppy::GP_data training_input(train_path, n_train);
    std::cout << "training input" << std::endl;
    utils::print(training_input.data, 0, 10, ", ");

    std::string out_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/interim_data/training_output.txt";
    gpppy::GP_data training_output(out_path, n_train);
    // std::cout << "training output" << std::endl;
    // utils::print(training_output.data, 0, 10, ", ");

    std::string test_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt";
    gpppy::GP_data test_input(test_path, n_test);
    // utils::print(test_input.data, 0, 10, ", ");

    /////////////////////
    ///// GP
    std::vector<bool> trainable = {true, true, true};
    gpppy::GP gp(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable);
    // std::vector<double> training_data = gp.get_training_input();
    // std::cout << "training input" << std::endl;
    // utils::print(training_data, 0, 5, ", ");

    std::cout << gp.repr() << std::endl;

    // Initialize HPX with the new arguments, don't run hpx_main
    utils::start_hpx_runtime(new_argc, new_argv);
    //double init_loss;
    //init_loss = gp.calculate_loss();
    //std::cout << "init loss: " << init_loss << std::endl;

    // std::size_t iter = 3;
    // for (std::size_t i; i < iter; i++)
    // {
    //     std::cout << "loss:" << gp.optimize_step(hpar, i) << std::endl;
    // }

    std::vector<double> losses;
    losses = gp.optimize(hpar);
    std::cout << "Loss" << std::endl;
    utils::print(losses);
    utils::suspend_hpx_runtime();

    utils::resume_hpx_runtime();
    std::vector<std::vector<double>> pred_and_uncert;
    pred_and_uncert = gp.predict_with_uncertainty(test_input.data, result.first, result.second);
    std::cout << "Prediction" << std::endl;
    utils::print(pred_and_uncert[0], 0, 10, ", ");
    std::cout << "Uncertainty" << std::endl;
    utils::print(pred_and_uncert[1], 0, 10, ", ");

    std::vector<double> pred;
    pred = gp.predict(test_input.data, result.first, result.second);
    std::cout << "Prediction" << std::endl;
    utils::print(pred, 0, 10, ", ");

    // Stop the HPX runtime
    utils::stop_hpx_runtime();

    std::cout << gp.repr() << std::endl;

    std::cout << hpar.repr() << std::endl;
    std::cout << "Ende" << std::endl;

    return 0;
}
