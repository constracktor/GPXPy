#define CALC_TYPE double
#define TYPE "%lf"

#include <hpx/local/chrono.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/format.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
    //std::size_t N = vm["N"].as<std::size_t>();
    //std::size_t T = vm["T"].as<std::size_t>();
    //std::string out = vm["out"].as<std::string>();

    hpx::chrono::high_resolution_timer t;
    /*
    std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> A_tiles;
    A_tiles.resize(T * T);

    for (std::size_t i = 0; i < T; ++i)
    {
       for (std::size_t j = 0; j < T; ++j)
       {
          A_tiles[i * T + j] = hpx::dataflow(&gen_tile, i, j, N, T);
       }
    }

    if (out == "debug")
    {
       for (std::size_t i = 0; i < T; ++i)
       {
          for (std::size_t j = 0; j < T; ++j)
          {
             print_tile(A_tiles[i * T + j].get(), "a", i, j, N);
          }
       }
    }

    lu_tiled(A_tiles, N, T);

    if (out == "debug")
    {
       for (std::size_t i = 0; i < T; ++i)
       {
          for (std::size_t j = 0; j < T; ++j)
          {
             print_tile(A_tiles[i * T + j].get(), "lu", i, j, N);
          }
       }
    }
    */
    double elapsed = t.elapsed();
    std::cout << "Elapsed " << elapsed << " s\n";
    return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
  {
    // hpx
    hpx::program_options::options_description desc_commandline;
    hpx::local::init_params init_args;
    ////////////////////////////////////////////////////////////////////////////
    // Setup input arguments
    desc_commandline.add_options()
        ("n_train", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of training samples (max 100 000)")
        ("n_test", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of test samples (max 5 000)")
        ("n_regressors", hpx::program_options::value<std::size_t>()->default_value(100),
        "Number of delayed input regressors")
    ;
    hpx::program_options::variables_map vm;
    hpx::program_options::store(hpx::program_options::parse_command_line(argc, argv, desc_commandline), vm);
    // GP parameters
    int       n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
    int       n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
    int       n_regressors = vm["n_regressors"].as<std::size_t>();
    CALC_TYPE    hyperparameters[3];
    // initalize hyperparameters to empirical moments of the data
    hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
    hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
    hyperparameters[2] = 0.001; // noise_variance = small value

    // data holders for assembly
    CALC_TYPE  training_input[n_train];
    CALC_TYPE   training_output[n_train];
    CALC_TYPE   test_input[n_test];
    CALC_TYPE   test_output[n_test];
    // data files
    FILE    *training_input_file;
    FILE    *training_output_file;
    FILE    *test_input_file;
    FILE    *test_output_file;

    ////////////////////////////////////////////////////////////////////////////
    // Load data
    training_input_file = fopen("../src/data/training/training_input.txt", "r");
    training_output_file = fopen("../src/data/training/training_output.txt", "r");
    test_input_file = fopen("../src/data/test/test_input_3.txt", "r");
    test_output_file = fopen("../src/data/test/test_output_3.txt", "r");
    if (training_input_file == NULL || training_output_file == NULL || test_input_file == NULL || test_output_file == NULL)
    {
      printf("return 1\n");
      return 1;
    }
    // load training data
    for (int i = 0; i < n_train; i++)
    {
      fscanf(training_input_file,TYPE,&training_input[i]);
      fscanf(training_output_file,TYPE,&training_output[i]);
    }
    // load test data
    for (int i = 0; i < n_test; i++)
    {
      fscanf(test_input_file,TYPE,&test_input[i]);
      fscanf(test_output_file,TYPE,&test_output[i]);
    }
    // close file streams
    fclose(training_input_file);
    fclose(training_output_file);
    fclose(test_input_file);
    fclose(test_output_file);
    printf("train %d, test %d, reg %d", n_train, n_test, n_regressors);
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
