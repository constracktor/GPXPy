#define CALC_TYPE float
#define TYPE "%f"

#include "headers/gp_functions.hpp"
#include "headers/tiled_algorithms_cpu.hpp"
#include "headers/tiled_algorithms_gpu.hpp"

#include <iostream>
#include <hpx/local/init.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
  // determine choleksy variant
  std::string cholesky = vm["cholesky"].as<std::string>();
  // GP parameters
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE    hyperparameters[3];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1;   // noise_variance = small value
  // tiled parameters
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  std::size_t tile_size = n_train / n_tiles;
  std::size_t tile_size_prediction = n_test / n_tiles;
  // HPX structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  //std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> train_output_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
  //hpx::shared_future<std::vector<CALC_TYPE>> ft_alpha;
  hpx::shared_future<std::vector<CALC_TYPE>> ft_prediction;
  hpx::shared_future<CALC_TYPE> ft_error;
  //std::vector<CALC_TYPE> cross_covariance;
  // data holders for assembly
  std::vector<CALC_TYPE>   training_input;
  std::vector<CALC_TYPE>   training_output;
  std::vector<CALC_TYPE>   test_input;
  std::vector<CALC_TYPE>   test_output;
  // data files
  FILE    *training_input_file;
  FILE    *training_output_file;
  FILE    *test_input_file;
  FILE    *test_output_file;
  FILE    *error_file;
  ////////////////////////////////////////////////////////////////////////////
  // initialize GPU stuff
  // get device
  std::size_t device = vm["device"].as<std::size_t>();
  // install cuda future polling handler
  hpx::cuda::experimental::enable_user_polling poll("default");
  // print GPU info
  hpx::cuda::experimental::target target(device);
  std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
       << target.native_handle().processor_name() << "\" "
       << "with compute capability "
       << target.native_handle().processor_family() << "\n";
  // create cublas executor
  hpx::cuda::experimental::cublas_executor cublas(device,
  CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
  ////////////////////////////////////////////////////////////////////////////
  // Load data
  training_input.resize(n_train);
  training_output.resize(n_train);
  test_input.resize(n_test);
  test_output.resize(n_test);
  training_input_file = fopen("../src/data/training/training_input.txt", "r");
  training_output_file = fopen("../src/data/training/training_output.txt", "r");
  test_input_file = fopen("../src/data/test/test_input.txt", "r");
  test_output_file = fopen("../src/data/test/test_output.txt", "r");
  if (training_input_file == NULL || training_output_file == NULL || test_input_file == NULL || test_output_file == NULL)
  {
    printf("Files not found!\n");
    return hpx::local::finalize();    // Handles HPX shutdown
  }
  // load training data
  std::size_t scanned_elements = 0;
  for (int i = 0; i < n_train; i++)
  {
    scanned_elements += fscanf(training_input_file,TYPE,&training_input[i]);
    scanned_elements += fscanf(training_output_file,TYPE,&training_output[i]);
  }
  if (scanned_elements != 2 * n_train)
  {
    printf("Error in reading training data!\n");
    return hpx::local::finalize();    // Handles HPX shutdown
  }
  // load test data
  scanned_elements = 0;
  for (int i = 0; i < n_test; i++)
  {
    scanned_elements += fscanf(test_input_file,TYPE,&test_input[i]);
    scanned_elements += fscanf(test_output_file,TYPE,&test_output[i]);
  }
  if (scanned_elements != 2 * n_test)
  {
    printf("Error in reading test data!\n");
    return hpx::local::finalize();    // Handles HPX shutdown
  }
  // close file streams
  fclose(training_input_file);
  fclose(training_output_file);
  fclose(test_input_file);
  fclose(test_output_file);
  //////////////////////////////////////////////////////////////////////////////
  // ASSEMBLE
  // Assemble covariance matrix vector
  K_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j <= i; j++)
     {
        K_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_covariance<CALC_TYPE>, "assemble_tiled"), i, j, tile_size, n_regressors, hyperparameters, training_input);
     }
  }
  // Assemble alpha
  alpha_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    alpha_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_output<CALC_TYPE>, "assemble_tiled"), i, tile_size, training_output);
  }
  // Assemble transposed covariance matrix vector
  cross_covariance_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; j++)
     {
        cross_covariance_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_cross_covariance<CALC_TYPE>, "assemble_tiled"), i, j, tile_size_prediction, tile_size, n_regressors, hyperparameters, test_input, training_input);
     }
  }
  // Assemble zero prediction
  prediction_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    prediction_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_zeros<CALC_TYPE>, "assemble_tiled"), tile_size_prediction);
  }
  //////////////////////////////////////////////////////////////////////////////
  // CHOLESKY
  // Compute Cholesky decomposition
  // only right-looking currently implemented
  right_looking_cholesky_tiled_cublas(cublas, K_tiles, tile_size, n_tiles);
  //////////////////////////////////////////////////////////////////////////////
  // TRIANGULAR SOLVE
  forward_solve_tiled(K_tiles, alpha_tiles, tile_size, n_tiles);
  backward_solve_tiled(K_tiles, alpha_tiles, tile_size, n_tiles);
  //////////////////////////////////////////////////////////////////////////////
  // PREDICT
  prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, tile_size_prediction, tile_size, n_tiles);
  // assemble and compute error
  ft_prediction = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&assemble<CALC_TYPE>), "prediction_tiled"), prediction_tiles, n_tiles, tile_size_prediction);
  ft_error = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&norm_2<CALC_TYPE>), "prediction_tiled"), ft_prediction, test_output, n_test);
  //////////////////////////////////////////////////////////////////////////////
  // write error to file
  CALC_TYPE average_error = ft_error.get() / n_test;
  error_file = fopen("error.csv", "w");
  std::cout << "average_error: " << average_error << '\n';
  fprintf(error_file, "\"error\",%lf\n", average_error);
  fclose(error_file);
  //////////////////////////////////////////////////////////////////////////////
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
        ("n_tiles", hpx::program_options::value<std::size_t>()->default_value(10),
         "Number of tiles per dimension -> n_tiles * n_tiles total")
        ("cholesky", hpx::program_options::value<std::string>()->default_value("right"),
         "Choose between left- right- or top-looking tiled Cholesky decomposition")
        ("device", hpx::program_options::value<std::size_t>()->default_value(0),
         "Device to use")
    ;
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
