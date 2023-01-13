// #define CALC_TYPE double
// #define TYPE "%lf"
#define CALC_TYPE float
#define TYPE "%f"

#include "headers/gp_functions.hpp"
#include "headers/tiled_algorithms_cpu.hpp"

#include <iostream>
#include <hpx/local/init.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
  // declare data structures
  // tiled future data structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
  // future data structures
  hpx::shared_future<CALC_TYPE> ft_error;
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
  //////////////////////////////////////////////////////////////////////////////
  // Get and set parameters
  // determine choleksy variant and problem size
  std::string cholesky = vm["cholesky"].as<std::string>();
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  // GP parameters
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE    hyperparameters[3];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1;   // noise_variance = small value
  // tile parameters
  std::size_t n_tile_size = vm["tile_size"].as<std::size_t>();
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  if (n_tile_size == 0 && n_tiles > 0)
  {
    n_tile_size = n_train / n_tiles;
  }
  else if (n_tile_size > 0 && n_tiles == 0)
  {
    n_tiles =  n_train / n_tile_size;
  }
  else
  {
    printf("Error: Please specify either a valid value for n_tile_size or n_tiles.\n");
    return hpx::local::finalize();
  }
  // set different tile size for prediction if n_test not dividable
  std::size_t m_tiles;
  std::size_t m_tile_size;
  if ((n_test % n_tile_size) > 0)
  {
    m_tiles = n_tiles;
    m_tile_size = n_test / m_tiles;
  }
  else
  {
    m_tiles = n_test / n_tile_size;
    m_tile_size = n_tile_size;
  }
  // output information about the tiles
  printf("N: %zu\n", n_train);
  printf("M: %zu\n", n_test);
  printf("Tile size in N dimension: %zu\n", n_tile_size);
  printf("Tile size in M dimension: %zu\n", m_tile_size);
  printf("Tiles in N dimension: %zu\n", n_tiles);
  printf("Tiles in M dimension: %zu\n", m_tiles);
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
    printf("Error: Files not found.\n");
    return hpx::local::finalize();
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
    printf("Error: Training data not correctly read.\n");
    return hpx::local::finalize();
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
    printf("Error: Test data not correctly read.\n");
    return hpx::local::finalize();
  }
  // close file streams
  fclose(training_input_file);
  fclose(training_output_file);
  fclose(test_input_file);
  fclose(test_output_file);
  //////////////////////////////////////////////////////////////////////////////
  // PART 1: ASSEMBLE
  // Assemble covariance matrix vector
  K_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j <= i; j++)
     {
        K_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_covariance<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size, n_regressors, hyperparameters, training_input);
     }
  }
  // Assemble alpha
  alpha_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    alpha_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_output<CALC_TYPE>, "assemble_tiled"), i, n_tile_size, training_output);
  }
  // Assemble transposed cross-covariance matrix vector
  cross_covariance_tiles.resize(m_tiles * n_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; j++)
     {
        cross_covariance_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(&gen_tile_cross_covariance<CALC_TYPE>, "assemble_tiled"), i, j, m_tile_size, n_tile_size, n_regressors, hyperparameters, test_input, training_input);
     }
  }
  // Assemble zero prediction
  prediction_tiles.resize(m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    prediction_tiles[i] = hpx::dataflow(hpx::annotated_function(&gen_tile_zeros<CALC_TYPE>, "assemble_tiled"), m_tile_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // PART 2: CHOLESKY SOLVE
  // Cholesky decomposition
  if (cholesky.compare("left") == 0)
  {
    left_looking_cholesky_tiled(K_tiles, n_tile_size, n_tiles);
  }
  else if (cholesky.compare("top") == 0)
  {
    top_looking_cholesky_tiled(K_tiles, n_tile_size, n_tiles);
  }
  else // set to "right" per default
  {
    right_looking_cholesky_tiled(K_tiles, n_tile_size, n_tiles);
  }
  // Triangular solve
  forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);
  backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);
  //////////////////////////////////////////////////////////////////////////////
  // PART 3: PREDICTION
  prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
  // compute error
  ft_error = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_error_norm<CALC_TYPE>), "prediction_tiled"), prediction_tiles, test_output, m_tiles, m_tile_size);
  //////////////////////////////////////////////////////////////////////////////
  // write error to file
  CALC_TYPE average_error = ft_error.get() / n_test;
  error_file = fopen("error.csv", "w");
  fprintf(error_file, "\"error\",%lf\n", average_error);
  fclose(error_file);
  return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline;
    hpx::local::init_params init_args;
    // Setup input arguments
    desc_commandline.add_options()
        ("n_train", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of training samples (max 100 000)")
        ("n_test", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of test samples (max 5 000)")
        ("n_regressors", hpx::program_options::value<std::size_t>()->default_value(100),
        "Number of delayed input regressors")
        ("n_tiles", hpx::program_options::value<std::size_t>()->default_value(0),
         "Number of tiles per dimension -> n_tiles * n_tiles total")
        ("tile_size", hpx::program_options::value<std::size_t>()->default_value(0),
         "Tile size per dimension -> tile_size * tile_size total entries")
        ("cholesky", hpx::program_options::value<std::string>()->default_value("right"),
         "Choose between right- left- or top-looking tiled Cholesky decomposition")
    ;
    // Run HPX main
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
