// #define CALC_TYPE double
// #define TYPE "%lf"
#define CALC_TYPE float
#define TYPE "%f"

#include "headers/gp_functions.hpp"
#include "headers/gp_functions_grad.hpp"
#include "headers/tiled_algorithms_cpu.hpp"

#include <iostream>
#include <hpx/init.hpp>

#include <fstream>
#include <iterator>

int hpx_main(hpx::program_options::variables_map &vm)
{
  // declare data structures
  // tiled future data structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_v_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_l_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prior_K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> y_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> t_cross_covariance_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_uncertainty_tiles;
  // data holders for adam
  std::vector<hpx::shared_future<CALC_TYPE>> m_T;
  std::vector<hpx::shared_future<CALC_TYPE>> v_T;
  // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> beta1_T;
  std::vector<hpx::shared_future<CALC_TYPE>> beta1_T;
  std::vector<hpx::shared_future<CALC_TYPE>> beta2_T;
  // future data structures
  hpx::shared_future<CALC_TYPE> loss_value;
  hpx::shared_future<CALC_TYPE> ft_error;
  // data holders for assembly
  std::vector<CALC_TYPE> training_input;
  std::vector<CALC_TYPE> training_output;
  std::vector<CALC_TYPE> test_input;
  std::vector<CALC_TYPE> test_output;
  // data files
  FILE *training_input_file;
  FILE *training_output_file;
  FILE *test_input_file;
  FILE *test_output_file;
  FILE *error_file;
  //////////////////////////////////////////////////////////////////////////////
  // Get and set parameters
  // determine choleksy variant and problem size
  std::string cholesky = vm["cholesky"].as<std::string>();
  std::size_t n_train = vm["n_train"].as<std::size_t>(); // max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();   // max 5*1000
  // GP parameters
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE hyperparameters[6];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1;   // noise_variance = small value
  hyperparameters[3] = 0.001; // learning rate
  hyperparameters[4] = 0.9;   // beta1
  hyperparameters[5] = 0.999; // beta2
  hyperparameters[6] = 1e-8;  // epsilon
  int opt_iter = 10;           // # optimisation steps --> move to vm["iter"]
  // tile parameters
  std::size_t n_tile_size = vm["tile_size"].as<std::size_t>();
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  if (n_tile_size == 0 && n_tiles > 0)
  {
    n_tile_size = n_train / n_tiles;
  }
  else if (n_tile_size > 0 && n_tiles == 0)
  {
    n_tiles = n_train / n_tile_size;
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
    scanned_elements += fscanf(training_input_file, TYPE, &training_input[i]);
    scanned_elements += fscanf(training_output_file, TYPE, &training_output[i]);
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
    scanned_elements += fscanf(test_input_file, TYPE, &test_input[i]);
    scanned_elements += fscanf(test_output_file, TYPE, &test_output[i]);
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

  for (int iter = 0; iter < opt_iter; iter++)
  {
    //////////////////////////////////////////////////////////////////////////////
    // PART 1: ASSEMBLE
    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j <= i; j++)
      {
        K_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_covariance<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size, n_regressors, hyperparameters, training_input);
      }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to vertical lengthscale
    grad_v_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        grad_v_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_grad_v<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size, n_regressors, hyperparameters, training_input);
      }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to lengthscale
    grad_l_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        grad_l_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_grad_l<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size, n_regressors, hyperparameters, training_input);
      }
    }
    // Assemble matrix that will be multiplied with derivates
    grad_K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        grad_K_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_identity<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size);
      }
    }
    // Assemble first and second momemnt vectors: m_T and v_T
    m_T.resize(3);
    v_T.resize(3);
    for (int i = 0; i < 3; i++)
    {
      m_T[i] = hpx::async(hpx::annotated_function(&gen_zero<CALC_TYPE>, "assemble_tiled"));
      v_T[i] = hpx::async(hpx::annotated_function(&gen_zero<CALC_TYPE>, "assemble_tiled"));
    }
    // std::ofstream k_file("./covariance.txt");
    // std::ostream_iterator<CALC_TYPE> k_iterator(k_file, "\n");
    // for (std::size_t i = 0; i < n_tiles; i++)
    // {
    //   for (std::size_t j = 0; j < n_tiles; j++)
    //   {
    //     std::vector<float> k_ = grad_K_tiles[i * n_tiles + j].get(); // Get the vector from the shared_future
    //     std::copy(k_.begin(), k_.end(), k_iterator);
    //     // std::vector<float> nuller = {0.0, 0.0, 0.0}; // Get the vector from the shared_future
    //     // std::copy(nuller.begin(), nuller.end(), k_iterator);
    //   }
    // }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      alpha_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output<CALC_TYPE>, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      y_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output<CALC_TYPE>, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble prior covariance matrix vector
    // prior_K_tiles.resize(m_tiles * m_tiles);
    // for (std::size_t i = 0; i < m_tiles; i++)
    // {
    //   prior_K_tiles[i * m_tiles + i] = hpx::async(hpx::annotated_function(&gen_tile_prior_covariance<CALC_TYPE>, "assemble_tiled"), i, i, m_tile_size, n_regressors, hyperparameters, test_input);
    // }
    // Assemble MxN cross-covariance matrix vector
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        cross_covariance_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance<CALC_TYPE>, "assemble_tiled"), i, j, m_tile_size, n_tile_size, n_regressors, hyperparameters, test_input, training_input);
      }
    }
    // Assemble NxM (transpose) cross-covariance matrix vector
    // t_cross_covariance_tiles.resize(n_tiles * m_tiles);
    // for (std::size_t i = 0; i < n_tiles; i++)
    // {
    //   for (std::size_t j = 0; j < m_tiles; j++)
    //   {
    //     t_cross_covariance_tiles[i * m_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance<CALC_TYPE>, "assemble_tiled"), i, j, n_tile_size, m_tile_size, n_regressors, hyperparameters, training_input, test_input);
    //   }
    // }
    // Assemble zero prediction
    prediction_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
      prediction_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros<CALC_TYPE>, "assemble_tiled"), m_tile_size);
    }
    // Assemble zero prediction
    // prediction_uncertainty_tiles.resize(m_tiles);
    // for (std::size_t i = 0; i < m_tiles; i++)
    // {
    //   prediction_uncertainty_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros<CALC_TYPE>, "assemble_tiled"), m_tile_size);
    // }
    // Assemble beta1_t and beta2_t

    beta1_T.resize(opt_iter);
    for (int i = 0; i < opt_iter; i++)
    {
      beta1_T[i] = hpx::async(hpx::annotated_function(&gen_beta_t<CALC_TYPE>, "assemble_tiled"), i + 1, hyperparameters, 4);
    }
    beta2_T.resize(opt_iter);
    for (int i = 0; i < opt_iter; i++)
    {
      beta2_T[i] = hpx::async(hpx::annotated_function(&gen_beta_t<CALC_TYPE>, "assemble_tiled"), i + 1, hyperparameters, 5);
    }

    // // Accessing and printing the result
    // for (std::size_t k = 0; k < beta1_T.size(); ++k)
    // {
    //   CALC_TYPE result = beta1_T[k].get(); // Correctly get the CALC_TYPE (float) from the shared_future
    //   printf("beta1 res: %lf\n", result);
    // }

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
      // right_looking_cholesky_tiled(K_tiles, n_tile_size, n_tiles);
      right_looking_cholesky_tiled_mkl(K_tiles, n_tile_size, n_tiles);
    }
    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Compute loss
    compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
    printf("iter: %d param: %lf loss: %lf\n", iter, hyperparameters[0], loss_value.get());
    update_grad_K_tiled_mkl(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size, n_tiles);

    forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
    backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // update_hyperparameter(grad_K_tiles, grad_l_tiles, hyperparameters, n_tile_size, n_tiles, 0);
    update_hyperparameter(grad_K_tiles, grad_l_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter, 0);
    // update_hyperparameter(grad_K_tiles, grad_v_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter, 1);
    // update_noise_variance(grad_K_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);

    // printf("lengthscale: %lf\n", hyperparameters[1]);
    std::ofstream myfile("hyper.txt");
    for (int count = 0; count < 3; count++)
    {
      myfile << hyperparameters[count] << " ";
    }
    myfile << "\n";
    myfile.close();

  }
  // Accessing and printing the result
  // for (std::size_t k = 0; k < m_T.size(); ++k)
  // {
    // CALC_TYPE result = m_T[k].get(); // Correctly get the CALC_TYPE (float) from the shared_future
    // printf("beta1 res: %lf\n", result);
  // }

  // Triangular solve K_N,N * A_NxM = K_NxM -> A_NxM = K^-1_NxN * K_NxM
  // forward_solve_tiled_matrix(K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
  // backward_solve_tiled_matrix(K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
  //////////////////////////////////////////////////////////////////////////////
  // PART 3: PREDICTION
  prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
  // posterior covariance matrix
  // posterior_covariance_tiled(cross_covariance_tiles, t_cross_covariance_tiles, prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
  // predicition uncertainty
  // prediction_uncertainty_tiled(prior_K_tiles, prediction_uncertainty_tiles, m_tile_size, m_tiles);
  //  compute error
  ft_error = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_error_norm<CALC_TYPE>), "prediction_tiled"), m_tiles, m_tile_size, test_output, prediction_tiles);
  ////////////////////////////////////////////////////////////////////////////
  // write error to file
  // std::ofstream output_file("./predictions.txt");
  // for (std::size_t k = 0; k < m_tiles; k++)
  // {
  //   std::ostream_iterator<CALC_TYPE> output_iterator(output_file, "\n");
  //   std::vector<float> result = prediction_tiles[k].get(); // Get the vector from the shared_future
  //   std::copy(result.begin(), result.end(), output_iterator);
  // }

  // std::ofstream myfile("hyper.txt");
  // for (int count = 0; count < 3; count++)
  // {
  //   myfile << hyperparameters[count] << " ";
  // }
  // myfile.close();

  // FILE *loss_file;
  // CALC_TYPE l = loss_value.get() / 1;
  // loss_file = fopen("./loss_f.txt", "w");
  // fprintf(loss_file, "\"loss\",%lf\n", l);
  // fclose(loss_file);

  printf("loss: %lf\n", loss_value.get());
  CALC_TYPE average_error = ft_error.get() / n_test;
  CALC_TYPE average_ = loss_value.get();
  error_file = fopen("error.csv", "w");
  fprintf(error_file, "\"error\",%lf\n", average_error);
  fprintf(error_file, "%lf\n", average_);
  fclose(error_file);
  return hpx::local::finalize(); // Handles HPX shutdown
}

int main(int argc, char *argv[])
{
  hpx::program_options::options_description desc_commandline;
  hpx::local::init_params init_args;
  // Setup input arguments
  desc_commandline.add_options()("n_train", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
                                 "Number of training samples (max 100 000)")("n_test", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
                                                                             "Number of test samples (max 5 000)")("n_regressors", hpx::program_options::value<std::size_t>()->default_value(100),
                                                                                                                   "Number of delayed input regressors")("n_tiles", hpx::program_options::value<std::size_t>()->default_value(0),
                                                                                                                                                         "Number of tiles per dimension -> n_tiles * n_tiles total")("tile_size", hpx::program_options::value<std::size_t>()->default_value(0),
                                                                                                                                                                                                                     "Tile size per dimension -> tile_size * tile_size total entries")("cholesky", hpx::program_options::value<std::string>()->default_value("right"),
                                                                                                                                                                                                                                                                                       "Choose between right- left- or top-looking tiled Cholesky decomposition");
  // Run HPX main
  init_args.desc_cmdline = desc_commandline;
  return hpx::local::init(hpx_main, argc, argv, init_args);
}
