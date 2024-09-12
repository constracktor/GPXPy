#define CALC_TYPE double
#define TYPE "%lf"
// #define CALC_TYPE float
// #define TYPE "%f"

#include "headers/gp_functions.hpp"
#include "headers/gp_functions_grad.hpp"
#include "headers/tiled_algorithms_cpu.hpp"

#include <iostream>
#include <iomanip>
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
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_I_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prior_K_tiles;
  std::vector<hpx::shared_future<std::vector<double>>> prior_inter_tiles;
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
  CALC_TYPE hyperparameters[7];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1;   // noise_variance = small value
  hyperparameters[3] = 0.1;   // learning rate
  hyperparameters[4] = 0.9;   // beta1
  hyperparameters[5] = 0.999; // beta2
  hyperparameters[6] = 1e-8;  // epsilon
  int opt_iter = 1;           // # optimisation steps --> move to vm["iter"]
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

  //////////////////////////////////////////////////////////////////////////////
  // Assemble beta1_t and beta2_t
  beta1_T.resize(opt_iter);
  for (int i = 0; i < opt_iter; i++)
  {
    beta1_T[i] = hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, hyperparameters, 4);
  }
  beta2_T.resize(opt_iter);
  for (int i = 0; i < opt_iter; i++)
  {
    beta2_T[i] = hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, hyperparameters, 5);
  }
  // Assemble first and second momemnt vectors: m_T and v_T
  m_T.resize(3);
  v_T.resize(3);
  for (int i = 0; i < 3; i++)
  {
    m_T[i] = hpx::async(hpx::annotated_function(&gen_zero, "assemble_tiled"));
    v_T[i] = hpx::async(hpx::annotated_function(&gen_zero, "assemble_tiled"));
  }
  // Assemble y
  y_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    y_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output, "assemble_y"), i, n_tile_size, training_output);
  }

  for (int iter = 0; iter < opt_iter; iter++)
  {
    //////////////////////////////////////////////////////////////////////////////
    // Assemble covariance matrix vector, derivative of covariance matrix vector w.r.t. to vertical lengthscale
    // and derivative of covariance matrix vector w.r.t. to lengthscale
    K_tiles.resize(n_tiles * n_tiles);
    grad_v_tiles.resize(n_tiles * n_tiles);
    grad_l_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j <= i; j++)
      {
        hpx::shared_future<std::vector<double>> cov_dists = hpx::async(hpx::annotated_function(&compute_cov_dist_vec, "assemble_cov_dist"), i, j,
                                                                       n_tile_size, n_regressors, hyperparameters, training_input);

        K_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_covariance_opt), "assemble_K"), i, j,
                                                 n_tile_size, n_regressors, hyperparameters, cov_dists);

        grad_v_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v), "assemble_gradv"), i, j,
                                                      n_tile_size, n_regressors, hyperparameters, cov_dists);

        grad_l_tiles[i * n_tiles + j] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l), "assemble_gradl"), i, j,
                                                      n_tile_size, n_regressors, hyperparameters, cov_dists);

        if (i != j)
        {
          grad_v_tiles[j * n_tiles + i] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v_trans), "assemble_gradv_t"),
                                                        n_tile_size, grad_v_tiles[i * n_tiles + j]);

          grad_l_tiles[j * n_tiles + i] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l_trans), "assemble_gradl_t"),
                                                        n_tile_size, grad_l_tiles[i * n_tiles + j]);
        }
      }
    }
    // Assemble placeholder matrix for K^-1 * (I - y*y^T*K^-1)
    grad_K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        grad_K_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_identity, "assemble_tiled"), i, j, n_tile_size);
      }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      alpha_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), n_tile_size);
    }
    // Assemble placeholder matrix for K^-1
    grad_I_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
      for (std::size_t j = 0; j < n_tiles; j++)
      {
        grad_I_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_identity, "assemble_identity_matrix"), i, j, n_tile_size);
      }
    }

    //////////////////////////////////////////////////////////////////////////////
    // PART 2: CHOLESKY SOLVE
    // Cholesky decomposition
    right_looking_cholesky_tiled_mkl(K_tiles, n_tile_size, n_tiles);

    // Compute K^-1 through L*L^T*X = I
    forward_solve_tiled_matrix(K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
    backward_solve_tiled_matrix(K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // Triangular solve K_NxN * alpha = y
    // forward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size, n_tiles);
    // backward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size, n_tiles);

    // inv(K)*y
    compute_gemm_of_invK_y(grad_I_tiles, y_tiles, alpha_tiles, n_tile_size, n_tiles);

    // std::ofstream k_file("./alpha_tiles.txt");
    // k_file << std::setprecision(12);
    // std::ostream_iterator<CALC_TYPE> k_iterator(k_file, "\n");
    // for (std::size_t i = 0; i < n_tiles; i++)
    // {
    //   std::vector<CALC_TYPE> k_ = alpha_tiles[i].get(); // Get the vector from the shared_future
    //   std::copy(k_.begin(), k_.end(), k_iterator);
    //   // std::vector<float> nuller = {0.0, 0.0, 0.0}; // Get the vector from the shared_future
    //   // std::copy(nuller.begin(), nuller.end(), k_iterator);
    // }

    // Compute loss
    compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
    // printf("iter: %d param: %.10lf loss: %.10lf\n", iter, hyperparameters[0], loss_value.get());
    // printf("iter: %d loss: %.12lf l: %.12lf, v: %.12lf, n: %.12lf\n", iter, loss_value.get(), hyperparameters[0], hyperparameters[1], hyperparameters[2]);

    // Fill y*y^T*inv(K)-I Cholesky Algorithms
    // update_grad_K_tiled_mkl(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size, n_tiles);

    // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
    // backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    update_hyperparameter(grad_I_tiles, grad_l_tiles, alpha_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter, 0);
    update_hyperparameter(grad_I_tiles, grad_v_tiles, alpha_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter, 1);
    update_noise_variance(grad_I_tiles, alpha_tiles, hyperparameters, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);

    printf("iter: %d loss: %.12lf l: %.12lf, v: %.12lf, n: %.12lf\n", iter, loss_value.get(), hyperparameters[0], hyperparameters[1], hyperparameters[2]);
  }

  //////////////////
  K_tiles.resize(n_tiles * n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    for (std::size_t j = 0; j <= i; j++)
    {
      K_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_covariance, "assemble_tiled"), i, j,
                                            n_tile_size, n_regressors, hyperparameters, training_input);
    }
  }
  alpha_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    alpha_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
  }
  // printf("l: %.12lf, v: %.12lf, n: %.12lf\n", hyperparameters[0], hyperparameters[1], hyperparameters[2]);
  // Assemble prior covariance matrix vector
  prior_K_tiles.resize(m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    prior_K_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_prior_covariance, "assemble_tiled"), i, i,
                                  m_tile_size, n_regressors, hyperparameters, test_input);
  }
  // Assemble MxN cross-covariance matrix vector
  cross_covariance_tiles.resize(m_tiles * n_tiles);
  // Assemble NxM (transpose) cross-covariance matrix vector
  t_cross_covariance_tiles.resize(n_tiles * m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    for (std::size_t j = 0; j < n_tiles; j++)
    {
      cross_covariance_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance, "assemble_pred"), i, j,
                                                           m_tile_size, n_tile_size, n_regressors, hyperparameters, test_input, training_input);

      t_cross_covariance_tiles[j * m_tiles + i] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gen_tile_cross_cov_T), "assemble_pred"),
                                                                m_tile_size, n_tile_size, cross_covariance_tiles[i * n_tiles + j]);
    }
  }
  // Assemble placeholder matrix for (K_MxN * (K^-1_NxN * K_NxM))
  prior_inter_tiles.resize(m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    prior_inter_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros_diag, "assemble_prior_inter"), m_tile_size);
  }
  // Assemble zero prediction
  prediction_tiles.resize(m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    prediction_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), m_tile_size);
  }
  // Assemble zero prediction
  prediction_uncertainty_tiles.resize(m_tiles);
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    prediction_uncertainty_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"), m_tile_size);
  }

  ///////////////////////////////////////////////////
  right_looking_cholesky_tiled_mkl(K_tiles, n_tile_size, n_tiles);
  // Triangular solve K_NxN * alpha = y
  forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);
  backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, n_tiles);

  /////////////////
  std::ofstream k_file("./tcross.txt");
  k_file << std::setprecision(12);
  std::ostream_iterator<CALC_TYPE> k_iterator(k_file, "\n");
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    for (std::size_t j = 0; j < n_tiles; j++)
    {
      std::vector<CALC_TYPE> k_ = t_cross_covariance_tiles[j * m_tiles + i].get(); // Get the vector from the shared_future
      std::copy(k_.begin(), k_.end(), k_iterator);
      // std::vector<float> nuller = {0.0, 0.0, 0.0}; // Get the vector from the shared_future
      // std::copy(nuller.begin(), nuller.end(), k_iterator);
    }
  }

  // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
  forward_solve_KK_tiled(K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
  // backward_solve_KK_tiled(K_tiles, cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

  //////////////////////////////////////////////////////////////////////////////
  // PART 3: PREDICTION
  prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
  // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
  posterior_covariance_tiled(t_cross_covariance_tiles, prior_inter_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

  // predicition uncertainty
  prediction_uncertainty_tiled(prior_K_tiles, prior_inter_tiles, prediction_uncertainty_tiles, m_tile_size, m_tiles);
  //  compute error
  ft_error = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&compute_error_norm), "prediction_tiled"), m_tiles, m_tile_size, test_output, prediction_tiles);
  //////////////////////////////////////////////////////////////////////////
  // write to file
  // predictions
  std::ofstream pred_file("./predictions.txt");
  std::ostream_iterator<CALC_TYPE> pred_iterator(pred_file, "\n");
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    std::vector<CALC_TYPE> k_ = prediction_tiles[i].get(); // Get the vector from the shared_future
    std::copy(k_.begin(), k_.end(), pred_iterator);
    // std::vector<float> nuller = {0.0, 0.0, 0.0}; // Get the vector from the shared_future
    // std::copy(nuller.begin(), nuller.end(), k_iterator);
  }
  // var of prediction
  std::ofstream var_file("./uncertainty.txt");
  std::ostream_iterator<CALC_TYPE> var_iterator(var_file, "\n");
  for (std::size_t i = 0; i < m_tiles; i++)
  {
    std::vector<CALC_TYPE> k_ = prediction_uncertainty_tiles[i].get(); // Get the vector from the shared_future
    std::copy(k_.begin(), k_.end(), var_iterator);
    // std::vector<float> nuller = {0.0, 0.0, 0.0}; // Get the vector from the shared_future
    // std::copy(nuller.begin(), nuller.end(), k_iterator);
  }
  // error
  CALC_TYPE average_error = ft_error.get() / n_test;
  error_file = fopen("error.csv", "w");
  fprintf(error_file, "\"error\",%lf\n", average_error);
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
