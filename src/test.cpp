#define CALC_TYPE double
#define TYPE "%lf"

#include <hpx/local/chrono.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/format.hpp>

#include <cassert>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

#include <boost/numeric/ublas/triangular.hpp>




#include <iostream>

namespace ublas = boost::numeric::ublas;

// Cholesky decomposition of A -> return factorized matrix L
std::vector<CALC_TYPE> potrf(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                             std::size_t N)
{
  std::vector<CALC_TYPE> L;
  L.resize(N * N);
  std::fill(L.begin(), L.end(), 0.0);

  auto A = ft_A.get(); // improve
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  // Initialize result matrix
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  // compute Cholesky
  for (size_t k=0 ; k < N; k++)
  {
    // compute squared diagonal entry
    CALC_TYPE qL_kk = A_blas(k,k) - ublas::inner_prod( ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ), ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ) );
    // check if positive
    if (qL_kk <= 0)
    {
      // abort ?
    }
    else
    {
      CALC_TYPE L_kk = std::sqrt( qL_kk );
      L_blas(k,k) = L_kk;

      ublas::matrix_column<ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> >> cLk(L_blas, k);
      ublas::project( cLk, ublas::range(k+1, N) )
        = ( ublas::project( ublas::column(A_blas, k), ublas::range(k+1, N) )
            - ublas::prod( ublas::project(L_blas, ublas::range(k+1, N), ublas::range(0, k)),
                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  L = L_blas.data();
  return L;
}

//C = A * B
std::vector<CALC_TYPE> gemm(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                           hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                           std::size_t N)
{
   std::vector<CALC_TYPE> C;
   C.resize(N * N);

   auto A = ft_A.get(); // improve
   auto B = ft_B.get();
   // convert to boost matrices
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
   A_blas.data() = A;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
   B_blas.data() = B;
   // Initialize result matrix
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
   C_blas = ublas::prod(A_blas, B_blas);

   C = C_blas.data();
   return C;
}

std::vector<CALC_TYPE> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles)
{
   std::size_t i_global,j_global;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N * N);

   for(std::size_t i = 0; i < N; i++)
   {
      i_global = (n_tiles - 1) * N + i;
      for(std::size_t j = 0; j < N; j++)
      {
         j_global = (n_tiles - 1) * N + j;
         tile[i * N + j] = 1; //covariance_function(i_global,j_global);
      }
   }
   return tile;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();

  std::size_t tile_size = n_train / n_tiles;

  hpx::chrono::high_resolution_timer t;


  ublas::vector<CALC_TYPE> vec;

/*
  std::cout << "Using Boost "
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;
        */
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  K_tiles.resize(n_tiles * n_tiles);
  // Assemble K
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; ++j)
     {
        //K_tiles[i * n_tiles + j] = hpx::dataflow(&gen_tile, i, j, tile_size, n_tiles);
     }
  }
  // Compute Cholesky decomposition
  //left_looking_cholesky(K_tiles,tile_size, n_tiles)
  // test gemm
  int N = 3;
  std::vector<CALC_TYPE> result;
  result.resize(N * N);

  std::vector<CALC_TYPE> A;
  std::vector<CALC_TYPE> B;
  A.resize(N * N);
  B.resize(N * N);
  /*
  for(int i = 0; i < N * N; i++)
  {
    A[i] = 1.0;
    B[i] = 1.0;
  }
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  // Initialize result matrix
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
  C_blas = ublas::prod(A_blas, B_blas);
  result = C_blas.data();
  for(int i = 0; i < N * N; i++)
  {
    //std::cout << A[i] << std::endl;
    //std::cout << B[i] << std::endl;
    std::cout << result[i] << std::endl;
    //std::cout << std::endl;
  }
  */
  //
  //template <class MATRIX>
  //typedef typename MATRIX::value_type T;
  //std::fill(A.begin(), A.end(), 0.0);
  A[0] = 4.0;A[1] = 4.0;A[2] = 2.0;
  A[3] = 4.0;A[4] = 8.0;A[5] =4.0;
  A[6] = 2.0;A[7] = 4.0;A[8] = 3.0;

  std::vector<CALC_TYPE> L;
  L.resize(N * N);
  std::fill(L.begin(), L.end(), 0.0);
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  // Initialize result matrix
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  // compute Cholesky
  for (size_t k=0 ; k < N; k++)
  {
    // compute squared diagonal entry
    CALC_TYPE qL_kk = A_blas(k,k) - ublas::inner_prod( ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ), ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ) );
    // check if positive
    /*
    if (qL_kk <= 0)
    {
      return ;
    }

    else*/
    {
      CALC_TYPE L_kk = std::sqrt( qL_kk );
      L_blas(k,k) = L_kk;

      ublas::matrix_column<ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> >> cLk(L_blas, k);
      ublas::project( cLk, ublas::range(k+1, N) )
        = ( ublas::project( ublas::column(A_blas, k), ublas::range(k+1, N) )
            - ublas::prod( ublas::project(L_blas, ublas::range(k+1, N), ublas::range(0, k)),
                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  L = L_blas.data();
  for(int i = 0; i < N * N; i++)
  {
    std::cout << L[i] << std::endl;
  }


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
        ("n_tiles", hpx::program_options::value<std::size_t>()->default_value(10),
        "Number of tiles per dimension -> n_tiles * n_tiles total")
    ;
    hpx::program_options::variables_map vm;
    hpx::program_options::store(hpx::program_options::parse_command_line(argc, argv, desc_commandline), vm);
    // GP parameters
    std::size_t       n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
    std::size_t       n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
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
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
