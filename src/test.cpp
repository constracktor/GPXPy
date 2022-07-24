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
//https://eigen.tuxfamily.org/index.php?title=Benchmark
#include <iostream>

namespace ublas = boost::numeric::ublas;
// read access to a const matrix is faster!!!!!!!!!!!!!!

// solve L*B = A where L triangular
std::vector<CALC_TYPE> trsm(hpx::shared_future<std::vector<CALC_TYPE>> ft_L,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            std::size_t N)
{
  const auto L = ft_L.get(); // improve
  const auto A = ft_A.get();
  // solution vector
  std::vector<CALC_TYPE> B;
  B.resize(N * N);
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  // TRSM
  //boost::numeric::ublas::blas_3::tsm (M1 &m1, const T &t, const M2 &m2, C)-> m2 * C = t * m1 ->   m1 = solve (m2, t * m1, C ());
  B_blas = ublas::solve (L_blas, A_blas, ublas::lower_tag());
  //ublas::inplace_solve (L_blas, A_blas, ublas::lower_tag());
  // reformat to std::vector
  B = B_blas.data();
  return B;
}

//  A = A + B * B^T
std::vector<CALC_TYPE> syrk(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                            std::size_t N)
{
  auto A = ft_A.get(); // improve
  auto B = ft_B.get();
  // solution vector
  std::vector<CALC_TYPE> A_updated;
  A_updated.resize(N * N);
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_updated_blas(N, N);
  //SYRK
  //boost::numeric::ublas::blas_3::srk (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2) -> m1 = t * m1 + t2 * (m2 * m2T)
  A_updated_blas = A_blas + ublas::prod(B_blas, ublas::trans(B_blas));
  // reformat to std::vector
  A_updated = A_updated_blas.data();
  return A_updated;
}



// Cholesky decomposition of A -> return factorized matrix L
std::vector<CALC_TYPE> potrf(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                             std::size_t N)
{
  auto A = ft_A.get();
  // solution vector
  std::vector<CALC_TYPE> L;
  L.resize(N * N);
  std::fill(L.begin(), L.end(), 0.0);
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  // POTRF (compute Cholesky)
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
      // set diagonal entry
      CALC_TYPE L_kk = std::sqrt( qL_kk );
      L_blas(k,k) = L_kk;
      // compute corresponding column
      ublas::matrix_column<ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> >> cLk(L_blas, k);
      ublas::project( cLk, ublas::range(k+1, N) )
        = ( ublas::project( ublas::column(A_blas, k), ublas::range(k+1, N) )
            - ublas::prod( ublas::project(L_blas, ublas::range(k+1, N), ublas::range(0, k)),
                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  // reformat to std::vector
  L = L_blas.data();
  return L;
}

//C = A * B
std::vector<CALC_TYPE> gemm(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                            std::size_t N)
{
   auto A = ft_A.get(); // improve
   auto B = ft_B.get();
   // solution vector
   std::vector<CALC_TYPE> C;
   C.resize(N * N);
   // convert to boost matrices
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
   A_blas.data() = A;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
   B_blas.data() = B;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
   // GEMM
   //boost::numeric::ublas::blas_3::gmm (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2, const M3 &m3)
   C_blas = ublas::prod(A_blas, B_blas);
   // reformat to std::vector
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
        K_tiles[i * n_tiles + j] = hpx::dataflow(&gen_tile, i, j, tile_size, n_tiles);
     }
  }
  // Compute Cholesky decomposition
  //left_looking_cholesky(K_tiles,tile_size, n_tiles)

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
