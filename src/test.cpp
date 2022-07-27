#define CALC_TYPE double
#define TYPE "%lf"
#include <math.h>

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
////////////////////////////////////////////////////////////////////////////////
// GP functions to assemble K
std::vector<CALC_TYPE> compute_regressor_vector(std::size_t row, std::size_t n_regressors, std::vector<CALC_TYPE> input)
{
  std::vector<CALC_TYPE> z_row;
  z_row.resize(n_regressors);
  for (std::size_t i = 0; i < n_regressors; i++)
  {
   int index = row - n_regressors + 1 + i;
   if (index < 0)
   {
      z_row[i] = 0.0;
   }
   else
   {
     z_row[i] = input[index];
   }
  }
  return z_row;
}

CALC_TYPE compute_covariance_function(std::size_t n_regressors, CALC_TYPE* hyperparameters, std::vector<CALC_TYPE> z_i, std::vector<CALC_TYPE> z_j)
{
  // Compute the Squared Exponential Covariance Function
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  CALC_TYPE distance = 0.0;
  for (std::size_t i = 0; i < n_regressors; i++)
  {
    distance += pow(z_i[i] - z_j[i],2);
  }
  return hyperparameters[1] * exp(-0.5 * hyperparameters[0] * distance);
}

std::vector<CALC_TYPE> gen_tile(std::size_t row, std::size_t col, std::size_t N_row, std::size_t N_col, std::size_t n_tiles, std::size_t n_regressors, CALC_TYPE *hyperparameters, std::vector<CALC_TYPE> z_i_input, std::vector<CALC_TYPE> z_j_input)
{
   std::size_t i_global,j_global;
   CALC_TYPE covariance_function;
   //std::vector<CALC_TYPE> z_i, z_j;
   //z_i.resize(n_regressors);
   //z_j.resize(n_regressors);

   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N_row * N_col);
   for(std::size_t i = 0; i < N_row; i++)
   {
      i_global = N_row * row + i;
      std::vector<CALC_TYPE> z_i = compute_regressor_vector(i_global, n_regressors, z_i_input);

      for(std::size_t j = 0; j < N_col; j++)
      {
         j_global = N_col * col + j;
         std::vector<CALC_TYPE> z_j= compute_regressor_vector(j_global, n_regressors, z_j_input);
         // compute covariance function
         covariance_function = compute_covariance_function(n_regressors, hyperparameters, z_i, z_j);
         if (i_global==j_global)
         {
           covariance_function += hyperparameters[2];
         }
         tile[i * N_row + j] = covariance_function;
      }
   }
   return tile;
}

std::vector<CALC_TYPE> gen_tile_test(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles)
{
   std::size_t i_global,j_global;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N * N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      for(std::size_t j = 0; j < N; j++)
      {
         j_global = N * col + j;

         if(i_global == j_global)
         {
           tile[i * N + j] = 2.0;
         }
         else if( (i_global == N*n_tiles -1&& j_global == 0) || (i_global == 0 && j_global == N*n_tiles-1) || (i_global + j_global)== 3*n_tiles*N/4)
         {
           tile[i * N + j] = .5;
         }
         else
         {
           tile[i * N + j] = 1.0;
         }
      }
   }
   return tile;
}
////////////////////////////////////////////////////////////////////////////////
// BLAS operations
// set tile to zero (for inplace)
std::vector<CALC_TYPE> zeros(std::size_t N)
{
  std::vector<CALC_TYPE> zeros;
  zeros.resize(N * N);
  std::fill(zeros.begin(), zeros.end(), 0.0);
  return zeros;
}

// solve L * B^T = A^T and return B where L triangular
std::vector<CALC_TYPE> trsm(hpx::shared_future<std::vector<CALC_TYPE>> ft_L,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            std::size_t N)
{
  auto L = ft_L.get(); // improve
  auto A = ft_A.get();
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
  B_blas = ublas::trans(ublas::solve(L_blas, ublas::trans(A_blas), ublas::lower_tag()));
  // reformat to std::vector
  B = B_blas.data();
  return B;
}

//  A = A - B * B^T
std::vector<CALC_TYPE> syrk(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                            std::size_t N)
{
  //hpx::when_all()
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
  A_updated_blas = A_blas - ublas::prod(B_blas,ublas::trans(B_blas));
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
      std::cout << qL_kk << '\n';
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

//C = C - A * B^T
std::vector<CALC_TYPE> gemm(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                            hpx::shared_future<std::vector<CALC_TYPE>> ft_C,
                            std::size_t N)
{
   auto A = ft_A.get(); // improve
   auto B = ft_B.get();
   auto C = ft_C.get();
   // solution vector
   std::vector<CALC_TYPE> C_updated;
   C.resize(N * N);
   // convert to boost matrices
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
   A_blas.data() = A;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
   B_blas.data() = B;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
   C_blas.data() = C;
   ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_updated_blas(N, N);
   // GEMM
   C_updated_blas = C_blas - ublas::prod(A_blas, ublas::trans(B_blas));
   // reformat to std::vector
   C_updated = C_updated_blas.data();
   return C_updated;
}

////////////////////////////////////////////////////////////////////////////////
// Tiled Cholesky Algorithms
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(&potrf, ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(&trsm, ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(&syrk, ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(&gemm, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
    // in theory not mandadory
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // set zero
      ft_tiles[k * n_tiles + m] = hpx::dataflow(&zeros, N);
    }
  }
}

void left_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(&syrk, ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
      for (std::size_t m = k + 1; m < n_tiles; m++)
      {
        // GEMM
        ft_tiles[m * n_tiles + k] = hpx::dataflow(&gemm, ft_tiles[m * n_tiles + n], ft_tiles[k * n_tiles + n], ft_tiles[m * n_tiles + k], N);
      }
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(&potrf, ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(&trsm, ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    // in theory not mandadory
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // set zero
      ft_tiles[k * n_tiles + m] = hpx::dataflow(&zeros, N);
    }
  }
}

void top_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      for (std::size_t m = 0; m < n; m++)
      {
        // GEMM
        ft_tiles[k * n_tiles + n] = hpx::dataflow(&gemm, ft_tiles[k * n_tiles + m], ft_tiles[n * n_tiles + m], ft_tiles[k * n_tiles + n], N);
      }
      // TRSM
      ft_tiles[k * n_tiles + n] = hpx::dataflow(&trsm, ft_tiles[n * n_tiles + n], ft_tiles[k * n_tiles + n], N);
    }
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(&syrk, ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(&potrf, ft_tiles[k * n_tiles + k], N);
    // in theory not mandadory
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // set zero
      ft_tiles[k * n_tiles + m] = hpx::dataflow(&zeros, N);
    }
  }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  // GP parameters
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE    hyperparameters[3];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1; // noise_variance = small value
  // tiled parameters
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  std::size_t tile_size = n_train / n_tiles;
  // HPX structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance;
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
  ////////////////////////////////////////////////////////////////////////////
  // Load data
  training_input.resize(n_train);
  training_output.resize(n_train);
  test_input.resize(n_test);
  test_output.resize(n_test);
  training_input_file = fopen("../src/data/training/training_input.txt", "r");
  training_output_file = fopen("../src/data/training/training_output.txt", "r");
  test_input_file = fopen("../src/data/test/test_input_3.txt", "r");
  test_output_file = fopen("../src/data/test/test_output_3.txt", "r");
  if (training_input_file == NULL || training_output_file == NULL || test_input_file == NULL || test_output_file == NULL)
  {
    printf("Files not found!\n");
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
  //////////////////////////////////////////////////////////////////////////////
  // Start timer
  hpx::chrono::high_resolution_timer t;
  //////////////////////////////////////////////////////////////////////////////
  // Assemble covariance matrix vector
/*
  K_tiles.resize(n_tiles * n_tiles);
  std::cout << "assembly start " <<'\n';
  // Assemble K
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; ++j)
     {
        K_tiles[i * n_tiles + j] = hpx::dataflow(&gen_tile, i, j, tile_size, tile_size, n_tiles, n_regressors, hyperparameters, training_input, training_input);
     }
  }
  //Assemble cross-covariacne (currently not tiled)
  cross_covariance.resize(1);
  cross_covariance[0] = hpx::dataflow(&gen_tile, 0, 0,n_train, n_test, 1, n_regressors, hyperparameters, training_input,test_input);
  std::cout << "assembly done " <<'\n';
  std::cout << "Cholesky start " <<'\n';
  // Compute Cholesky decomposition
  //left_looking_cholesky_tiled(K_tiles,tile_size, n_tiles);

  // Currently quick and dirty triangular solve
  // Assemble to big matrix
  /*
  std::vector<CALC_TYPE> L;
  L.resize(n_train * n_train);

  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; ++j)
     {
        auto tile = K_tiles[i * n_tiles + j].get();
        for(std::size_t k = 0; k < tile_size * tile_size; k++)
        {
          L[i * tile_size * tile_size * n_tiles + j * tile_size * tile_size + k] = tile[k];
        }
     }
  }
  std::cout << "Cholesky done " <<'\n';
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(n_train, n_train);
  L_blas.data() = L;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > cross_covariance_blas(n_train, n_test);
  cross_covariance_blas.data() = cross_covariance[0].get();
  // convert to boost vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > alpha(n_train,1);
  alpha.data() = training_output;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > y_test(n_test,1);
  y_test.data() = test_output;
  std::cout << "boost stuff done " <<'\n';
  // solve triangular systems
  ublas::inplace_solve(L_blas, alpha, ublas::lower_tag() );
  std::cout << "first solving done " <<'\n';
  ublas::inplace_solve(ublas::trans(L_blas), alpha, ublas::upper_tag());
  // make predictions
  std::cout << "second solving done " <<'\n';
  alpha = ublas::prod(ublas::trans(cross_covariance_blas), alpha);
  // compute error
  CALC_TYPE error = 1.0;//ublas::norm_2(alpha - y_test);
  std::cout << "average_error: " << error / n_test << '\n';
*/
std::size_t N = 2;
std::size_t T = 2;
std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> tiles;
tiles.resize(T * T);
for (std::size_t i = 0; i < T; ++i)
{
   for (std::size_t j = 0; j < T; ++j)
   {
      tiles[i * T + j] = hpx::dataflow(&gen_tile_test, i, j, N, T);
   }
}

std::cout << "before:"<< std::endl;
for (std::size_t i = 0; i < T; i++)
{
   for (std::size_t j = 0; j < T; ++j)
   {
     std::vector<CALC_TYPE> A = tiles[i*T+j].get();
     std::cout << "tile: " << i*T+j << std::endl;
     for(int k = 0; k < N; ++k)
     {
        for(int m = 0; m < N; ++m)
        {
           std::ostringstream os;
           os << A[k * N + m] << " ";
           std::cout << os.str();
        }
        std::cout << std::endl;
     }
     std::cout << std::endl;
   }
 }

top_looking_cholesky_tiled(tiles,N, T);

std::cout << "after:" << std::endl;
for (std::size_t i = 0; i < T; i++)
{
   //std::vector<CALC_TYPE> x = tiles[i];
   for (std::size_t j = 0; j < T; ++j)
   {
     std::vector<CALC_TYPE> A = tiles[i*T+j].get();
     std::cout << "tile: " << i*T+j << std::endl;
     for(int k = 0; k < N; ++k)
     {
        for(int m = 0; m < N; ++m)
        {
           std::ostringstream os;
           os << A[k * N + m] << " ";
           std::cout << os.str();
        }
        std::cout << std::endl;
     }
     std::cout << std::endl;
   }
}

std::vector<CALC_TYPE> L;
L.resize(N * N * T * T);

for (std::size_t k = 0; k < T; k++)
{
   for (std::size_t l = 0; l < T; ++l)
   {
      auto tile = tiles[k * T + l].get();
      for(std::size_t i = 0; i < N; i++)
      {
         std::size_t i_global = N * k + i;
         for(std::size_t j = 0; j < N; j++)
         {
            std::size_t j_global = N * l + j;

            L[i_global * N*T + j_global] = tile[i * N + j];
         }
      }
   }
}
for(std::size_t i = 0; i < N*T; i++)
{
  for(std::size_t j = 0; j < N*T; j++)
  {
   std::cout << L[i * N*T + j] <<" ";
  }
  std::cout << '\n';
}

std::cout << "Cholesky done " <<'\n';
// convert to boost matrices
ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N * T, N * T);
L_blas.data() = L;
// convert to boost vectors
std::vector<CALC_TYPE> a;
a.resize(N*T);
a[0] = 1.0;
a[1] = 1.0;
a[2] = 1.0;
a[3] = 1.0;
ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > alpha(N*T,1);
alpha.data() = a;
std::cout << "boost stuff done " <<'\n';
// solve triangular systems
ublas::inplace_solve(L_blas, alpha, ublas::lower_tag() );
std::cout << "first solving done " <<'\n';
ublas::inplace_solve(ublas::trans(L_blas), alpha, ublas::upper_tag());

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
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
