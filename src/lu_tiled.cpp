#include <hpx/local/chrono.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/format.hpp>

#include <cstdint>
#include <iostream>

#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_transform.hpp>

#include <hpx/program_options.hpp>

#include <algorithm>
#include <vector>
#include <iostream>

#define CALC_TYPE double

void print_tile(std::vector<CALC_TYPE> A, std::string id, std::size_t row, std::size_t col, std::size_t N)
{
   for(int i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
         std::ostringstream os;
         os << id << " " << row * N + i << " " << col * N + j << " " << A[i * N + j] << std::endl;
         std::cout << os.str();
      }
   }
}

std::vector<CALC_TYPE> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t T)
{
   std::vector<CALC_TYPE> v;
   v.resize(N * N);
   std::srand(row * T + col);
   for(int i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
         v[i * N + j] = (double) (std::rand() % 10000 - 5000) / 1000;
      }
   }
   return v;
}

//return inv
std::vector<CALC_TYPE> inversion(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                                 std::size_t N)
{
   auto A = ft_A.get();
   double tmp;
   std::vector<CALC_TYPE> v;
   v.resize(N * N);

   for(int i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
         v[i * N + j] = 0;
      }
      v[i * N + i] = 1;
   }
   for(int k = 0; k < N; ++k) {
      tmp = A[k * N + k];
      for(int j = 0; j < N; ++j) {
         v[k * N + j] /= tmp;
         A[k * N + j] /= tmp;
      }

      // can be parallalized
      for(int i = 0; i < k; ++i) {
         tmp = A[i * N + k];
         for(int j = 0; j < N; ++j) {
            v[i * N + j] -= tmp * v[k * N + j];
            A[i * N + j] -= tmp * A[k * N + j];
         }
      }
      for(int i = k + 1; i < N; ++i) {
         tmp = A[i * N + k];
         for(int j = 0; j < N; ++j) {
            v[i * N + j] -= tmp * v[k * N + j];
            A[i * N + j] -= tmp * A[k * N + j];
         }
      }
   }
   return v;
}

//A = A * B
std::vector<CALC_TYPE> pmm(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                           hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                           std::size_t N)
{
   std::vector<CALC_TYPE> v;
   v.resize(N * N);
   auto A = ft_A.get();
   auto B = ft_B.get();

   // i and j can be parallalized
   for(int i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
         v[i * N + j] = 0;
         for(int k = 0; k < N; ++k) {
            v[i * N + j] += A[i * N + k] * B[k * N + j];
         }
      }
   }
   return v;
}

//C = C - A * B
std::vector<CALC_TYPE> pmm_d(hpx::shared_future<std::vector<CALC_TYPE>> ft_A,
                             hpx::shared_future<std::vector<CALC_TYPE>> ft_B,
                             hpx::shared_future<std::vector<CALC_TYPE>> ft_C,
                             std::size_t N)
{
   auto A = ft_A.get();
   auto B = ft_B.get();
   auto C = ft_C.get();

   // i and j can be parallalized
   for(int i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
         for(int k = 0; k < N; ++k) {
            C[i * N + j] -= A[i * N + k] * B[k * N + j];
         }
      }
   }
   return C;
}


void lu_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t T)
{
    std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> ft_inv;
    ft_inv.resize(T);

    for (std::size_t k = 0; k < T - 1; ++k)
    {
       ft_inv[k] = hpx::dataflow(&inversion, ft_tiles[k * T + k], N);
       for (std::size_t i = k + 1; i < T; ++i)
       {
          ft_tiles[i * T + k] = hpx::dataflow(&pmm, ft_tiles[i * T + k], ft_inv[k], N);
          for (std::size_t j = k + 1; j < T; ++j)
          {
             ft_tiles[i * T + j] = hpx::dataflow(&pmm_d, ft_tiles[i * T + k], ft_tiles[k * T + j], ft_tiles[i * T + j], N);
          }
       }
    }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t N = vm["N"].as<std::size_t>();
    std::size_t T = vm["T"].as<std::size_t>();
    std::string out = vm["out"].as<std::string>();

    hpx::chrono::high_resolution_timer t;

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

    double elapsed = t.elapsed();
    std::cout << "Elapsed " << elapsed << " s\n";
    return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline;

    desc_commandline.add_options()
        ("N", hpx::program_options::value<std::size_t>()->default_value(10),
         "Dimension of each Tile (N*N elements per tile)")
        ("T", hpx::program_options::value<std::size_t>()->default_value(10),
         "Number of Tiles in each dimension (T*T tiles)")
        ("out", hpx::program_options::value<std::string>()->default_value("no"),
         "(debug) => print matrices in coo format")
    ;

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
