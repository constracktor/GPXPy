#ifndef TILED_ALGORITHMS_GPU_H_INCLUDED
#define TILED_ALGORITHMS_GPU_H_INCLUDED

//#include <hpx/local/future.hpp>
#include "cublas_adapter.hpp"

// right-looking tiled Cholesky algorithm using cuBLAS
template <typename T>
void right_looking_cholesky_tiled_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                         std::vector<hpx::shared_future<std::vector<T>>> &ft_tiles,
                                         std::size_t N,
                                         std::size_t n_tiles)

{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(&potrf<T>), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::unwrapping(&trsm<T>), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    // using cublas for tile update
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = syrk_cublas<T>(cublas, ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = gemm_cublas<T>(cublas, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}
#endif
