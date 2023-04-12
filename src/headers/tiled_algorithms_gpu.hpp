#ifndef TILED_ALGORITHMS_GPU_H_INCLUDED
#define TILED_ALGORITHMS_GPU_H_INCLUDED

//#include <hpx/local/future.hpp>
#include <iostream>
#include "kokkos_adapter.hpp"

// right-looking tiled Cholesky algorithm using Kokkos
template <typename T, typename ExecutionSpace>
void right_looking_cholesky_tiled_kokkos(ExecutionSpace &&inst,
                                         std::vector<hpx::shared_future<host_buffer_t<T>>> &ft_tiles,
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
    // using kokkos for tile update
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&syrk_kokkos<T,ExecutionSpace>), "cholesky_tiled"), inst,  ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&gemm_kokkos<T,ExecutionSpace>), "cholesky_tiled"), inst, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}
#endif
