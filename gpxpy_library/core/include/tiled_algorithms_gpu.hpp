#ifndef TILED_ALGORITHMS_GPU_H
#define TILED_ALGORITHMS_GPU_H

#include "adapter_cublas.hpp"

/**
 * Right-looking tiled Cholesky algorithm using cuBLAS
 */
void right_looking_cholesky_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::size_t N, std::size_t n_tiles)
{
    // Counter to equally split workload among the cublas executors.
    // Currently only one cublas executor.  TODO: change to multiple executors.
    std::size_t counter = 0;
    std::size_t n_executors = 1;  // TODO: set to cublas.size();

    for (std::size_t k = 0; k < n_tiles; k++) {
        // POTRF
        ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(&potrf),
                                                  ft_tiles[k * n_tiles + k], N);

        for (std::size_t m = k + 1; m < n_tiles; m++) {
            // TRSM
            ft_tiles[m * n_tiles + k] =
                hpx::dataflow(hpx::unwrapping(&trsm), ft_tiles[k * n_tiles + k],
                              ft_tiles[m * n_tiles + k], N);
        }

        // using cublas for tile update
        for (std::size_t m = k + 1; m < n_tiles; m++) {
            // TODO: uncomment to use multiple cublas executors
            // // increase or reset counter
            // counter = (counter < n_executors - 1) ? counter + 1 : 0;

            // SYRK
            ft_tiles[m * n_tiles + m] =
                syrk(cublas[counter], ft_tiles[m * n_tiles + m],
                     ft_tiles[m * n_tiles + k], N);

            for (std::size_t n = k + 1; n < m; n++) {
                // TODO: uncomment to use multiple cublas executors
                // // increase or reset counter
                // counter = (counter < n_executors - 1) ? counter + 1 : 0;

                // GEMM
                ft_tiles[m * n_tiles + n] = gemm(
                    cublas[counter], ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
            }
        }
    }
}

#endif // end of TILED_ALGORITHMS_GPU_H
