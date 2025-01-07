#include "gp_kernels.hpp"
#include <hpx/async_cuda/cuda_future.hpp>

// frequently used names
using hpx::cuda::experimental::check_cuda_error;

// Kernel function to compute covariance
__global__ void compute_tile_covariance(
    double *d_tile,
    const double *d_input,
    std::size_t N_tile,
    std::size_t n_regressors,
    std::size_t row_offset,
    std::size_t col_offset,
    gpxpy_hyper::SEKParams sek_params)
{
    // Compute the global indices of the thread
    std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_tile && j < N_tile)
    {
        std::size_t i_global = N_tile * row_offset + i;
        std::size_t j_global = N_tile * col_offset + j;

        double distance = 0.0;
        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            int offset = -n_regressors + 1 + k;
            int i_local = i_global + offset;
            int j_local = j_global + offset;

            double z_ik = (i_local >= 0) ? d_input[i_local] : 0.0;
            double z_jk = (j_local >= 0) ? d_input[j_local] : 0.0;
            distance += (z_ik - z_jk) * (z_ik - z_jk);
        }

        // Compute the covariance value
        double covariance = sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        // Add noise variance if diagonal
        if (i_global == j_global)
        {
            covariance += sek_params.noise_variance;
        }

        d_tile[i * N_tile + j] = covariance;
    }
}

// Wrapper function to invoke the CUDA kernel
void cuda_compute_tile_covariance(
    double *d_tile,
    const double *d_input,
    std::size_t N_tile,
    std::size_t n_regressors,
    std::size_t row_offset,
    std::size_t col_offset,
    gpxpy_hyper::SEKParams sek_params,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N_tile + 15) / 16, (N_tile + 15) / 16);

    compute_tile_covariance<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_tile, d_input, N_tile, n_regressors, row_offset, col_offset, sek_params);
    check_cuda_error(cudaStreamSynchronize(stream));
}
