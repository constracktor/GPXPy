#ifndef TILED_ALGORITHMS_GPU_H
#define TILED_ALGORITHMS_GPU_H

#include "../include/adapter_cublas.hpp"

// Tiled Cholesky Algorithm ------------------------------------------------ {{{

void right_looking_cholesky_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::size_t N,
    std::size_t n_tiles)
{
    // Counter to equally split workload among the cublas executors.
    // Currently only one cublas executor.  TODO: change to multiple executors.
    std::size_t counter = 0;
    std::size_t n_executors = 1; // TODO: set to cublas.size();

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

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{}

void backward_solve_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{}

// Tiled Triangular Solve Algorithms for matrices (K * X = B)
void forward_solve_tiled_matrix(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

void backward_solve_tiled_matrix(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

// }}} -------------------------------- end of Tiled Triangular Solve Algorithms

// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
// Tiled Triangular Solve Algorithms for Matrices (K * X = B)
void forward_solve_KcK_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

void compute_gemm_of_invK_y(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_invK,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_y,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_alpha,
    std::size_t N,
    std::size_t n_tiles)
{}

// Tiled Loss
void compute_loss_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_alpha,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_y,
    hpx::shared_future<double>& loss,
    std::size_t N,
    std::size_t n_tiles)
{}

// Tiled Prediction
void prediction_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_vector,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_rhs,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

// Tiled Diagonal of Posterior Covariance Matrix
void posterior_covariance_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_inter_tiles,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

// Tiled Diagonal of Posterior Covariance Matrix
void full_cov_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_priorK,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{}

// Tiled Prediction Uncertainty
void prediction_uncertainty_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_inter,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{}

// Tiled Prediction Uncertainty
void pred_uncer_tiled(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{}

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>>& ft_tiles,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_v1,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_v2,
    std::size_t N,
    std::size_t n_tiles)
{}

// Perform a gradient scent step for selected hyperparameter using Adam
// algorithm
void update_hyperparameter(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_alpha,
    double* hyperparameters,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>>& m_T,
    std::vector<hpx::shared_future<double>>& v_T,
    const std::vector<hpx::shared_future<double>>& beta1_T,
    const std::vector<hpx::shared_future<double>>& beta2_T,
    int iter,
    int param_idx)
{}

// Update noise variance using gradient decent + Adam
void update_noise_variance(
    std::vector<hpx::cuda::experimental::cublas_executor> cublas,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>>& ft_alpha,
    double* hyperparameters,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>>& m_T,
    std::vector<hpx::shared_future<double>>& v_T,
    const std::vector<hpx::shared_future<double>>& beta1_T,
    const std::vector<hpx::shared_future<double>>& beta2_T,
    int iter)
{}

#endif
