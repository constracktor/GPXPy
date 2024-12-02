#include "../include/tiled_algorithms_gpu.hpp"

#include "../include/adapter_cublas.hpp"

namespace gpu
{

// Tiled Cholesky Algorithm -------------------------------------------- {{{

void right_looking_cholesky_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::size_t N,
    std::size_t n_tiles)
{
    // Counter to equally split workload among the cublas executors.
    // Currently only one cublas executor.
    std::size_t counter = 0;
    std::size_t n_executors = 1;  // TODO: set to cublas.size();

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] = hpx::dataflow(
            hpx::annotated_function(&potrf, "cholesky_tiled_gpu"), &stream, ft_tiles[k * n_tiles + k], N);

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(&trsm, "cholesky_tiled_gpu"),
                &cublas[counter],
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }

        // using cublas for tile update
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TODO: uncomment to use multiple cublas executors
            // // increase or reset counter
            // counter = (counter < n_executors - 1) ? counter + 1 : 0;

            // SYRK
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::annotated_function(&syrk, "cholesky_tiled_gpu"),
                &cublas[counter],
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);

            for (std::size_t n = k + 1; n < m; n++)
            {
                // TODO: uncomment to use multiple cublas executors
                // // increase or reset counter
                // counter = (counter < n_executors - 1) ? counter + 1 : 0;

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(&gemm, "cholesky_tiled_gpu"),
                    &cublas[counter],
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{ }

void backward_solve_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Triangular Solve Algorithms for matrices (K * X = B)
void forward_solve_tiled_matrix(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

void backward_solve_tiled_matrix(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// }}} ---------------------------- end of Tiled Triangular Solve Algorithms

// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
// Tiled Triangular Solve Algorithms for Matrices (K * X = B)
void forward_solve_KcK_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

void compute_gemm_of_invK_y(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Loss
void compute_loss_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    hpx::shared_future<double> &loss,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Prediction
void prediction_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Diagonal of Posterior Covariance Matrix
void posterior_covariance_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter_tiles,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Diagonal of Posterior Covariance Matrix
void full_cov_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Prediction Uncertainty
void prediction_uncertainty_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{ }

// Tiled Prediction Uncertainty
void pred_uncer_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{ }

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v1,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v2,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Perform a gradient scent step for selected hyperparameter using Adam
// algorithm
static double
update_hyperparameter(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    double &hyperparameter,  // lengthscale or vertical-lengthscale
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    int param_idx)  // 0 for lengthscale, 1 for vertical-lengthscale
{ }

double update_lengthscale(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.lengthscale,
        sek_params,
        adam_params,
        N,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        0);
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.vertical_lengthscale,
        sek_params,
        adam_params,
        N,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        1);
}

double update_noise_variance(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{ }

}  // namespace gpu
