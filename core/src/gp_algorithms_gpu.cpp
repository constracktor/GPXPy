#ifdef GPXPY_WITH_CUDA
    #include "../include/gp_algorithms_gpu.hpp"

    #include "../include/gp_kernels.hpp"
    #include "../include/target.hpp"
    #include "../include/tiled_algorithms_gpu.hpp"

namespace gpu
{

double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   gpxpy_hyper::SEKParams sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input)
{
    /* // formula in papers:
    // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
    // noise_variance for diagonal must be added outside this function

    double z_ik = 0.0;
    double z_jk = 0.0;
    double distance = 0.0;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        int offset = -n_regressors + 1 + k;
        int i_local = i_global + offset;
        int j_local = j_global + offset;

        if (i_local >= 0)
        {
            z_ik = i_input[i_local];
        }
        if (j_local >= 0)
        {
            z_jk = j_input[j_local];
        }
        distance += pow(z_ik - z_jk, 2);
    }
    return sek_params.vertical_lengthscale * exp(-1.0 / (2.0 * pow(sek_params.lengthscale, 2.0)) * distance); */
}

double *
gen_tile_covariance(std::size_t row,
                    std::size_t col,
                    std::size_t N_tile,
                    std::size_t n_regressors,
                    gpxpy_hyper::SEKParams sek_params,
                    const std::vector<double> &input)
{
    /* std::size_t i_global, j_global;
    double covariance_function;

    // Initialize tile
    double *tile;
    hpx::cuda::experimental::check_cuda_error(
        cudaMalloc((void **) &tile, N_tile * N_tile * sizeof(double)));

    for (std::size_t i = 0; i < N_tile; i++)
    {
        i_global = N_tile * row + i;
        for (std::size_t j = 0; j < N_tile; j++)
        {
            j_global = N_tile * col + j;

            // compute covariance function
            covariance_function = compute_covariance_function(
                i_global, j_global, n_regressors, sek_params, input, input);

            // add noise variance on diagonal
            if (i_global == j_global)
            {
                covariance_function += sek_params.noise_variance;
            }
            tile[i * N_tile + j] = covariance_function;
        }
    }
    return std::move(tile); */
}

std::vector<double>
gen_tile_full_prior_covariance(std::size_t row,
                               std::size_t col,
                               std::size_t N,
                               std::size_t n_regressors,
                               gpxpy_hyper::SEKParams sek_params,
                               const std::vector<double> &input)
{
    /* std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance_function = compute_covariance_function(
                i_global, j_global, n_regressors, sek_params, input, input);
            tile[i * N + j] = covariance_function;
        }
    }
    return std::move(tile); */
}

std::vector<double>
gen_tile_prior_covariance(std::size_t row,
                          std::size_t col,
                          std::size_t N,
                          std::size_t n_regressors,
                          gpxpy_hyper::SEKParams sek_params,
                          const std::vector<double> &input)
{
    /* std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        j_global = N * col + i;
        // compute covariance function
        covariance_function =
            compute_covariance_function(i_global, j_global, n_regressors, sek_params, input, input);
        tile[i] = covariance_function;
    }
    return std::move(tile); */
}

std::vector<double>
gen_tile_cross_covariance(std::size_t row,
                          std::size_t col,
                          std::size_t N_row,
                          std::size_t N_col,
                          std::size_t n_regressors,
                          gpxpy_hyper::SEKParams sek_params,
                          const std::vector<double> &row_input,
                          const std::vector<double> &col_input)
{
    /* std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N_row * N_col);
    for (std::size_t i = 0; i < N_row; i++)
    {
        i_global = N_row * row + i;
        for (std::size_t j = 0; j < N_col; j++)
        {
            j_global = N_col * col + j;
            // compute covariance function
            covariance_function = compute_covariance_function(
                i_global, j_global, n_regressors, sek_params, row_input, col_input);
            tile[i * N_col + j] = covariance_function;
        }
    }
    return std::move(tile); */
}

std::vector<double>
gen_tile_cross_cov_T(std::size_t N_row,
                     std::size_t N_col,
                     const std::vector<double> &cross_covariance_tile)
{
    /* std::vector<double> transposed;
    transposed.resize(N_row * N_col);
    for (std::size_t i = 0; i < N_row; ++i)
    {
        for (std::size_t j = 0; j < N_col; j++)
        {
            transposed[j * N_row + i] =
                cross_covariance_tile[i * N_col + j];
        }
    }
    return std::move(transposed); */
}

std::vector<double> gen_tile_output(std::size_t row,
                                    std::size_t N,
                                    const std::vector<double> &output)
{
    /* std::size_t i_global;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        tile[i] = output[i_global];
    }
    return std::move(tile); */
}

double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    /* double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        auto a = tiles[k];
        for (std::size_t i = 0; i < tile_size; i++)
        {
            std::size_t i_global = tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return std::move(sqrt(error)); */
}

std::vector<double> gen_tile_zeros(std::size_t N)
{
    /* // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    std::fill(tile.begin(), tile.end(), 0.0);
    return std::move(tile); */
}

hpx::shared_future<std::vector<double>>
predict(const std::vector<double> &training_input,
        const std::vector<double> &training_output,
        const std::vector<double> &test_input,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors,
        gpxpy_hyper::SEKParams sek_params,
        std::shared_ptr<gpxpy::Target> target)
{
    /* // declare tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prediction_tiles;

    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled_K"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] =
            hpx::async(hpx::annotated_function(&gen_tile_output,
                                               "assemble_tiled_alpha"),
                       i,
                       n_tile_size,
                       training_output);
    }
    // Assemble MxN cross-covariance matrix vector
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(
                               &gen_tile_cross_covariance, "assemble_pred"),
                           i,
                           j,
                           m_tile_size,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           test_input,
                           training_input);
        }
    }
    // Assemble placeholder for prediction
    prediction_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
            m_tile_size);
    }

    // Compute Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Compute predictions
    prediction_tiled(target.cublas_executors, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);

    // Get predictions and uncertainty to return them
    std::vector<double> pred;
    pred.reserve(test_input.size());  // preallocate memory
    for (std::size_t i; i < m_tiles; i++)
    {
        pred.insert(pred.end(), prediction_tiles[i].get().begin(), prediction_tiles[i].get().end());
    }

    // Return computed data
    return hpx::async([pred]()
                      { return pred; }); */
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_uncertainty(const std::vector<double> &training_input,
                         const std::vector<double> &training_output,
                         const std::vector<double> &test_input,
                         int n_tiles,
                         int n_tile_size,
                         int m_tiles,
                         int m_tile_size,
                         int n_regressors,
                         gpxpy_hyper::SEKParams sek_params,
                         std::shared_ptr<gpxpy::Target> target)
{
    /* // declare tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prior_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prior_inter_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        t_cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prediction_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        prediction_uncertainty_tiles;

    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble prior covariance matrix vector
    prior_K_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prior_K_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_prior_covariance,
                                    "assemble_tiled"),
            i,
            i,
            m_tile_size,
            n_regressors,
            sek_params,
            test_input);
    }
    // Assemble MxN cross-covariance matrix vector
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    // Assemble NxM (transpose) cross-covariance matrix vector
    t_cross_covariance_tiles.resize(n_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(
                               &gen_tile_cross_covariance, "assemble_pred"),
                           i,
                           j,
                           m_tile_size,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           test_input,
                           training_input);

            t_cross_covariance_tiles[j * m_tiles + i] =
                hpx::dataflow(hpx::annotated_function(
                                  hpx::unwrapping(&gen_tile_cross_cov_T),
                                  "assemble_pred"),
                              m_tile_size,
                              n_tile_size,
                              cross_covariance_tiles[i * n_tiles + j]);
        }
    }
    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    prior_inter_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prior_inter_tiles[i] =
            hpx::async(hpx::annotated_function(&gen_tile_zeros_diag,
                                               "assemble_prior_inter"),
                       m_tile_size);
    }
    // Assemble placeholder for prediction
    prediction_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
            m_tile_size);
    }
    // Assemble placeholder for uncertainty
    prediction_uncertainty_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_uncertainty_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
            m_tile_size);
    }

    //////////////////////////////////////////////////////////////////////////////
    //// Compute Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
    //// Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    //// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(target.cublas_executors, K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
    // backward_solve_KK_tiled(K_tiles, cross_covariance_tiles, n_tile_size,
    // m_tile_size, n_tiles, m_tiles);

    //////////////////////////////////////////////////////////////////////////////
    //// Compute predictions
    prediction_tiled(target.cublas_executors, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    posterior_covariance_tiled(target.cublas_executors, t_cross_covariance_tiles, prior_inter_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    //// Compute predicition uncertainty
    prediction_uncertainty_tiled(target.cublas_executors, prior_K_tiles, prior_inter_tiles, prediction_uncertainty_tiles, m_tile_size, m_tiles);

    //// Get predictions and uncertainty to return them
    std::vector<double> pred_full;
    std::vector<double> pred_var_full;
    pred_full.reserve(test_input.size());      // preallocate memory
    pred_var_full.reserve(test_input.size());  // preallocate memory
    for (std::size_t i; i < m_tiles; i++)
    {
        pred_full.insert(pred_full.end(), prediction_tiles[i].get().begin(), prediction_tiles[i].get().end());
        pred_var_full.insert(pred_var_full.end(),
                             prediction_uncertainty_tiles[i].get().begin(),
                             prediction_uncertainty_tiles[i].get().end());
    }

    // Return computed data
    return hpx::async([pred_full, pred_var_full]()
                      {
            std::vector<std::vector<double>> result(2);
            result[0] = pred_full;
            result[1] = pred_var_full;
            return result; }); */
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_full_cov(const std::vector<double> &training_input,
                      const std::vector<double> &training_output,
                      const std::vector<double> &test_input,
                      int n_tiles,
                      int n_tile_size,
                      int m_tiles,
                      int m_tile_size,
                      int n_regressors,
                      gpxpy_hyper::SEKParams sek_params,
                      std::shared_ptr<gpxpy::Target> target)
{
    /* double hyperparameters[3];

    // declare tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prior_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prior_inter_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        t_cross_covariance_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> prediction_tiles;
    std::vector<hpx::shared_future<std::vector<double>>>
        prediction_uncertainty_tiles;

    //////////////////////////////////////////////////////////////////////////////
    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble prior covariance matrix vector
    prior_K_tiles.resize(m_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            prior_K_tiles[i * m_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_full_prior_covariance,
                                        "assemble_prior_tiled"),
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                test_input);

            if (i != j)
            {
                prior_K_tiles[j * m_tiles + i] = hpx::dataflow(
                    hpx::annotated_function(
                        hpx::unwrapping(&gen_tile_grad_l_trans),
                        "assemble_prior_tiled"),
                    m_tile_size,
                    prior_K_tiles[i * m_tiles + j]);
            }
        }
    }
    // Assemble MxN cross-covariance matrix vector
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    // Assemble NxM (transpose) cross-covariance matrix vector
    t_cross_covariance_tiles.resize(n_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(
                               &gen_tile_cross_covariance, "assemble_pred"),
                           i,
                           j,
                           m_tile_size,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           test_input,
                           training_input);

            t_cross_covariance_tiles[j * m_tiles + i] =
                hpx::dataflow(hpx::annotated_function(
                                  hpx::unwrapping(&gen_tile_cross_cov_T),
                                  "assemble_pred"),
                              m_tile_size,
                              n_tile_size,
                              cross_covariance_tiles[i * n_tiles + j]);
        }
    }
    // Assemble placeholder for prediction
    prediction_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
            m_tile_size);
    }
    // Assemble placeholder for uncertainty
    prediction_uncertainty_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_uncertainty_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
            m_tile_size);
    }
    //////////////////////////////////////////////////////////////////////////////
    //// Compute Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
    //// Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    //// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(target.cublas_executors, K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    //////////////////////////////////////////////////////////////////////////////
    //// Compute predictions
    prediction_tiled(target.cublas_executors, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);
    // posterior covariance matrix K_MxM - (K_MxN * K^-1_NxN) * K_NxM
    full_cov_tiled(target.cublas_executors, t_cross_covariance_tiles, prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
    //// Compute predicition uncertainty
    pred_uncer_tiled(target.cublas_executors, prior_K_tiles, prediction_uncertainty_tiles, m_tile_size, m_tiles);

    //// Get predictions and uncertainty to return them
    std::vector<double> pred;
    std::vector<double> pred_var;
    pred.reserve(test_input.size());      // preallocate memory
    pred_var.reserve(test_input.size());  // preallocate memory
    for (std::size_t i; i < m_tiles; i++)
    {
        pred.insert(pred.end(), prediction_tiles[i].get().begin(), prediction_tiles[i].get().end());
        pred_var.insert(pred_var.end(),
                        prediction_uncertainty_tiles[i].get().begin(),
                        prediction_uncertainty_tiles[i].get().end());
    }

    // Return computed data
    return hpx::async([pred, pred_var]()
                      {
            std::vector<std::vector<double>> result(2);
            result[0] = pred;
            result[1] = pred_var;
            return result; }); */
}

hpx::shared_future<double>
compute_loss(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             int n_tiles,
             int n_tile_size,
             int n_regressors,
             gpxpy_hyper::SEKParams sek_params,
             std::shared_ptr<gpxpy::Target> target)
{
    /* // declare data structures
    // tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    hpx::shared_future<double> loss_value;

    //////////////////////////////////////////////////////////////////////////////
    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }

    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    // Compute loss
    compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
    // Return loss
    return loss_value; */
}

hpx::shared_future<std::vector<double>>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gpxpy_hyper::SEKParams &sek_params,
         std::vector<bool> trainable_params,
         const gpxpy_hyper::AdamParams &adam_params,
         std::shared_ptr<gpxpy::Target> target)
{
    /* // declaretiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // data holder for computed loss values
    std::vector<double> losses;
    losses.resize(adam_params.opt_iter);
    //////////////////////////////////////////////////////////////////////////////
    // Assemble beta1_t and beta2_t
    beta1_T.resize(adam_params.opt_iter);
    for (int i = 0; i < adam_params.opt_iter; i++)
    {
        beta1_T[i] = hpx::async(
            hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, adam_params.beta1);
    }
    beta2_T.resize(adam_params.opt_iter);
    for (int i = 0; i < adam_params.opt_iter; i++)
    {
        beta2_T[i] = hpx::async(
            hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, adam_params.beta2);
    }
    // Assemble first and second momemnt vectors: m_T and v_T
    m_T.resize(3);
    v_T.resize(3);
    for (int i = 0; i < 3; i++)
    {
        m_T[i] = hpx::async(
            hpx::annotated_function(&gen_moment, "assemble_tiled"));
        v_T[i] = hpx::async(
            hpx::annotated_function(&gen_moment, "assemble_tiled"));
    }

    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_y"), i, n_tile_size, training_output);
    }

    // Perform optimization
    for (int iter = 0; iter < adam_params.opt_iter; iter++)
    {
        // Assemble covariance matrix vector, derivative of covariance
        // matrix vector w.r.t. to vertical lengthscale and derivative of
        // covariance matrix vector w.r.t. to lengthscale
        K_tiles.resize(n_tiles * n_tiles);
        grad_v_tiles.resize(n_tiles * n_tiles);
        grad_l_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j <= i; j++)
            {
                hpx::shared_future<std::vector<double>> cov_dists =
                    hpx::async(
                        hpx::annotated_function(&compute_cov_dist_vec,
                                                "assemble_cov_dist"),
                        i,
                        j,
                        n_tile_size,
                        n_regressors,
                        sek_params,
                        training_input);

                K_tiles[i * n_tiles + j] = hpx::dataflow(
                    hpx::annotated_function(
                        hpx::unwrapping(&gen_tile_covariance_opt),
                        "assemble_K"),
                    i,
                    j,
                    n_tile_size,
                    n_regressors,
                    sek_params,
                    cov_dists);

                grad_v_tiles[i * n_tiles + j] =
                    hpx::dataflow(hpx::annotated_function(
                                      hpx::unwrapping(&gen_tile_grad_v),
                                      "assemble_gradv"),
                                  i,
                                  j,
                                  n_tile_size,
                                  n_regressors,
                                  sek_params,
                                  cov_dists);

                grad_l_tiles[i * n_tiles + j] =
                    hpx::dataflow(hpx::annotated_function(
                                      hpx::unwrapping(&gen_tile_grad_l),
                                      "assemble_gradl"),
                                  i,
                                  j,
                                  n_tile_size,
                                  n_regressors,
                                  sek_params,
                                  cov_dists);

                if (i != j)
                {
                    grad_v_tiles[j * n_tiles + i] = hpx::dataflow(
                        hpx::annotated_function(
                            hpx::unwrapping(&gen_tile_grad_v_trans),
                            "assemble_gradv_t"),
                        n_tile_size,
                        grad_v_tiles[i * n_tiles + j]);

                    grad_l_tiles[j * n_tiles + i] = hpx::dataflow(
                        hpx::annotated_function(
                            hpx::unwrapping(&gen_tile_grad_l_trans),
                            "assemble_gradl_t"),
                        n_tile_size,
                        grad_l_tiles[i * n_tiles + j]);
                }
            }
        }
        // Assemble placeholder matrix for K^-1 * (I - y*y^T*K^-1)
        grad_K_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j < n_tiles; j++)
            {
                grad_K_tiles[i * n_tiles + j] =
                    hpx::async(hpx::annotated_function(&gen_tile_identity,
                                                       "assemble_tiled"),
                               i,
                               j,
                               n_tile_size);
            }
        }
        // Assemble alpha
        alpha_tiles.resize(n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            alpha_tiles[i] = hpx::async(
                hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
                n_tile_size);
        }
        // Assemble placeholder matrix for K^-1
        grad_I_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j < n_tiles; j++)
            {
                grad_I_tiles[i * n_tiles + j] = hpx::async(
                    hpx::annotated_function(&gen_tile_identity,
                                            "assemble_identity_matrix"),
                    i,
                    j,
                    n_tile_size);
            }
        }

        //////////////////////////////////////////////////////////////////////////////
        // Cholesky decomposition
        right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
        // Compute K^-1 through L*L^T*X = I
        forward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
        backward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

        // Triangular solve K_NxN * alpha = y
        // forward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size,
        // n_tiles); backward_solve_tiled(grad_I_tiles, alpha_tiles,
        // n_tile_size, n_tiles);

        // inv(K)*y
        compute_gemm_of_invK_y(target.cublas_executors, grad_I_tiles, y_tiles, alpha_tiles, n_tile_size, n_tiles);

        // Compute loss
        compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
        losses[iter] = loss_value.get();

        // Compute I-y*y^T*inv(K) -> NxN matrix
        // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles,
        // n_tile_size, n_tiles);

        // Compute K^-1 *(I - y*y^T*K^-1)
        // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
        // n_tile_size, n_tiles, n_tiles);
        // backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
        // n_tile_size, n_tiles, n_tiles);

        // Update the hyperparameters
        if (trainable_params[0])
        {  // lengthscale
            sek_params.lengthscale = update_lengthscale(grad_I_tiles, grad_l_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
        }
        if (trainable_params[1])
        {  // vertical_lengthscale
            sek_params.vertical_lengthscale = update_vertical_lengthscale(grad_I_tiles, grad_v_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
        }
        if (trainable_params[2])
        {  // noise_variance
            sek_params.noise_variance = update_noise_variance(grad_I_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);
        }
    }
    // Update hyperparameter attributes in Gaussian process model
    // Return losses
    return hpx::async([losses]()
                      { return losses; }); */
}

hpx::shared_future<double>
optimize_step(const std::vector<double> &training_input,
              const std::vector<double> &training_output,
              int n_tiles,
              int n_tile_size,
              int n_regressors,
              int iter,
              gpxpy_hyper::SEKParams &sek_params,
              std::vector<bool> trainable_params,
              gpxpy_hyper::AdamParams &adam_params,
              std::shared_ptr<gpxpy::Target> target)
{
    /* // declare tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // make shared future
    for (std::size_t i; i < 3; i++)
    {
        hpx::shared_future<double> m =
            hpx::make_ready_future(adam_params.M_T[i]);  //.share();
        m_T.push_back(m);
        hpx::shared_future<double> v =
            hpx::make_ready_future(adam_params.V_T[i]);  //.share();
        v_T.push_back(v);
    }

    // Assemble beta1_t and beta2_t
    beta1_T.resize(1);
    beta1_T[0] =
        hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"),
                   iter + 1,
                   adam_params.beta1);

    beta2_T.resize(1);
    beta2_T[0] =
        hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"),
                   iter + 1,
                   adam_params.beta1);

    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to vertical
    // lengthscale
    grad_v_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_v_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_grad_v, "assemble_tiled"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to lengthscale
    grad_l_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_l_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_grad_l, "assemble_tiled"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Assemble matrix that will be multiplied with derivates
    grad_K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_identity,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble placeholder matrix for K^-1
    grad_I_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_I_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_identity,
                                        "assemble_identity_matrix"),
                i,
                j,
                n_tile_size);
        }
    }

    // Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Compute K^-1 through L*L^T*X = I
    forward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
    backward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // Compute loss
    compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);

    // // Fill I-y*y^T*inv(K)
    // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size,
    // n_tiles);

    // // Compute K^-1 * (I-y*y^T*K^-1)
    // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
    // n_tile_size, n_tiles, n_tiles); backward_solve_tiled_matrix(K_tiles,
    // grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // Update the hyperparameters
    if (trainable_params[0])
    {  // lengthscale
        sek_params.lengthscale = update_lengthscale(grad_I_tiles, grad_l_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
    }
    if (trainable_params[1])
    {  // vertical_lengthscale
        sek_params.vertical_lengthscale = update_vertical_lengthscale(grad_I_tiles, grad_v_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
    }
    if (trainable_params[2])
    {  // noise_variance
        sek_params.noise_variance = update_noise_variance(grad_I_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);
    }

    // Update hyperparameter attributes (first and second moment) for Adam
    for (std::size_t i; i < 3; i++)
    {
        adam_params.M_T[i] = m_T[i].get();
        adam_params.V_T[i] = v_T[i].get();
    }

    // Return loss value
    double loss = loss_value.get();
    return hpx::async([loss]()
                      { return loss; }); */
}

hpx::shared_future<std::vector<std::vector<double>>>
cholesky(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gpxpy_hyper::SEKParams sek_params,
         std::shared_ptr<gpxpy::Target> target)
{
    /* hpx::cuda::experimental::enable_user_polling poll("default");
    // Tiled future data structure is matrix represented as vector of tiles.
    // Tiles are represented as vector, each wrapped in a shared_future.
    std::vector<hpx::shared_future<double *>> K_tiles;

    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);

    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }

    // Calculate Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);

    std::vector<std::vector<double>> result(n_tiles * n_tiles);

    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            hpx::cuda::experimental::check_cuda_error(
                cudaMemcpyAsync(result[i * n_tiles + j], K_tiles[i * n_tiles + j].get(), n_tile_size * n_tile_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    return hpx::make_ready_future(result);

    // Get & return predictions and uncertainty
    // std::vector<std::vector<double>> result(n_tiles * n_tiles);
    // for (std::size_t i = 0; i < n_tiles; i++)
    // {
    //     for (std::size_t j = 0; j <= i; j++)
    //     {
    //         result[i * n_tiles + j] = K_tiles[i * n_tiles + j].get();
    //     }
    // }
    // return hpx::async([result]()
    //                   { return result; });
    */
}

}  // end of namespace gpu

#endif  // end of GPX_WITH_CUDA
