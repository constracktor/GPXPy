#include "../include/gp_optimizer.hpp"

#include "../include/adapter_mkl.hpp"
#include <numeric>

namespace gpxpy_hyper
{

AdamParams::AdamParams(double lr,
                       double b1,
                       double b2,
                       double eps,
                       int opt_i,
                       std::vector<double> M_T_init,
                       std::vector<double> V_T_init) :
    learning_rate(lr),
    beta1(b1),
    beta2(b2),
    epsilon(eps),
    opt_iter(opt_i),
    M_T(M_T_init),
    V_T(V_T_init)
{ }

std::string AdamParams::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    // clang-format off
        oss << "Hyperparameters: [learning_rate=" << learning_rate
                            << ", beta1=" << beta1
                            << ", beta2=" << beta2
                            << ", epsilon=" << epsilon
                            << ", opt_iter=" << opt_iter << "]";
    // clang-format on

    return oss.str();
}

}  // namespace gpxpy_hyper

double to_constrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(1.0 + exp(parameter)) + 1e-6;
    }
    else
    {
        return log(1.0 + exp(parameter));
    }
}

double to_unconstrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(exp(parameter - 1e-6) - 1.0);
    }
    else
    {
        return log(exp(parameter) - 1.0);
    }
}

double compute_sigmoid(const double parameter)
{
    return 1.0 / (1.0 + exp(-parameter));
}

double compute_covariance_dist_func(std::size_t i_global,
                                    std::size_t j_global,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams sek_params,
                                    const std::vector<double> &i_input,
                                    const std::vector<double> &j_input)
{
    // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
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

    return -1.0 / (2.0 * pow(sek_params.lengthscale, 2.0)) * distance;
}

std::vector<double> compute_cov_dist_vec(std::size_t row,
                                         std::size_t col,
                                         std::size_t N,
                                         std::size_t n_regressors,
                                         gpxpy_hyper::SEKParams sek_params,
                                         const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double cov_dist;
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
            cov_dist =
                compute_covariance_dist_func(i_global, j_global, n_regressors, sek_params, input, input);
            tile[i * N + j] = cov_dist;
        }
    }
    return std::move(tile);
}

std::vector<double>
gen_tile_covariance_opt(std::size_t row,
                        std::size_t col,
                        std::size_t N,
                        std::size_t n_regressors,
                        gpxpy_hyper::SEKParams sek_params,
                        const std::vector<double> &cov_dists)
{
    std::size_t i_global, j_global;
    double covariance;
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
            covariance = sek_params.vertical_lengthscale * exp(cov_dists[i * N + j]);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance += sek_params.noise_variance;
            }
            tile[i * N + j] = covariance;
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_grad_v(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams &sek_params,
                                    const std::vector<double> &cov_dists)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    double hyperparam_der =
        compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute covariance function
            tile[i * N + j] = exp(cov_dists[i * N + j]) * hyperparam_der;
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_grad_l(std::size_t row,
                                    std::size_t col,
                                    std::size_t N,
                                    std::size_t n_regressors,
                                    gpxpy_hyper::SEKParams sek_params,
                                    const std::vector<double> &cov_dists)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    double hyperparam_der =
        compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute covariance function
            tile[i * N + j] = -2.0 * (sek_params.vertical_lengthscale / sek_params.lengthscale) * cov_dists[i * N + j] * exp(cov_dists[i * N + j]) * hyperparam_der;
        }
    }
    return std::move(tile);
}

std::vector<double>
gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile)
{
    std::vector<double> transposed;
    transposed.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // Mapping (i, j) in the original matrix to (j, i) in the transposed
            // matrix
            transposed[j * N + i] = grad_l_tile[i * N + j];
        }
    }
    return std::move(transposed);
}

std::vector<double>
gen_tile_grad_l_trans(std::size_t N, const std::vector<double> &grad_l_tile)
{
    std::vector<double> transposed;
    transposed.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // Mapping (i, j) in the original matrix to (j, i) in the transposed
            // matrix
            transposed[j * N + i] = grad_l_tile[i * N + j];
        }
    }
    return std::move(transposed);
}

double gen_beta_T(int t, double beta)
{
    return pow(beta, t);
}

double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N)
{
    double l = 0.0;
    l += dot(y_tile, alpha_tile, N);
    for (std::size_t i = 0; i < N; i++)
    {
        // Add the squared difference to the error
        l += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]);
    }
    return l;
}

double
add_losses(const std::vector<double> &losses, std::size_t N, std::size_t n)
{
    double l = 0.0;
    for (std::size_t i = 0; i < n; i++)
    {
        // Add the squared difference to the error
        l += losses[i];
    }
    l += N * n * log(2.0 * M_PI);
    return 0.5 * l / (N * n);
}

double compute_gradient(const double &grad_l,
                        const double &grad_r,
                        std::size_t N,
                        std::size_t n_tiles)
{
    double grad = 0.0;
    grad = 1.0 / (2.0 * N * n_tiles) * (grad_l - grad_r);

    return std::move(grad);
}

double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              double noise_variance,
                              std::size_t N,
                              std::size_t n_tiles)
{
    // Initialize tile
    double trace = 0.0;
    double hyperparam_der =
        compute_sigmoid(to_unconstrained(noise_variance, true));
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        auto tile = ft_tiles[d * n_tiles + d];
        for (std::size_t i = 0; i < N; ++i)
        {
            trace += (tile[i * N + i] * hyperparam_der);
        }
    }
    trace = 1.0 / (2.0 * N * n_tiles) * trace;
    return std::move(trace);
}

double
update_first_moment(const double &gradient, double m_T, const double &beta_1)
{
    return beta_1 * m_T + (1.0 - beta_1) * gradient;
}

double
update_second_moment(const double &gradient, double v_T, const double &beta_2)
{
    return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
}

hpx::shared_future<double> update_param(const double unconstrained_hyperparam,
                                        gpxpy_hyper::SEKParams sek_params,
                                        gpxpy_hyper::AdamParams adam_params,
                                        double m_T,
                                        double v_T,
                                        const std::vector<double> beta1_T,
                                        const std::vector<double> beta2_T,
                                        int iter)
{
    double alpha_T =
        sek_params.noise_variance * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
    return hpx::make_ready_future(unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + adam_params.epsilon));
}

std::vector<double>
gen_tile_identity(std::size_t row, std::size_t col, std::size_t N)
{
    std::size_t i_global, j_global;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    std::fill(tile.begin(), tile.end(), 0.0);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = i; j <= i; j++)
        {
            j_global = N * col + j;
            if (i_global == j_global)
            {
                tile[i * N + j] = 1.0;
            }
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_zeros_diag(std::size_t N)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    std::fill(tile.begin(), tile.end(), 0.0);
    return std::move(tile);
}

double gen_moment()
{
    double z = 0.0;
    return z;
}

double sum_gradleft(const std::vector<double> &diagonal, double grad)
{
    grad += std::reduce(diagonal.begin(), diagonal.end());
    return grad;
}

double sum_gradright(const std::vector<double> &inter_alpha,
                     const std::vector<double> &alpha,
                     double grad,
                     std::size_t N)
{
    grad += dot(inter_alpha, alpha, N);
    return grad;
}

double sum_noise_gradleft(const std::vector<double> &ft_invK,
                          double grad,
                          gpxpy_hyper::SEKParams sek_params,
                          std::size_t N,
                          std::size_t n_tiles)
{
    double noise_der =
        compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    for (std::size_t i = 0; i < N; ++i)
    {
        grad += (ft_invK[i * N + i] * noise_der);
    }
    return std::move(grad);
}

double sum_noise_gradright(const std::vector<double> &alpha,
                           double grad,
                           gpxpy_hyper::SEKParams sek_params,
                           std::size_t N)
{
    double noise_der =
        compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    grad += (noise_der * dot(alpha, alpha, N));
    return grad;
}
