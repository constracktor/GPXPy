#include "../include/gpxpy_c.hpp"

#include "../include/gp_functions.hpp"
#include "../include/target.hpp"
#include "../include/utils_c.hpp"
#include <cstdio>
#include <hpx/future.hpp>
#include <iomanip>
#include <sstream>

#ifdef GPXPY_USE_CUDA
    #include <cuda_runtime.h>
    #include <hpx/modules/async_cuda.hpp>
#endif

// namespace for GPXPy library entities
namespace gpxpy
{

GP_data::GP_data(const std::string &f_path, int n) :
    n_samples(n),
    file_path(f_path)
{
    data = utils::load_data(f_path, n);
}

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       gpxpy_hyper::SEKParams sek_params,
       int n_r,
       std::vector<bool> trainable_bool,
       Target device) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    sek_params(sek_params),
    n_regressors(n_r),
    trainable_params(trainable_bool),
    device(device)
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       double lengthscale,
       double vertical_lengthscale,
       double noise_variance,
       int n_r,
       std::vector<bool> trainable_bool,
       Target device) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    sek_params(lengthscale, vertical_lengthscale, noise_variance),
    n_regressors(n_r),
    trainable_params(trainable_bool),
    device(device)
{ }

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << sek_params.lengthscale
        << ", vertical_lengthscale=" << sek_params.vertical_lengthscale
        << ", noise_variance=" << sek_params.noise_variance
        << ", n_regressors=" << n_regressors
        << ", trainable_params l=" << trainable_params[0]
        << ", trainable_params v=" << trainable_params[1]
        << ", trainable_params n=" << trainable_params[2] << "]";
    return oss.str();
}

std::vector<double> GP::get_training_input() const
{
    return _training_input;
}

std::vector<double> GP::get_training_output() const
{
    return _training_output;
}

std::vector<double> GP::predict(const std::vector<double> &test_data,
                                int m_tiles,
                                int m_tile_size)
{
    std::vector<double> result;
    hpx::run_as_hpx_thread([this, &result, &test_data, m_tiles, m_tile_size]()
                           { result =
                                 predict_on_target(_training_input, _training_output, test_data, _n_tiles, _n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, device)
                                     .get(); });
    return result;
}

std::vector<std::vector<double>> GP::predict_with_uncertainty(
    const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    std::vector<std::vector<double>> result;
    hpx::run_as_hpx_thread(
        [this, &result, &test_input, m_tiles, m_tile_size]()
        {
            result = predict_with_uncertainty_on_target(
                         _training_input, _training_output, test_input, _n_tiles, _n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, device)
                         .get();
        });
    return result;
}

std::vector<std::vector<double>> GP::predict_with_full_cov(
    const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    std::vector<std::vector<double>> result;
    hpx::run_as_hpx_thread(
        [this, &result, &test_input, m_tiles, m_tile_size]()
        {
            result = predict_with_full_cov_on_target(
                         _training_input, _training_output, test_input, _n_tiles, _n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, device)
                         .get();
        });
    return result;
}

std::vector<double>
GP::optimize(const gpxpy_hyper::AdamParams &adam_hyperparams)
{
    std::vector<double> losses;
    hpx::run_as_hpx_thread([this, &losses, &adam_hyperparams]()
                           { losses =
                                 optimize_on_target(_training_input, _training_output, _n_tiles, _n_tile_size, n_regressors, sek_params, trainable_params, adam_hyperparams, device)
                                     .get(); });
    return losses;
}

double GP::optimize_step(gpxpy_hyper::AdamParams &adam_params, int iter)
{
    double loss;
    hpx::run_as_hpx_thread([this, &loss, &adam_params, iter]()
                           { loss = optimize_step_on_target(_training_input, _training_output, _n_tiles, _n_tile_size, n_regressors, iter, sek_params, trainable_params, adam_params, device)
                                        .get(); });
    return loss;
}

double GP::calculate_loss()
{
    double loss;
    hpx::run_as_hpx_thread([this, &loss]()
                           { loss = compute_loss_on_target(_training_input, _training_output, _n_tiles, _n_tile_size, n_regressors, sek_params, device)
                                        .get(); });
    return loss;
}

std::vector<std::vector<double>> GP::cholesky()
{
    std::vector<std::vector<double>> result;
    hpx::run_as_hpx_thread([this, &result]()
                           { result =
                                 cholesky_on_target(_training_input, _training_output, _n_tiles, _n_tile_size, n_regressors, sek_params, device)
                                     .get(); });
    return result;
}

}  // namespace gpxpy
