#include "../include/gp_functions.hpp"

#include "../include/gp_algorithms_cpu.hpp"
#include "../include/target.hpp"
#include <cmath>
#include <vector>

// for GPU algorithms
#ifdef GPXPY_WITH_CUDA
    #include "../include/gp_algorithms_gpu.hpp"
    #include "../include/tiled_algorithms_gpu.hpp"
#endif

hpx::shared_future<std::vector<double>>
predict_on_target(const std::vector<double> &training_input,
                  const std::vector<double> &training_output,
                  const std::vector<double> &test_input,
                  int n_tiles,
                  int n_tile_size,
                  int m_tiles,
                  int m_tile_size,
                  int n_regressors,
                  gpxpy_hyper::SEKParams sek_params,
                  gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::predict(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, target);
    }
    else
    {
        return cpu::predict(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
    }
#else
    return cpu::predict(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
#endif
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_uncertainty_on_target(const std::vector<double> &training_input,
                                   const std::vector<double> &training_output,
                                   const std::vector<double> &test_input,
                                   int n_tiles,
                                   int n_tile_size,
                                   int m_tiles,
                                   int m_tile_size,
                                   int n_regressors,
                                   gpxpy_hyper::SEKParams sek_params,
                                   gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::predict_with_uncertainty(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, target);
    }
    else
    {
        return cpu::predict_with_uncertainty(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
    }
#else
    return cpu::predict_with_uncertainty(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
#endif
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_full_cov_on_target(const std::vector<double> &training_input,
                                const std::vector<double> &training_output,
                                const std::vector<double> &test_input,
                                int n_tiles,
                                int n_tile_size,
                                int m_tiles,
                                int m_tile_size,
                                int n_regressors,
                                gpxpy_hyper::SEKParams sek_params,
                                gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::predict_with_full_cov(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params, target);
    }
    else
    {
        return cpu::predict_with_full_cov(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
    }
#else
    return cpu::predict_with_full_cov(training_input, training_output, test_input, n_tiles, n_tile_size, m_tiles, m_tile_size, n_regressors, sek_params);
#endif
}

hpx::shared_future<double>
compute_loss_on_target(const std::vector<double> &training_input,
                       const std::vector<double> &training_output,
                       int n_tiles,
                       int n_tile_size,
                       int n_regressors,
                       gpxpy_hyper::SEKParams sek_params,
                       gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::compute_loss(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params, target);
    }
    else
    {
        return cpu::compute_loss(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
    }
#else
    return cpu::compute_loss(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
#endif
}

hpx::shared_future<std::vector<double>>
optimize_on_target(const std::vector<double> &training_input,
                   const std::vector<double> &training_output,
                   int n_tiles,
                   int n_tile_size,
                   int n_regressors,
                   gpxpy_hyper::SEKParams &sek_params,
                   std::vector<bool> trainable_params,
                   const gpxpy_hyper::AdamParams &adam_params,
                   gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::optimize(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params, trainable_params, adam_params, target);
    }
    else
    {
        return cpu::optimize(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params, trainable_params, adam_params);
    }
#else
    return cpu::optimize(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params, trainable_params, adam_params);
#endif
}

hpx::shared_future<double>
optimize_step_on_target(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors,
                        int iter,
                        gpxpy_hyper::SEKParams &sek_params,
                        std::vector<bool> trainable_params,
                        gpxpy_hyper::AdamParams &adam_params,
                        gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::optimize_step(training_input, training_output, n_tiles, n_tile_size, n_regressors, iter, sek_params, trainable_params, adam_params, target);
    }
    else
    {
        return cpu::optimize_step(training_input, training_output, n_tiles, n_tile_size, n_regressors, iter, sek_params, trainable_params, adam_params);
    }
#else
    return cpu::optimize_step(training_input, training_output, n_tiles, n_tile_size, n_regressors, iter, sek_params, trainable_params, adam_params);
#endif
}

hpx::shared_future<std::vector<std::vector<double>>>
cholesky_on_target(const std::vector<double> &training_input,
                   const std::vector<double> &training_output,
                   int n_tiles,
                   int n_tile_size,
                   int n_regressors,
                   gpxpy_hyper::SEKParams sek_params,
                   gpxpy::Target &target)
{
#ifdef GPXPY_WITH_CUDA
    if (target.is_gpu())
    {
        return gpu::cholesky(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params, target);
    }
    else
    {
        return cpu::cholesky(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
    }
#else
    return cpu::cholesky(training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
#endif
}
