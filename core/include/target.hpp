#ifndef TARGET_H
#define TARGET_H

#ifdef GPXPY_WITH_CUDA
    #include <cuda_runtime.h>
    #include <hpx/async_cuda/cublas_executor.hpp>
#endif

namespace gpxpy
{

struct Target
{
    /**
     * @brief Returns true if target is CPU.
     */
    virtual bool is_cpu() = 0;

    /**
     * @brief Returns true if target is GPU.
     */
    virtual bool is_gpu() = 0;

    virtual ~Target() {};

  protected:
    Target() = default;
};

struct CPU : public Target
{
  public:
    /**
     * @brief Returns CPU target.
     */
    CPU();

    /**
     * @brief Returns true because target is CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns false because CPU target is not GPU.
     */
    bool is_gpu() override;
};

struct CUDA_GPU : public Target
{
    int id;
    int n_streams;

    /**
     * TODO: documentation
     */
    CUDA_GPU(int id, int n_streams);

    bool is_cpu() override;

    bool is_gpu() override;
};

/**
 * @brief Returns handle for CPU target.
 */
CPU get_cpu();

/**
 * @brief Returns handle for GPU target.
 *
 * @param id ID of GPU.
 * @param n_streams Number of streams to be created on GPU.
 */
CUDA_GPU get_gpu(int id, int n_streams);

/**
 * @brief Returns handle for GPU target with ID 0.
 *
 * Uses only one stream, so single-threaded GPU execution.
 */
CUDA_GPU get_gpu();

/**
 * @brief Lists available GPUs with their properties.
 */
void print_available_gpus();

/**
 * @brief Returns number of available GPUs.
 */
int gpu_count();

}  // end of namespace gpxpy

#endif  // end of TARGET_H
