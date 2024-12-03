#ifndef TARGET_H
#define TARGET_H

#include <hpx/async_cuda/cublas_executor.hpp>

namespace gpxpy
{
class Target
{
  public:
    enum class TargetType { CPU,
                            GPU };

    /**
     * @brief Type of target: TargetType::CPU or TargetType::GPU.
     */
    TargetType type;

    /**
     * @brief ID of target. For CPU, ID is 0. For GPU, ID is 0, 1, 2, ...
     */
    int id;

    /**
     * @brief Number of executors on target: n for GPU, (1 for CPU)
     *
     * Executors are used for parallel computation and memory transfer.
     */
    int n_executors;

    std::vector<hpx::cuda::experimental::cublas_executor> cublas_executors;

    Target(std::string type, int id, int n_executors);

    /**
     * @brief Lists available GPUs with their properties.
     */
    static void print_available_gpus();

    /**
     * @brief Returns number of available GPUs.
     */
    static int gpu_count();

    /**
     * @brief Returns handle for CPU target.
     */
    static Target get_cpu();

    /**
     * @brief Returns handle for GPU target.
     *
     * @param id ID of GPU.
     * @param n_executors Number of executors to be created on GPU.
     */
    static Target get_gpu(int id, int n_executors);

    /**
     * @brief Returns handle for GPU target with ID 0.
     *
     * Uses only one executor, so single-threaded GPU execution.
     */
    static Target get_gpu();

    /**
     * @brief Returns true if target is CPU.
     */
    inline bool is_cpu() { return this->type == TargetType::CPU; }

    /**
     * @brief Returns true if target is GPU.
     */
    inline bool is_gpu() { return this->type == TargetType::GPU; }

  private:
    /**
     * @brief Constructor creates handle for target.
     *
     * @param type Type of target: TargetType::CPU or TargetType::GPU.
     * @param id ID of target. For CPU, ID is 0. For GPU, ID is 0, 1, 2, ...
     * @param n_executors Number of executors on target: 1 for CPU, n for
     * GPU.
     */
    Target(TargetType type, int id, int n_executors);

    Target() = delete;
};

// #ifdef GPXPY_WITH_CUDA
template <typename T>
class DualVector
{
  private:
    std::vector<T> h_data;
    T *d_data;
    std::size_t size;
    Target &target;

  public:
    DualVector(std::size_t size, Target &target) :
        size(size),
        target(target)
    {
        if (target.is_cpu())
        {
            h_data.resize(size);
        }
        else if (target.is_gpu())
        {
            h_data.resize(size);
            cudaMalloc(&d_data, size * sizeof(T));
        }
    }

    ~DualVector()
    {
        if (target.is_gpu())
        {
            cudaFree(d_data);
        }
    }

    /**
     * @brief Set data on target.
     *
     * If target is GPU, data is copied from host to device.
     *
     * @param data Data to be set on target.
     */
    void set(std::vector<T> &data)
    {
        if (target.is_cpu())
        {
            h_data = data;
        }
        else if (target.is_gpu())
        {
            cudaMemcpy(d_data, data.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    /**
     * @brief Get data from target.
     *
     * If device is GPU, data is copied from device to host.
     */
    std::vector<T> get()
    {
        if (target.is_cpu())
        {
            return h_data;
        }
        else if (target.is_gpu())
        {
            std::vector<T> data(size);
            cudaMemcpy(data.data(), d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
            return data;
        }
    }

    /**
     * @brief Resize data on target.
     */
    void resize(std::size_t size)
    {
        if (target.is_cpu())
        {
            h_data.resize(size);
        }
        else if (target.is_gpu())
        {
            cudaFree(d_data);
            cudaMalloc(&d_data, size * sizeof(T));
        }
    }
};

// #endif

}  // end of namespace gpxpy

#endif  // end of TARGET_H
