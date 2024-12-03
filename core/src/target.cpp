#include "../include/target.hpp"

#include "../include/cuda_utils.hpp"

#ifdef GPXPY_WITH_CUDA
    #include <cuda_runtime.h>
    #include <iostream>
#endif

namespace gpxpy
{

Target::Target(std::string type, int id, int n_executors) :
    id(id),
    n_executors(n_executors)
{
    // lowercase type string
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    if (type == "cpu")
    {
        this->type = TargetType::CPU;
    }
    else if (type == "gpu")
    {
        this->type = TargetType::GPU;
    }
    else
    {
        throw std::runtime_error("Invalid target type.");
    }
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (this->type == TargetType::GPU && id >= deviceCount)
    {
        throw std::runtime_error("Requested GPU device is not available.");
    }
    for (int i = 0; i < n_executors; ++i)
    {
        cublas_executors.emplace_back(
            id, CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
    }
#endif
}

Target::Target(Target::TargetType type, int id, int n_executors) :
    type(type),
    id(id),
    n_executors(n_executors)
{
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (type == TargetType::GPU && id >= deviceCount)
    {
        throw std::runtime_error("Requested GPU device is not available.");
    }
    for (int i = 0; i < n_executors; ++i)
    {
        cublas_executors.emplace_back(
            id, CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
    }
#endif
}

void Target::print_available_gpus()
{
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // clang-format off
        std::cout
            << "Device " << i << ": " << deviceProp.name << "\n"
            << "  Total Global Memory: " << deviceProp.totalGlobalMem << "\n"
            << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << "\n"
            << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n"
            << "  Total Constant Memory: " << deviceProp.totalConstMem << "\n"
            << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n"
            << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << "\n"
            << "  Clock Rate: " << deviceProp.clockRate << " kHz\n"
            << "  Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz\n"
            << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        // clang-format on
    }
#else
    std::cerr
        << "CUDA is not available - There are no GPUs available. You only "
           "can use `get_cpu()` to utilize the CPU for computation."
        << std::endl;
#endif
}

int Target::gpu_count()
{
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
#else
    throw not_compiled_with_cuda_exception();
#endif
}

Target Target::get_cpu() { return Target(TargetType::CPU, 0, 1); }

Target Target::get_gpu(int id, int n_executors)
{
#ifdef GPXPY_WITH_CUDA
    return Target(TargetType::GPU, id, n_executors);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

Target Target::get_gpu()
{
#ifdef GPXPY_WITH_CUDA
    return Target(TargetType::GPU, 0, 1);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

}  // namespace gpxpy
