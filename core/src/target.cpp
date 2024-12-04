#include "../include/target.hpp"

#include <iostream>

namespace gpxpy
{

bool CPU::is_cpu() { return false; }

bool CPU::is_gpu() { return true; }

CUDA_GPU::CUDA_GPU(int id, int n_streams) :
    id(id),
    n_streams(n_streams)
{
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (id >= deviceCount)
    {
        throw std::runtime_error("Requested GPU device is not available.");
    }
#else
    throw not_compiled_with_cuda_exception();
#endif
}

bool CUDA_GPU::is_cpu() { return false; }

bool CUDA_GPU::is_gpu() { return true; }

CPU get_cpu() { return CPU(); }

CUDA_GPU get_gpu(int id, int n_executors)
{
#ifdef GPXPY_WITH_CUDA
    return CUDA_GPU(id, n_executors);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

CUDA_GPU get_gpu()
{
#ifdef GPXPY_WITH_CUDA
    return CUDA_GPU(0, 1);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

void print_available_gpus()
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
    std::cout
        << "CUDA is not available - There are no GPUs available. You can only "
           "`get_cpu()` to utilize the CPU for computation."
        << std::endl;
#endif
}

int gpu_count()
{
#ifdef GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
#else
    std::cout
        << "CUDA is not available - There are no GPUs available. You can only "
           "`get_cpu()` to utilize the CPU for computation."
        << std::endl;
    return 0;
#endif
}

}  // namespace gpxpy
