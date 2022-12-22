#ifndef CUBLAS_ADAPTER_H_INCLUDED
#define CUBLAS_ADAPTER_H_INCLUDED

#include <hpx/local/future.hpp>
#include <hpx/modules/async_cuda.hpp>

//C = C - A * B^T
template <typename T>
hpx::shared_future<std::vector<T>> gemm_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                               hpx::shared_future<std::vector<T>> A,
                                               hpx::shared_future<std::vector<T>> B,
                                               hpx::shared_future<std::vector<T>> C,
                                               std::size_t N);

//  A = A - B * B^T
template <typename T>
hpx::shared_future<std::vector<T>> syrk_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                               hpx::shared_future<std::vector<T>> A,
                                               hpx::shared_future<std::vector<T>> B,
                                               std::size_t N);

#endif
