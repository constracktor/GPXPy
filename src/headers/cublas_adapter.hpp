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
                                               std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  T *d_A, *d_B, *d_C;
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_A, size * sizeof(T)));
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_B, size * sizeof(T)));
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_C, size * sizeof(T)));
  // copy data from host to device when ready
  std::vector<hpx::future<void>> copy_host_to_device;
  std::vector<T> h_A = A.get();
  copy_host_to_device.push_back(hpx::async(cublas, cudaMemcpyAsync, d_A, h_A.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  std::vector<T> h_B = B.get();
  copy_host_to_device.push_back(hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  std::vector<T> h_C = C.get();
  copy_host_to_device.push_back(hpx::async(cublas, cudaMemcpyAsync, d_C, h_C.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  hpx::wait_all(copy_host_to_device);
  // compute GEMM (C = C - A * B^T) on device
  // note cublas uses column major ordering : (A * B^T)^T = B * A^T
  auto compute_on_device = hpx::async(cublas, cublasSgemm, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
  compute_on_device.get();
  // copy the result back to the host
  auto copy_device_to_host = hpx::async(cublas, cudaMemcpyAsync, h_C.data(), d_C, size * sizeof(T), cudaMemcpyDeviceToHost);
  copy_device_to_host.get();
  // free memory
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
  // return future
  return hpx::make_ready_future(h_C);
}

//  A = A - B * B^T
template <typename T>
hpx::shared_future<std::vector<T>> syrk_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                               hpx::shared_future<std::vector<T>> A,
                                               hpx::shared_future<std::vector<T>> B,
                                               std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  T *d_A, *d_B;
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_A, size * sizeof(T)));
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_B, size * sizeof(T)));
  // copy data from host to device when ready
  std::vector<hpx::future<void>> copy_host_to_device;
  std::vector<T> h_A = A.get();
  copy_host_to_device.push_back(hpx::async(cublas, cudaMemcpyAsync, d_A, h_A.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  std::vector<T> h_B = B.get();
  copy_host_to_device.push_back(hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  hpx::wait_all(copy_host_to_device);
  // compute SYRK (A = A - B * B^T) on device
  auto compute_on_device = hpx::async(cublas, cublasSgemm, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_B, N, &beta, d_A, N);
  compute_on_device.get();
  // copy the result back to the host
  auto copy_device_to_host = hpx::async(cublas, cudaMemcpyAsync, h_A.data(), d_A, size * sizeof(T), cudaMemcpyDeviceToHost);
  copy_device_to_host.get();
  // free memory
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
  // return future
  return hpx::make_ready_future(h_A);
}
#endif
