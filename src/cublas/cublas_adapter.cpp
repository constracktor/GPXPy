#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/async_cuda.hpp>

#include <hpx/local/chrono.hpp>

#include <iostream>
#include <vector>

#define CALC_TYPE float

template <typename T>
// std::vector<T> gemm_cublas(hpx::cuda::experimental::cublas_executor& cublas,
//                                     std::vector<T> h_A,
//                                     std::vector<T> h_B,
//                                     std::vector<T> h_C,
//                                     std::size_t N)
// {
// std::vector<T> gemm_cublas(
//                                     std::vector<T> h_A,
//                                     std::vector<T> h_B,
//                                     std::vector<T> h_C,
//                                     std::size_t N)
// {

hpx::shared_future<std::vector<T>> gemm_cublas(hpx::shared_future<std::vector<T>> A,
                                    hpx::shared_future<std::vector<T>> B,
                                    hpx::shared_future<std::vector<T>> C,
                                    std::size_t N)
{
  // not elegant
  std::vector<T> h_A = A.get();
  std::vector<T> h_B = B.get();
  std::vector<T> h_C = C.get();
  //hpx::when_all(result, done_sorting)

  hpx::cuda::experimental::enable_user_polling poll("default");
  std::size_t device = 0;
  hpx::cuda::experimental::cublas_executor cublas(device,
      CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});

  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // Allocate device memory
  T *d_A, *d_B, *d_C;
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_A, size * sizeof(T)));
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_B, size * sizeof(T)));
  hpx::cuda::experimental::check_cuda_error(cudaMalloc((void**) &d_C, size * sizeof(T)));
  // copy data from host to device
  auto copy_future1 = hpx::async(cublas, cudaMemcpyAsync, d_A, h_A.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  auto copy_future2 = hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  auto copy_future = hpx::async(cublas, cudaMemcpyAsync, d_C, h_C.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  // not elegant
  copy_future1.get();
  copy_future2.get();
  copy_future.get();
  // compute GEMM (C = C - A * B^T) on device
  // note cublas is column major ordering : (A * B^T)^T = B * A^T
  auto fut = hpx::async(cublas, cublasSgemm, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
  // wait until the operation completes
  fut.get();
  // copy the result back to the host
  auto copy_finished = hpx::async(cublas, cudaMemcpyAsync, h_C.data(), d_C, size * sizeof(T), cudaMemcpyDeviceToHost);
  // just wait for the device->host copy to complete if it hasn't already
  copy_finished.get();
  // free memory
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
  ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
  // return future
  return hpx::make_ready_future(h_C);
}
//
// // -------------------------------------------------------------------------
// int hpx_main(hpx::program_options::variables_map& vm)
// {
//     std::size_t device = vm["device"].as<std::size_t>();
//     std::size_t sizeMult = vm["sizemult"].as<std::size_t>();
//     //int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;
//     int block_size = 8;
//     unsigned int N = 2;//1 * block_size * sizeMult; // 2*32*10=640
//     // Allocate host memory for matrices A and B
//     unsigned int size = 4;//N*N;
//     // Fill A,B and C
//     std::vector<CALC_TYPE> h_A(size);
//     std::vector<CALC_TYPE> h_B(size);
//     std::vector<CALC_TYPE> h_C(size);
//     h_C[0]=1;h_C[1]=2;h_C[2]=3;h_C[3]=4;
//     h_A[0]=2;h_A[1]=1;h_A[2]=2;h_A[3]=3;
//     h_B[0]=3;h_B[1]=2;h_B[2]=4;h_B[3]=1;
//
//     // // GPU stuff
//     // install cuda future polling handler
//     // hpx::cuda::experimental::enable_user_polling poll("default");
//     // // use a larger block size for Fermi and above, query default cuda target properties
//     // hpx::cuda::experimental::target target(device);
//     // //int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;
//     //
//     // std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
//     //           << target.native_handle().processor_name() << "\" "
//     //           << "with compute capability "
//     //           << target.native_handle().processor_family() << "\n";
//     //
//     hpx::chrono::high_resolution_timer t1;
//     // install cuda future polling handler
//     // hpx::cuda::experimental::enable_user_polling poll("default");
//     // hpx::cuda::experimental::cublas_executor cublas(device,
//     //     CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
//     double us1 = t1.elapsed_microseconds();
//     std::cout << "Build: " << us1
//               << std::endl;
//
//     hpx::chrono::high_resolution_timer t;
//     size_t iterations = 1;
//     hpx::future<std::vector<CALC_TYPE>> C;
//     for (size_t i = 0; i < iterations; i++)
//     {
//       C = gemm_cublas<float>(hpx::make_ready_future(h_A), hpx::make_ready_future(h_B), hpx::make_ready_future(h_C), N);
//       //h_C = gemm_cublas<float>(h_A, h_B, h_C, N);
//       //h_C = gemm_cublas<float>(cublas, h_A, h_B, h_C, N);
//     }
//     double us = t.elapsed_microseconds();
//     std::cout << "Average: " << us
//               << std::endl;
//
//     // print C
//     h_C = C.get();
//     for (size_t i = 0; i < N; i++)
//     {
//       for (size_t j = 0; j < N; j++)
//       {
//         std::cout << h_C[i * N + j] << ' ';
//       }
//       std::cout << '\n';
//     }
//     return hpx::local::finalize();
// }
//
// // -------------------------------------------------------------------------
// int main(int argc, char** argv)
// {
//     using namespace hpx::program_options;
//     options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
//     // clang-format off
//     cmdline.add_options()
//         ("device",
//         hpx::program_options::value<std::size_t>()->default_value(0),
//         "Device to use")
//         ("sizemult",
//         hpx::program_options::value<std::size_t>()->default_value(5),
//         "Multiplier");
//
//     // clang-format on
//     hpx::local::init_params init_args;
//     init_args.desc_cmdline = cmdline;
//
//     auto result = hpx::local::init(hpx_main, argc, argv, init_args);
//     return result;
// }
