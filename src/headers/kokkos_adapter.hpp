#ifndef KOKKOS_ADAPTER_H_INCLUDED
#define KOKKOS_ADAPTER_H_INCLUDED

#include <hpx/local/future.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

//C = C - A * B^T
template <typename T, typename ExecutionSpace>
hpx::shared_future<host_buffer_t<T>> gemm_kokkos(ExecutionSpace &&inst,
                                               hpx::shared_future<host_buffer_t<T>> A,
                                               hpx::shared_future<host_buffer_t<T>> B,
                                               hpx::shared_future<host_buffer_t<T>> C,
                                               std::size_t N)
                                               
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
              "ExecutionSpace is not a Kokkos execution space");
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  Kokkos::View<T *, typename std::decay<ExecutionSpace>::type> d_A("d_A", size);
  Kokkos::View<T *, typename std::decay<ExecutionSpace>::type> d_B("d_B", size);
  Kokkos::View<T *, typename std::decay<ExecutionSpace>::type> d_C("d_C", size);
  // copy data from host to device when ready
  Kokkos::View<T *, Kokkos::DefaultHostExecutionSpace> h_A("h_A", size);
  h_A = A.get();
  hpx::kokkos::deep_copy_async(inst, d_A, h_A);
  Kokkos::View<T *, Kokkos::DefaultHostExecutionSpace> h_B("h_B", size);
  h_B = B.get();
  hpx::kokkos::deep_copy_async(inst, d_B, h_B);
  Kokkos::View<T *, Kokkos::DefaultHostExecutionSpace> h_C("h_C", size);
  h_C = C.get();
  hpx::kokkos::deep_copy_async(inst, d_C, h_C);

  // compute GEMM (C = C - A * B^T) on device
  // note cublas uses column major ordering : (A * B^T)^T = B * A^T
  hpx::kokkos::parallel_for_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0,
                                                                     N),
      KOKKOS_LAMBDA(int i) 
      { 
        for (int j=0; j < N; j++)
        {
          for(int k=0; k < N; k++)
          {
            d_C(i * N + j) = d_C(i * N + j) + d_A(i * N + k) * d_B(j * N + k);
          }
        }
      });
  // copy the result back to the host
  hpx::kokkos::deep_copy_async(inst, h_C, d_C);
  inst.fence();
  // return future
  return hpx::make_ready_future(h_C);
}



// template <typename ExecutionSpace> 
//   void test(ExecutionSpace &&inst) {
//     static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
//                   "ExecutionSpace is not a Kokkos execution space");
//     test_parallel_for(inst);
//     test_parallel_reduce(inst);
//     test_parallel_scan(inst);
// }

// template <typename ExecutionSpace>
// void test_parallel_for(ExecutionSpace &&inst) {
//   int const n = 43;

//   Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
//       parallel_for_result_host("parallel_for_result_host", n);
//   Kokkos::View<int *, typename std::decay<ExecutionSpace>::type>
//       parallel_for_result("parallel_for_result", n);
//   for (std::size_t i = 0; i < n; ++i) {
//     parallel_for_result_host(i) = 0;
//   }
//   hpx::kokkos::deep_copy_async(inst, parallel_for_result,
//                                parallel_for_result_host);
//   hpx::kokkos::parallel_for_async(
//       Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0,
//                                                                      n),
//       KOKKOS_LAMBDA(int i) { parallel_for_result(i) = i; });
//   hpx::kokkos::deep_copy_async(inst, parallel_for_result_host,
//                                parallel_for_result);
//   inst.fence();
//   for (std::size_t i = 0; i < n; ++i) {
//     HPX_KOKKOS_DETAIL_TEST(parallel_for_result_host(i) == i);
//   }
// }