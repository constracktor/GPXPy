#ifndef HKOKKOS_ADAPTER_H_INCLUDED
#define HKOKKOS_ADAPTER_H_INCLUDED

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
 // static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
 //             "ExecutionSpace is not a Kokkos execution space");
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_A("d_A", size);
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_B("d_B", size);
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_C("d_C", size);
  // copy data from host to device when ready
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_A("h_A", size);
  host_buffer_t<T> hA = A.get();
  for (std::size_t i = 0; i < size; ++i) {
    h_A(i) = hA[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_A, h_A);
  
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_B("h_B", size);
  host_buffer_t<T> hB = B.get();
  for (std::size_t i = 0; i < size; ++i) {
    h_B(i) = hB[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_B, h_B);
  
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_C("h_C", size);
  host_buffer_t<T> hC = C.get();
  for (std::size_t i = 0; i < size; ++i) {
    h_C(i) = hC[i];
  }
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
            d_C(i * N + j) = d_C(i * N + j) - d_A(i * N + k) * d_B(j * N + k);
          }
        }
      });
  // copy the result back to the host
  hpx::kokkos::deep_copy_async(inst, h_C, d_C);
  inst.fence();
  for (std::size_t i = 0; i < size; ++i) {
    hC[i] = h_C(i);
  }
  // // return future
  return hpx::make_ready_future(hC);
}
#endif


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
//   int const n = 43;call_cblas_dot

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





// template<class Scalar, class Device>
// Scalar dot(View<const Scalar* , Device> a,
//            View<const Scalar*, Device> b) {
// // Check for Cuda memory and call cublas if true
// #ifdef KOKKOS_HAVE_CUDA
//   if(std::is_same<typename Device::memory_space,
//                            CudaSpace>::value ||
//      std::is_same<typename Device::memory_space,
//                            CudaUVMSpace>::value) {
//     return call_cublas_dot(a.ptr_on_device(), b.ptr_on_device(),
//                            a.extent_0() );
//   }
// #endif

// // Call CBlas on the host otherwise
//   if(std::is_same<typename Device::memory_space,HostSpace>::value) {
//     return call_cblas_dot(a.ptr_on_device(), b.ptr_on_device(),
//                           a.extent_0() );
//   }
// }