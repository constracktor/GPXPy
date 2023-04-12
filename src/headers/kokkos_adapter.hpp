#ifndef HKOKKOS_ADAPTER_H_INCLUDED
#define HKOKKOS_ADAPTER_H_INCLUDED

#include <hpx/local/future.hpp>
#include <hpx/kokkos.hpp>
#include <hpx/kokkos/detail/polling_helper.hpp>

//C = C - A * B^T                                       
template <typename T, typename ExecutionSpace>
host_buffer_t<T> gemm_kokkos(ExecutionSpace &&inst,
                             host_buffer_t<T> A,
                             host_buffer_t<T> B,
                             host_buffer_t<T> C,
                             std::size_t N)    
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_A("d_A", size);
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_B("d_B", size);
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_C("d_C", size);
  // copy data from host to device
  // A
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_A("h_A", size);
  for (std::size_t i = 0; i < size; ++i) 
  {
    h_A(i) = A[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_A, h_A);
  // B
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_B("h_B", size);
  for (std::size_t i = 0; i < size; ++i) 
  {
    h_B(i) = B[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_B, h_B);
  // C
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_C("h_C", size);
  for (std::size_t i = 0; i < size; ++i) 
  {
    h_C(i) = C[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_C, h_C);
  // compute GEMM (C = C - A * B^T)
  hpx::kokkos::parallel_for_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0, N),
      KOKKOS_LAMBDA(int i) 
      { 
        for (int j=0; j < N; j++)
        {
          for(int k=0; k < N; k++)
          {
            d_C(i * N + j) = beta * d_C(i * N + j) + alpha * d_A(i * N + k) * d_B(j * N + k);
          }
        }
      });
  // copy the result back to the host
  hpx::kokkos::deep_copy_async(inst, h_C, d_C);
  inst.fence();
  for (std::size_t i = 0; i < size; ++i) 
  {
    C[i] = h_C(i);
  }
  return C;
}

//  A = A - B * B^T
template <typename T, typename ExecutionSpace>
host_buffer_t<T> syrk_kokkos(ExecutionSpace &&inst,
                      host_buffer_t<T> A,
                      host_buffer_t<T> B,
                      std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_A("d_A", size);
  Kokkos::View<T*, typename std::decay<ExecutionSpace>::type> d_B("d_B", size);
  // copy data from host to device
  // A
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_A("h_A", size);
  for (std::size_t i = 0; i < size; ++i) 
  {
    h_A(i) = A[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_A, h_A);
  // B
  Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> h_B("h_B", size);
  for (std::size_t i = 0; i < size; ++i) 
  {
    h_B(i) = B[i];
  }
  hpx::kokkos::deep_copy_async(inst, d_B, h_B);
  // compute SYRK A = A - B * B^T
  hpx::kokkos::parallel_for_async(
      Kokkos::RangePolicy<typename std::decay<ExecutionSpace>::type>(inst, 0, N),
      KOKKOS_LAMBDA(int i) 
      { 
        for (int j=0; j < N; j++)
        {
          for(int k=0; k < N; k++)
          {
            d_A(i * N + j) = beta * d_A(i * N + j) + alpha * d_B(i * N + k) * d_B(j * N + k);
          }
        }
      });
  // copy the result back to the host
  hpx::kokkos::deep_copy_async(inst, h_A, d_A);
  inst.fence();
  for (std::size_t i = 0; i < size; ++i) 
  {
    A[i] = h_A(i);
  }
  return A;
}
#endif

// How to Access CUBLAS and CBLAS with Kokkos
// note cublas uses column major ordering : (A * B^T)^T = B * A^T
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