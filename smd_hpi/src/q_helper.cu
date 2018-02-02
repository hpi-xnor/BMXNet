/*!
 * Copyright (c) 2016 by Contributors
 * \file q_helper.cu
 * \brief CUDA kernel function for q_helper.cuh
 * \author HPI-DeepLearning
*/

#ifndef MXNET_Q_HELPER_CU
#define MXNET_Q_HELPER_CU  

#include <float.h> 
#include <math.h>
#include "./q_helper.h"

#define QHELPER_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)


namespace mshadow{
namespace cuda{
namespace q_helper_cu{
// calc the next X^2
unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


__global__ void reduce_max_kernel(const float *input, float *d_out,  int size) {
  int tid         = threadIdx.x;                              // Local thread index
  int myId        = blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

  extern __shared__ float temp[]; //shared memory

  // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
  temp[tid] = (myId < size) ? input[myId] : -FLT_MAX;

  // --- make sure that all the shared memory loads have been completed
  __syncthreads();

  // --- Reduction in shared memory. Only half of the threads contribute to reduction.
  for (unsigned int s=blockDim.x/2; s>0; s>>=1){
      if (tid < s) { temp[tid] = fmaxf(temp[tid], temp[tid + s]); }
      // --- make sure that all memory operations have been completed
      __syncthreads();
  }

  if (tid == 0) {
      d_out[blockIdx.x] = temp[0];
  }
}

__global__ void reduce_sum_kernel(const float *input, float *d_out,  int size) {
  int tid         = threadIdx.x;                              // Local thread index
  int myId        = blockIdx.x*(blockDim.x*2) + threadIdx.x;   // Global thread index

  extern __shared__ float tempsum[]; //shared memory

  // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
  tempsum[tid] = (myId < size) ? input[myId] + input[myId+blockDim.x] : 0.0f;

  // --- make sure that all the shared memory loads have been completed
  __syncthreads();

  // --- Reduction in shared memory. Only half of the threads contribute to reduction.
  for (unsigned int s=blockDim.x/2; s>0; s>>=1){
      if (tid < s) { tempsum[tid] += tempsum[tid + s]; }
      // --- make sure that all memory operations have been completed
      __syncthreads();
  }

  if (tid == 0) {
      d_out[blockIdx.x] = tempsum[0];
  }
}


extern "C" 
float launch_max_reduce(float *input, int size_of_array)
{  
  if(size_of_array == 0) return 0.0;

  int NumThreads  = (size_of_array < kMaxThreadsPerBlock) ? nextPow2(size_of_array) : kMaxThreadsPerBlock;
  int NumBlocks   = (size_of_array + NumThreads - 1) / NumThreads;
  CheckLaunchParam(NumBlocks, NumThreads, "reduce_max-kernel");

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

  // --- allocate device memory for temporary results
  unsigned int mem_size_tmp = sizeof(float) * NumBlocks;
  float* d_tmp;
  QHELPER_CUDA_CHECK(cudaMalloc((void**) &d_tmp, mem_size_tmp));
  QHELPER_CUDA_CHECK(cudaMemset(d_tmp, 0, mem_size_tmp));

  // --- reduce2  STAGE 1
  reduce_max_kernel<<<NumBlocks, NumThreads, smemSize>>>(input, d_tmp, size_of_array);

  // --- recalc parameters
  int old_NumBlocks = NumBlocks;
  NumThreads  = (NumBlocks < kMaxThreadsPerBlock) ? nextPow2(NumBlocks) : kMaxThreadsPerBlock;
  NumBlocks   = (old_NumBlocks + NumThreads - 1) / NumThreads;
  smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

  // --- allocate device memory for final result
  unsigned int mem_size_f = sizeof(float)*NumBlocks;
  float* d_final;
  QHELPER_CUDA_CHECK(cudaMalloc((void**) &d_final, mem_size_f));
  QHELPER_CUDA_CHECK(cudaMemset(d_final, 0, mem_size_f));

  // --- reduce2  STAGE 2
  reduce_max_kernel<<<NumBlocks, NumThreads, smemSize>>>(d_tmp, d_final, old_NumBlocks);

  // --- copy final result from device to host
  float* h_final = (float*)malloc(mem_size_f);
  QHELPER_CUDA_CHECK(cudaMemcpy(h_final, d_final, mem_size_f, cudaMemcpyDeviceToHost));

  // --- find the maximum on the host   STAGE 3
  float result_reduce0 = -FLT_MAX;
  for (int i=0; i<NumBlocks; i++) result_reduce0 = fmaxf(h_final[i], result_reduce0);
  //printf("Result = %f\n", result_reduce0);

  // --- release memory
  free(h_final);
  QHELPER_CUDA_CHECK(cudaFree(d_final));
  QHELPER_CUDA_CHECK(cudaFree(d_tmp));
  return result_reduce0;
}  // launch_max_reduce


/*
 * We apply a multiple-stage strategy for max reduce.
 * 1. calc the maximum for each block
 * 2. calc the maximum among all block-max
 */
extern "C" 
double launch_max_reduce_d(double *input, int size_of_array)
{
  CHECK(false) << "only float supported";
  return .0;
}



extern "C" 
float launch_mean_reduce(float *input, int size_of_array)  
{ 
  if(size_of_array == 0) return .0f;

  int NumThreads  = (size_of_array < kMaxThreadsPerBlock) ? nextPow2(size_of_array) : kMaxThreadsPerBlock;
  int NumBlocks   = (size_of_array + NumThreads - 1) / NumThreads;
  CheckLaunchParam(NumBlocks, NumThreads, "reduce_mean-kernel");

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

  // --- allocate device memory for temporary results
  unsigned int mem_size_tmp = sizeof(float) * NumBlocks;
  float* d_tmp;
  QHELPER_CUDA_CHECK(cudaMalloc((void**) &d_tmp, mem_size_tmp));
  QHELPER_CUDA_CHECK(cudaMemset(d_tmp, 0, mem_size_tmp));

  // --- reduce2  STAGE 1
  reduce_sum_kernel<<<NumBlocks, NumThreads, smemSize>>>(input, d_tmp, size_of_array);

  // --- recalc parameters
  int old_NumBlocks = NumBlocks;
  NumThreads  = (NumBlocks < kMaxThreadsPerBlock) ? nextPow2(NumBlocks) : kMaxThreadsPerBlock;
  NumBlocks   = (old_NumBlocks + NumThreads - 1) / NumThreads;
  smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

  // --- allocate device memory for final result
  unsigned int mem_size_f = sizeof(float)*NumBlocks;
  float* d_final;
  QHELPER_CUDA_CHECK(cudaMalloc((void**) &d_final, mem_size_f));  
  QHELPER_CUDA_CHECK(cudaMemset(d_final, 0, mem_size_f));

  // --- reduce2  STAGE 2
  reduce_sum_kernel<<<NumBlocks, NumThreads, smemSize>>>(d_tmp, d_final, old_NumBlocks);

  // --- copy final result from device to host
  float* h_final = (float*)malloc(mem_size_f);
  QHELPER_CUDA_CHECK(cudaMemcpy(h_final, d_final, mem_size_f, cudaMemcpyDeviceToHost));

  // --- find the sum on the host   STAGE 3
  float result_reduce0 = .0f;
  for (int i=0; i<NumBlocks; i++) {
    result_reduce0 += h_final[i];
  }

  // --- release memory
  free(h_final);  
  QHELPER_CUDA_CHECK(cudaFree(d_final));
  QHELPER_CUDA_CHECK(cudaFree(d_tmp));  
  return result_reduce0/(float)size_of_array;
}

/*
 * We apply a multiple-stage strategy for mean reduce.
 * 1. calc the sum for each block
 * 2. calc the sum among all block-sum
 * 3. calc mean
 */  
extern "C" 
double launch_mean_reduce_d(double *input, int size_of_array)  
{ 
  CHECK(false) << "only float supported";
  return .0;
}  

} // namespace q_helper_cu
} // namespace cuda
} // namespace mshadow




#endif //MXNET_Q_HELPER_CU
