/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cu
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
#include "./q_fully_connected-inl.h"
#include <mshadow/tensor.h>
#include "./xnor_kernels.h"

namespace mshadow {
namespace cuda {

inline void QFullyConnectedForward(const Tensor<gpu, 2, float> &data,
                                const Tensor<gpu, 2, float> &wmat,
                                const Tensor<gpu, 2, float> &out) {
                                	    
	//======== TODO: able to support arbitrary input channel size ==========//
	CHECK_EQ(data.size(1) % BITS_PER_BINARY_WORD, 0) << "input channel number for binary fully_connected layer is not divisible by 32.";
                            
	//get matrix dimension		
	int m, n, k;
	int basic_factor_nchannel_input = BITS_PER_BINARY_WORD;
	m = data.size(0);
	n = data.size(1);
	k = wmat.size(1);	
	
	//check matrix dims:
	// 	data.size(1) should equal wmat.size(0)
	//	out should have dims (m, k)
	CHECK_EQ((int)data.size(1), (int)wmat.size(0));
	CHECK_EQ((int)out.size(0), (int)data.size(0));
	CHECK_EQ((int)out.size(1), (int)wmat.size(1));
	
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	
	//set memory
	float *fA = data.dptr_; 
	float *fB = wmat.dptr_;
	float *fC = out.dptr_;	
			
	//set bit memory
	//!!NOTE!! here we save 32 float numbers into one binary word
	unsigned int *Aconc, *Bconc;
	cudaMalloc(&Aconc, m*n/basic_factor_nchannel_input*sizeof(int));
	cudaMalloc(&Bconc, n*k/basic_factor_nchannel_input*sizeof(int));				
	
	//concatinates matrix (m x n) -> (m x n/32)
	// kMaxThreadsPerBlock defined in "mxnet/mshadow/mshadow/cuda/tensor_gpu-inl.cuh"
	int threads_per_block = kMaxThreadsPerBlock;
	int blocks_per_grid = m * n / (threads_per_block * basic_factor_nchannel_input) + 1;
	concatenate_rows_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(fA, Aconc, m * n / basic_factor_nchannel_input);

	//concatinates matrix (n x k) -> (n/32 x k)
	threads_per_block = kMaxThreadsPerBlock;
	blocks_per_grid = k / threads_per_block + 1;
	concatenate_cols_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(fB, Bconc, n, k);
	cudaDeviceSynchronize();
	
	//perform xnor gemm
	threads_per_block = BLOCK_SIZE_XNOR;
	dim3 blockDim(threads_per_block, threads_per_block);
	dim3 gridDim(k / threads_per_block + 1, m / threads_per_block + 1);
	xnor_gemm<<<gridDim, blockDim, 0, stream>>>(Aconc, Bconc, fC, m, n / basic_factor_nchannel_input, k);		
	cudaDeviceSynchronize();	
			
	cudaFree(Aconc);
	cudaFree(Bconc);
}
}  // namespace cuda


  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     const Tensor<gpu, 1, float> &wmat_binarized,
                                     Tensor<gpu, 2, float> &out) {
    CHECK(false) << "cuda with pre-binarized weights not implemented";
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     const Tensor<gpu, 2, float> &wmat,
                                     Tensor<gpu, 2, float> &out) {
    cuda::QFullyConnectedForward(data, wmat, out);
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     const Tensor<gpu, 1, DType> &wmat_binarized,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     const Tensor<gpu, 2, DType> &wmat,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }
} // namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(QFullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QFullyConnectedOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
