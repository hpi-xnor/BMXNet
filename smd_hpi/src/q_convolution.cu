/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include <mshadow/tensor.h>
#if MXNET_USE_CUDNN == 1
#include "./q_cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN
#include "./xnor_kernels.h"

namespace mshadow {
namespace cuda {

/*
 * m: number of output channels (num_filter) per group
 * n: number of pixels of output images per channel (output dimension)
 * k: number of input channels per group * kernel size(e.g., 3x3=9)
 */
inline void _BinaryConvolutionForward(int m, int n, int k,
										mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
										Tensor<gpu, 1, float> &workspace,
										const Tensor<gpu, 2, float> &in_col,
										Tensor<gpu, 2, float> &temp_dst) {

	CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * k);

	// check matrix dims:
	// temp_dst should have dims (m x n)	
	CHECK_EQ((int)temp_dst.size(0), m);
	CHECK_EQ((int)temp_dst.size(1), n);	
	CHECK_EQ(n, (int)in_col.size(1));
	CHECK_EQ(k, (int)in_col.size(0));
	
	cudaStream_t stream = Stream<gpu>::GetStream(temp_dst.stream_);
	
	//set memory
	float *fB = in_col.dptr_;
	float *fC = temp_dst.dptr_;					
	xnor_cuda::BINARY_WORD* binary_col = (xnor_cuda::BINARY_WORD*) workspace.dptr_;	

	//concatinates matrix (k x n) -> (k/32 x n)
	int threads_per_block = xnor_cuda::get_next_block_dim(n);
	dim3 conc_block(threads_per_block,1,1);
  	dim3 conc_grid(n/threads_per_block+1,1);
	xnor_cuda::concatenate_cols_kernel<<<conc_grid, conc_block, 0, stream>>>(fB, binary_col, k, n);
	cudaDeviceSynchronize();
	
	//perform xnor gemm
	threads_per_block = xnor_cuda::get_next_block_dim(m, k/xnor_cuda::BITS_PER_BINARY_WORD, n);
	// Shared memory used to store Asub and Bsub respectively
  	int memsize = threads_per_block*threads_per_block*sizeof(xnor_cuda::BINARY_WORD)*2;
	dim3 block(threads_per_block, threads_per_block, 1);
	dim3 grid(n/threads_per_block + 1, m/threads_per_block + 1);
	xnor_cuda::xnor_gemm<<<grid, block, memsize, stream>>>((xnor_cuda::BINARY_WORD*)wmat_binarized, binary_col, fC, 
	                                                       m, k/xnor_cuda::BITS_PER_BINARY_WORD, n, 
	                                                       threads_per_block);		
	cudaDeviceSynchronize();	
}
}  // namespace cuda

	inline void QConvolutionForward(int m, int n, int k,
									mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
		cuda::_BinaryConvolutionForward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
	}

	inline void QConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, float> &wmat,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
		cudaStream_t stream = Stream<gpu>::GetStream(temp_dst.stream_);
    	float *fA = wmat.dptr_;
    	mxnet::op::xnor_cpu::BINARY_WORD *wmat_binarized;
		cudaMalloc(&wmat_binarized, m*k/xnor_cuda::BITS_PER_BINARY_WORD*sizeof(mxnet::op::xnor_cpu::BINARY_WORD));
    	//concatinates matrix (m x k) -> (m x k/32)
		int threads_per_block = xnor_cuda::get_next_block_dim(m*k/xnor_cuda::BITS_PER_BINARY_WORD);
	 	int blocks_per_grid = m * k / (threads_per_block * xnor_cuda::BITS_PER_BINARY_WORD) + 1;
	 	dim3 conc_block(threads_per_block,1,1);
  		dim3 conc_grid(blocks_per_grid,1);
	 	xnor_cuda::concatenate_rows_kernel<<<conc_grid, conc_block, 0, stream>>>(fA, (xnor_cuda::BINARY_WORD*)wmat_binarized, m * k / xnor_cuda::BITS_PER_BINARY_WORD);
	 	cudaDeviceSynchronize();
    	cuda::_BinaryConvolutionForward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
    	cudaFree(wmat_binarized);
	}

	template<typename DType>
	inline void QConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, DType> &wmat,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}

	template<typename DType>
	inline void QConvolutionForward(int m, int n, int k,
									mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}
} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(QConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
	Operator *op = NULL;
	// If 1D convolution, use MXNet implementation
	if (param.kernel.ndim() == 1) {
		MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new QConvolutionOp<gpu, DType>(param);
		})
		return op;
	}

	// depth wise conv
	if (param.num_filter == param.num_group &&
			param.layout.value() == mshadow::kNCHW &&
			param.num_filter == (*in_shape)[conv::kData][1] &&
			param.kernel.ndim() == 2 &&
			param.dilate == mshadow::Shape2(1, 1) &&
			dtype == mshadow::kFloat32) {
		LOG(WARNING) << "depth wise conv selected, not available in q_convolution, falling back to old convolution/cudnn";
	  //op = new DepthwiseConvolutionOp<float>(param, *in_shape, *out_shape);
		//return op;
	}

	#if MXNET_USE_CUDNN == 1
	// On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off || param.binarized_weights_only) {
      op = new QConvolutionOp<gpu, DType>(param);
    } else if (!QCuDNNConvolutionOp<DType>::Supports(param, compute_type, compute_type, ctx)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      op = new QConvolutionOp<gpu, DType>(param);
    } else {
      op = new QCuDNNConvolutionOp<DType>(param, compute_type, compute_type,
                                         *in_shape, *out_shape, ctx);
    }
  })
	#else
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
			op = new QConvolutionOp<gpu, DType>(param);
	})
	#endif  // MXNET_USE_CUDNN
	return op;

}

}  // namespace op
}  // namespace mxnet

