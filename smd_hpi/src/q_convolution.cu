/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include <mshadow/tensor.h>
#include "./xnor_kernels.h"

namespace mshadow {
namespace cuda {

inline void QConvolutionForward(const Tensor<gpu, 4, float> &data,
                                const Tensor<gpu, 2, float> &wmat,
                                const Tensor<gpu, 4, float> &out,
                                const mxnet::op::QConvolutionParam &param,
                                const Tensor<gpu, 2, float> &data_col,
								const Tensor<gpu, 2, float> &temp_dst) {
	//get matrix dimension
	//wmat.size(1) should equal data_col.size(0)
	//TODO: check temp_dst structure should be (m x k)
	int m, n, k;
	int nchan_in = 32;
	m = wmat.size(0);
	n = wmat.size(1);
	k = data_col.size(1);

	//set memory
	float *fA = wmat.dptr_; 
	float *fB = data_col.dptr_;
	float *fC;
	cudaMalloc(&fC, m * k * sizeof(float));
	cudaMemset(fC, 0, m * k * sizeof(int));
		
	//set bit memory
	//!!NOTE!! here we save 32 float numbers into one binary word
	unsigned int *Aconc, *Bconc;
	cudaMalloc(&Aconc, m*n/nchan_in*sizeof(int));
	cudaMalloc(&Bconc, n*k/nchan_in*sizeof(int));
			
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	int block = 64;
	int grid = m * n / (block * nchan_in)  + 1;
	concatenate_rows_kernel<<<grid, block, 0, stream>>>(fA, Aconc, m * n / nchan_in);
	
	grid = n / block + 1;
	concatenate_cols_kernel<<<grid, block, 0, stream>>>(fB, Bconc, n, k);
	cudaDeviceSynchronize();
	
	//perform xnor gemm
	block = 16;
	dim3 blockDim(block, block);
	dim3 gridDim(k / block + 1, m / block + 1);
	xnor_gemm<<<gridDim, blockDim, 0, stream>>>(Aconc, Bconc, fC, m, n / nchan_in, k);		
	//gemm<<<gridDim, blockDim, 0, stream>>>(fA, fB, fC, m, n, k);
	cudaDeviceSynchronize();
	cudaMemcpy(temp_dst.dptr_, fC, m * k * sizeof(float), cudaMemcpyDeviceToDevice);
	
	/*
	float* result = (float*)malloc(m * k * sizeof(float));
	cudaMemcpy(result, temp_dst.dptr_, m * k * sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 20; ++i){
		std::cout << result[i];
		std::cout << ", ";
	}
		
	printf("\n");
	free(result);
	*/
			
	/*
	int * result = (int*)malloc(m * n /nchan_in*sizeof(int));
	cudaMemcpy(result, Aconc,  m*n/nchan_in*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i < m*n/nchan_in; ++i){	
		std::cout << result[i];
		std::cout << ", ";	
	}		
	printf("\n");
	free(result);
	*/

	cudaFree(fC);	
	cudaFree(Aconc);
	cudaFree(Bconc);
}
}  // namespace cuda


inline void QConvolutionForward(const Tensor<gpu, 4, float> &data,
                                const Tensor<gpu, 2, float> &wmat,
                                const Tensor<gpu, 4, float> &out,
                                const mxnet::op::QConvolutionParam &param,
                                const Tensor<gpu, 2, float> &data_col,
								const Tensor<gpu, 2, float> &temp_dst) {
	cuda::QConvolutionForward(data, wmat, out, param, data_col, temp_dst);
}

template<typename DType>
inline void QConvolutionForward(const Tensor<gpu, 4, DType> &data,
                                const Tensor<gpu, 2, DType> &wmat,
                                const Tensor<gpu, 4, DType> &out,
                                const mxnet::op::QConvolutionParam &param,
								const Tensor<gpu, 2, DType> &data_col,
								const Tensor<gpu, 2, DType> &temp_dst) {
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
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

