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

inline unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;

    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<< (i));
    }
    
    return rvalue;
}

inline void QConvolutionForward(const Tensor<gpu, 4, float> &data,
                                const Tensor<gpu, 2, float> &wmat,
                                const Tensor<gpu, 2, float> &in_col,
                                const Tensor<gpu, 2, float> &temp_dst,
                                const Tensor<gpu, 4, float> &out,
                                const mxnet::op::QConvolutionParam &param) {
	//get matrix dimension
	//wmat.size(1) should equal in_col.size(0)
	//TODO: check temp_dst structure should be (m x k)
	int m, n, k;
	int nchannel_input = 32;
	m = wmat.size(0);
	n = wmat.size(1);
	k = in_col.size(1);

	//set memory
	float *fA = wmat.dptr_; 
	float *fB = in_col.dptr_;
	float *fC = temp_dst.dptr_;
	//cudaMalloc(&fC, m * k * sizeof(float));
	cudaMemset(fC, 0, m * k * sizeof(int));
		
	//set bit memory
	//!!NOTE!! here we save 32 float numbers into one binary word
	unsigned int *Aconc, *Bconc;
	cudaMalloc(&Aconc, m*n/nchannel_input*sizeof(int));
	cudaMalloc(&Bconc, n*k/nchannel_input*sizeof(int));
		
	unsigned int *Bconc_host = (unsigned int*)malloc(n*k/nchannel_input*sizeof(int));
			
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	int block = 1;
	int grid = m * n / (block * nchannel_input);
	concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, m * n / nchannel_input);
	cudaDeviceSynchronize();
	
	float* fB_host = (float*)malloc(n * k * sizeof(float));
	cudaMemcpy(fB_host, fB, n * k * sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int x=0; x < k; ++x){
		for(int y=0; y<(n/32); y++){
			float * array = new float[32];
			for(int b=0; b<32; ++b){
				//std::cout<< "fB height: ";
				//std::cout << y*32+b << std::endl;
				array[b] = fB_host[(y*32+b)*k + x];
			}
				//std::cout<< "Bconc index: ";
				//std::cout << y*k + x << std::endl;
			Bconc_host[y*k + x]=concatenate(array);			
			delete[] array;
		}
	}
	
	cudaMemcpy(Bconc, Bconc_host, n*k/nchannel_input*sizeof(int), cudaMemcpyHostToDevice);
	//grid = n / block;
	//concatenate_cols_kernel<<<1, block>>>(fB, Bconc, n, k);
	//cudaDeviceSynchronize();
	
	//perform xnor gemm
	block = 16;
	dim3 blockDim(block, block);
	dim3 gridDim(k / block + 1, m / block + 1);
	xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, m, n / nchannel_input, k);		
	//gemm<<<gridDim, blockDim>>>(fA, fB, fC, m, n, k);
	cudaDeviceSynchronize();
	
	//cudaMemcpy(temp_dst.dptr_, fC, m * k * sizeof(float), cudaMemcpyDeviceToDevice);
	
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
	int * result = (int*)malloc(m * n /nchannel_input*sizeof(int));
	cudaMemcpy(result, Aconc,  m*n/nchannel_input*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i < m*n/nchannel_input; ++i){	
		std::cout << result[i];
		std::cout << ", ";	
	}		
	printf("\n");
	free(result);
	*/

	//cudaFree(fC);	
	cudaFree(Aconc);
	cudaFree(Bconc);
	free(Bconc_host);
}
}  // namespace cuda


inline void QConvolutionForward(const Tensor<gpu, 4, float> &data,
                                const Tensor<gpu, 2, float> &wmat,
                                const Tensor<gpu, 2, float> &in_col,
                                const Tensor<gpu, 2, float> &temp_dst,
                                const Tensor<gpu, 4, float> &out,
                                const mxnet::op::QConvolutionParam &param) {
	cuda::QConvolutionForward(data, wmat, in_col, temp_dst, out, param);
}

template<typename DType>
inline void QConvolutionForward(const Tensor<gpu, 4, DType> &data,
                                const Tensor<gpu, 2, DType> &wmat,
                                const Tensor<gpu, 2, DType> &in_col,
                                const Tensor<gpu, 2, DType> &temp_dst,
                                const Tensor<gpu, 4, DType> &out,
                                const mxnet::op::QConvolutionParam &param) {
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

