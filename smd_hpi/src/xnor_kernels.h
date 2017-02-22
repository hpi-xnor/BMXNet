/*
 * xnor_kernels.h
 *
 *  Created on: Feb 21, 2017
 *      Author: fb10dl02
 */

#ifndef SMD_HPI_SRC_XNOR_KERNELS_H_
#define SMD_HPI_SRC_XNOR_KERNELS_H_

__global__ void gemm(float* A, float* B, float* C, int m, int n, int k);
__device__ unsigned int concatenate(float* array);
__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size);
__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n);
__device__ float* deconcatenate(unsigned int x);
__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size);
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k);

#endif /* SMD_HPI_SRC_XNOR_KERNELS_H_ */
