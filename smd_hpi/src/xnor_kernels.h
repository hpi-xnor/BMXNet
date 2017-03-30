/*
 * xnor_kernels.h
 *
 *  Created on: Feb 21, 2017
 *      Author: fb10dl02
 */

#ifndef SMD_HPI_SRC_XNOR_KERNELS_H_
#define SMD_HPI_SRC_XNOR_KERNELS_H_

typedef unsigned int BINARY_WORD;
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
#define BLOCK_SIZE_XNOR 16

__global__ void gemm(float* A, float* B, float* C, int m, int n, int k);
__device__ unsigned int concatenate(float* array);
__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size);
__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n);
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k);

#endif /* SMD_HPI_SRC_XNOR_KERNELS_H_ */
