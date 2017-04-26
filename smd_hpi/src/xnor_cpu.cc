/*!
 * Copyright (c) 2017 by Contributors
 * \file xnor_cpu.cc
 * \brief implementation of xnor-gemm operator for cpu
 * \author HPI-DeepLearning
*/

#include "xnor_cpu.h"

namespace mxnet {
namespace op {
namespace xnor_cpu {

#define UNROLLN	6

void xnor_gemm(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
    int m,k,n;
    #pragma omp parallel for    
    for (m = 0; m < M; ++m) {
    	#pragma omp parallel for
      for (k = 0; k < ((K / UNROLLN) * UNROLLN); k+=UNROLLN) {
        BINARY_WORD A_PART[UNROLLN];
        A_PART[0] = A[m*lda+k];
        A_PART[1] = A[m*lda+k+1];
        A_PART[2] = A[m*lda+k+2];
        A_PART[3] = A[m*lda+k+3];
        A_PART[4] = A[m*lda+k+4];
        A_PART[5] = A[m*lda+k+5];
        #pragma omp parallel for
        for (n = 0; n < N; ++n) {
          int popc[UNROLLN];
          popc[0] = __builtin_popcountl(~(A_PART[0] ^ B[(k+0)*ldb+n]));
          popc[1] = __builtin_popcountl(~(A_PART[1] ^ B[(k+1)*ldb+n]));
          popc[2] = __builtin_popcountl(~(A_PART[2] ^ B[(k+2)*ldb+n]));
          popc[3] = __builtin_popcountl(~(A_PART[3] ^ B[(k+3)*ldb+n]));
          popc[4] = __builtin_popcountl(~(A_PART[4] ^ B[(k+4)*ldb+n]));
          popc[5] = __builtin_popcountl(~(A_PART[5] ^ B[(k+5)*ldb+n]));
          C[m*ldc+n] += popc[0] + popc[1] + popc[2] + popc[3] + popc[4] + popc[5];
        }
      }

      #pragma omp parallel for 
      for (k=(K / UNROLLN) * UNROLLN; k < K; ++k) {
    		BINARY_WORD A_PART = A[m*lda+k];
	  		#pragma omp parallel for
        for (n = 0; n < N; ++n) {
        	C[m * ldc + n] += __builtin_popcountl(~(A_PART ^ B[k * ldb + n]));
        }
      }
    }


  }


  // write popc in int array, in the end convert back (only very small perf. improvement)
  void xnor_gemm_convert_to_int(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
    int m,k,n;
    int popc[M*N];
    #pragma omp parallel for collapse(2)    
    for (m = 0; m < M; ++m) {
      for (k = 0; k < K; k++) {
        BINARY_WORD A_PART = A[m*lda+k];
        #pragma omp parallel for
        for (n = 0; n < N; ++n) {
          popc[m*ldc+n] += __builtin_popcountl(~(A_PART ^ B[k*ldb+n]));
        }
      }
    }

    for (int i=0; i < M*N; i++) {
    	C[i] = popc[i];
    }
  }


  // our baseline
  void xnor_gemm_baseline(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
    int m,k,n;
    #pragma omp parallel for collapse(2)    
    for (m = 0; m < M; ++m) {
      for (k = 0; k < K; k++) {
        BINARY_WORD A_PART = A[m*lda+k];
        #pragma omp parallel for
        for (n = 0; n < N; ++n) {
          C[m*ldc+n] += __builtin_popcountl(~(A_PART ^ B[k*ldb+n]));
        }
      }
    }
  }


} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
