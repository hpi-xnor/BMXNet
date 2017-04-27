/*!
 * Copyright (c) 2017 by Contributors
 * \file xnor_cpu.h
 * \brief implementation of xnor-gemm operator for cpu
 * \author HPI-DeepLearning
*/

#ifndef MXNET_XNOR_CPU_H
#define MXNET_XNOR_CPU_H

#include <dmlc/logging.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <limits.h>
#include <tgmath.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>


namespace mxnet {
namespace op {
namespace xnor_cpu {

  // variable, position, value
  #define BIT_SET(var, pos, val) var |= (val << pos)
  
  //uint32_t, uint64_t, __int128
  #if BINARY_WORD_32 == 1
    typedef uint32_t BINARY_WORD;
  #endif
  #if BINARY_WORD_64 == 1
    typedef uint64_t BINARY_WORD;
  #endif

  const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);

  /**
  * @brief returns a mshadow dtype with corresponding bitwidth to BINARY_WORD
  *
  */
  inline mshadow::TypeFlag corresponding_dtype() {
    if (BITS_PER_BINARY_WORD == 32) {
      return mshadow::kFloat32;
    } else if (BITS_PER_BINARY_WORD == 64) {
      return mshadow::kFloat64;
    }
    assert(false);
    return mshadow::kFloat32;
  }

  /**
  * @brief a helper method for print out bit wise result
  * of a binary_word
  *
  */
  inline void print_int2Bin ( BINARY_WORD a )
  {
     
    for (int i=0; i <BITS_PER_BINARY_WORD; i++ )
    {
      if( a & (1 << i) ) 
        std::cout << 1;
      else
        std::cout << 0;
    }
    std::cout<<std::endl;
  }

  inline void print_int2Bin64 ( uint64_t a )
  {
     
    for (int i=0; i <64; i++ )
    {
      if( a & (1 << i) ) 
        std::cout << 1;
      else
        std::cout << 0;
    }
    std::cout<<std::endl;
  }

  /**
  * @brief this method scales the _popc(xnor(...)) result
  * into the dot(-1...1) result
  * Example: if scale range is 8, then 
  * the dot product result based -1 and 1:
  * -8  -6  -4  -2  0 2 4 6 8
  * XNOR&POPC result:
  *  0   1   2   3  4 5 6 7 8
  * so the equa should be:
  * dot_ouput = 2 * xnor_output - scale_range
  */
  inline float xnor_to_binary_dot ( float num, int scale_range)
  {
    return 2*num - scale_range;
  }

  /**
   * @brief gets the mean value over all elements of a weight volume
   *
   */
  inline float get_alpha(float* weight, int width, int height, int depth) {
    float accum = 0.0f;
    for (int z = 0; z < depth; ++z) {
      for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
          accum += std::abs(weight[z * (width * height) + x * height + y]);
        }
      }
    }
    return accum / (float) (width * height * depth);
  }

  /**
   * @brief collects all mean values across all input filters (alpha value as described in xnor paper)
   *
   */
  inline void get_alpha_plane(float* alpha_plane_out, float* weights,
                              int num_weights,
                              int kernel_width, int kernel_height,
                              int input_depth) {
    for (int i = 0; i < num_weights; i++) {
      alpha_plane_out[i] = get_alpha(&weights[i * kernel_height * kernel_width * input_depth], kernel_height, kernel_width, input_depth);
    }
  }

  /**
   * @brief plane with mean off all input channels for input volume (A plane as described in xnor paper)
   *
   */
  inline void get_A_planes(float* A_planes_out, float* input,
                           int input_depth, int input_width, int input_height,
                           int batch_size) {
    for (int i = 0; i < batch_size; i++) {
      for (int x = 0; x < input_width; ++x) {
        for (int y = 0; y < input_height; ++y) {
          float accum = 0.0f;
          for (int z = 0; z < input_depth; ++z) {
            accum += std::abs(input[i * (input_depth * input_width * input_height) +
                           z * (input_width * input_height) +
                           x * input_height +
                           y]);
          }
          A_planes_out[i * input_width * input_height +
                       x * input_height +
                       y] = accum / (float) input_depth;
        }
      }
    }
  }
  /**
   * @brief A plane convolved with k which is defined as a w*h matrix where every
   *        element is 1/(w*h)               (K plane as described in xnor paper)
   *
   */
  inline void get_K_planes(float* K_planes_out, float* A_planes,
                           int input_width, int input_height,
                           int kernel_width, int kernel_height,
                           int batch_size) {
    int K_width = (input_width - kernel_width + 2 * 0/*padding*/) / 1/*stride*/ + 1;
    int K_height = (input_height - kernel_height + 2 * 0/*padding*/) / 1/*stride*/ + 1;

    //@todo: super naive "conv" (no real conv since our k matrix has same elements everywhere)
    for (int i = 0; i < batch_size; i ++) {
      // for every batch
      for (int kx = 0; kx < K_width; kx++) {
        for (int ky = 0; ky < K_height; ky++) {
          // for every kx, ky in our output plane
          float accum = 0;
          // we do collect the sum of all values covered by the kernel
          for (int ix = kx; ix < kx + kernel_width; ix++) {
            for (int iy = ky; iy < ky + kernel_height; iy++) {
              accum += A_planes[i * input_width * input_height +
                                ix * input_height +
                                iy];
            }
          }
          // and multiply them with 1/(w * h)
          K_planes_out[i * K_width * K_height +
                       kx * K_height +
                       ky] = accum / ((float) kernel_height * kernel_width);
        }
      }
    }
  }

  /**
   * @brief pointwise multiplication of two same-size matrices
   *
   */
  inline void pointwise_mul_mm(float *output, const float *input, int size){
    for (int i = 0; i < size; i++) {
      output[i] *= input[i];
    }
  }

  /**
   * @brief pointwise multiplication of matrix with a scalar
   *
   */
  inline void pointwise_mul_scalar(float *output, const float scalar, int size){
    for (int i = 0; i < size; i++) {
      output[i] *= scalar;
    }
  }

  /**
   * @brief binarize an array of floats via the sign function into a single BINARY_WORD
   *
   */
  inline BINARY_WORD concatenate(float* array)
  {
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;

    for (int i = 0; i < BITS_PER_BINARY_WORD; i++)
    {
      sign = (array[i]>=0);
      rvalue = rvalue | (sign<< (i));
    }

    return rvalue;
  }

  /**
   * @brief binarize matrix
   *
   */
  inline void get_binary_row(float* row, BINARY_WORD * b_row, int size){

    #pragma omp parallel for
    for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
      BINARY_WORD rvalue=0;
      BINARY_WORD sign;
      for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
        sign = (row[i+j]>=0);
        BIT_SET(rvalue, j, sign);
      }
      b_row[i/BITS_PER_BINARY_WORD] = rvalue;
    }
  }

  /**
  * @brief binarize matrix column wise
  *
  */
  inline void get_binary_col(float* col, BINARY_WORD * b_col, int n, int k){        
    
    for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
      #pragma omp parallel for
      for(int x=0; x < k; ++x){          
        BINARY_WORD rvalue=0;
        BINARY_WORD sign;    
        for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
          sign = (col[(y*BITS_PER_BINARY_WORD+b)*k + x]>=0);          
          BIT_SET(rvalue, b, sign);
        }
        b_col[y*k + x] = rvalue;
      }
    }    
  }

  /**
   * @brief based-line xnor-gemm implementation without 
   * dot product, but use XNOR and POPCNT
   * __builtin_popcountll suitable for both 32bit and 64bit 
   *
   *
   */
  inline void xnor_gemm(int M, int K, int N,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
    int i,n,k;
    #pragma omp parallel for collapse(2)    
    for(i = 0; i < M; ++i){         
      for(n = 0; n < N; ++n){ 
        BINARY_WORD A_PART = A[i*lda+n];
        #pragma omp parallel for
        for(k = 0; k < K; ++k){          
          C[i*ldc+k] += __builtin_popcountll(~(A_PART ^ B[n*ldb+k]));
          
          /* testing code, will be removed wenn everything works fine.
          std::cout << "A_PART: ";
          print_int2Bin(A_PART);
          std::cout << "B_PART: ";
          print_int2Bin(B[n*ldb+k]);
          std::cout << "_XNOR_: ";
          print_int2Bin(~(A_PART ^ B[n*ldb+k]));
          std::cout << "POPC_: ";
          std::cout << __builtin_popcountl(~(A_PART ^ B[n*ldb+k])) << std::endl;
          */
        }
      }
    }
  }


  /**
   * @brief simple naive baseline gemm implementation
   *
   */
  inline void baseline_gemm(int M, int K, int N,
                            float *A, int lda,
                            float *B, int ldb,
                            float *C, int ldc){
    int i,n,k;
    #pragma omp parallel for collapse(2) 
    for(i = 0; i < M; ++i){
      for(n = 0; n < N; ++n){
        float A_PART = A[i*lda+n];
        #pragma omp parallel for
        for(k = 0; k < K; ++k){
          C[i*ldc+k] += A_PART * B[n*ldb+k];
        }
      }
    }
  }

  //========================================================================//
  //                       Optimized XNOR GEMM                              //
  //========================================================================//
  /* Create macros so that the matrices are stored in column-major order */
  #define A(i,j) a[ (j)*lda + (i) ]
  #define B(i,j) b[ (j)*ldb + (i) ]
  #define C(i,j) c[ (j)*ldc + (i) ]

  /* Block sizes which are based on L2-cache of cpu*/
  #define mc 128
  #define kc 128
  #define nc 256

  inline void pack_matrixB( int k, BINARY_WORD *b, int ldb, BINARY_WORD *b_to )
  {
    int j;    
    for( j=0; j<k; j++){  /* loop over rows of B */
      BINARY_WORD 
      *b_ij_pntr = &B( 0, j );

      *b_to = *b_ij_pntr;
      *(b_to+1) = *(b_ij_pntr+1);
      *(b_to+2) = *(b_ij_pntr+2);
      *(b_to+3) = *(b_ij_pntr+3);
      b_to += 4;
    }
  }

  inline void add_dot_4x4( int k, BINARY_WORD *a, int lda,  BINARY_WORD *b, int ldb, float *c, int ldc )
  {
    int p;
    register BINARY_WORD 
      /* hold contributions to
         C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) 
         C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ) 
         C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ) 
         C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
         c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
         c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
         c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
         c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg;     
    register BINARY_WORD 
          b_0p_reg,
          b_1p_reg,
          b_2p_reg,
          b_3p_reg,
          a_p0_reg,
          a_p1_reg,
          a_p2_reg,
          a_p3_reg;
    BINARY_WORD 
      /* Point to the current elements in the four columns of A */
      *a_p0_pntr, *a_p1_pntr, *a_p2_pntr, *a_p3_pntr; 
      
    a_p0_pntr = &A( 0, 0 );
    a_p1_pntr = &A( 0, 1 );
    a_p2_pntr = &A( 0, 2 );
    a_p3_pntr = &A( 0, 3 );

    c_00_reg = 0;   c_01_reg = 0;   c_02_reg = 0;   c_03_reg = 0;
    c_10_reg = 0;   c_11_reg = 0;   c_12_reg = 0;   c_13_reg = 0;
    c_20_reg = 0;   c_21_reg = 0;   c_22_reg = 0;   c_23_reg = 0;
    c_30_reg = 0;   c_31_reg = 0;   c_32_reg = 0;   c_33_reg = 0;

    for ( p=0; p<k; p++ ){
      b_0p_reg = B( 0, p );
      b_1p_reg = B( 1, p );
      b_2p_reg = B( 2, p );
      b_3p_reg = B( 3, p );

      a_p0_reg = *a_p0_pntr++;
      a_p1_reg = *a_p1_pntr++;
      a_p2_reg = *a_p2_pntr++;
      a_p3_reg = *a_p3_pntr++;
   
      /* First row and second rows */
      c_00_reg += __builtin_popcountll(~(b_0p_reg ^ a_p0_reg));
      c_10_reg += __builtin_popcountll(~(b_1p_reg ^ a_p0_reg));

      c_01_reg += __builtin_popcountll(~(b_0p_reg ^ a_p1_reg));
      c_11_reg += __builtin_popcountll(~(b_1p_reg ^ a_p1_reg));

      c_02_reg += __builtin_popcountll(~(b_0p_reg ^ a_p2_reg));
      c_12_reg += __builtin_popcountll(~(b_1p_reg ^ a_p2_reg));

      c_03_reg += __builtin_popcountll(~(b_0p_reg ^ a_p3_reg));
      c_13_reg += __builtin_popcountll(~(b_1p_reg ^ a_p3_reg));

      /* Third and fourth rows */
      c_20_reg += __builtin_popcountll(~(b_2p_reg ^ a_p0_reg));
      c_30_reg += __builtin_popcountll(~(b_3p_reg ^ a_p0_reg));

      c_21_reg += __builtin_popcountll(~(b_2p_reg ^ a_p1_reg));
      c_31_reg += __builtin_popcountll(~(b_3p_reg ^ a_p1_reg));

      c_22_reg += __builtin_popcountll(~(b_2p_reg ^ a_p2_reg));
      c_32_reg += __builtin_popcountll(~(b_3p_reg ^ a_p2_reg));

      c_23_reg += __builtin_popcountll(~(b_2p_reg ^ a_p3_reg));
      c_33_reg += __builtin_popcountll(~(b_3p_reg ^ a_p3_reg));
    }

    C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
    C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
    C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
    C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
  }

  inline void xnor_gemm2_inner_kernel( int m, int n, int k, BINARY_WORD *a, int lda, 
                                         BINARY_WORD *b, int ldb,
                                         float *c, int ldc, int first_time )
  {
    int i, j;
    BINARY_WORD packedB[ n * k ];   
    
    #pragma omp parallel for
    for ( j=0; j<n; j+=4 ){          /* Loop over the columns of C, unrolled by 4 */  
        if(first_time)
          pack_matrixB( k, &B( j, 0 ), ldb, &packedB[ j*k ] ); 
      #pragma omp parallel for
      for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
        /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
        one routine (four inner products) */
          //add_dot_4x4( k, &A( 0,i ), lda, &B( j,0 ), ldb, &C( j,i ), ldc );
        add_dot_4x4( k, &A( 0,i ), lda, &packedB[ j*k ], 4, &C( j,i ), ldc );
      }
    }
  }


 /**
  * applys blocking, packing, loop unrolling, register vars to improve the
  * xnor_gemm performance. ~100% performance improvement without openmp
  * compared with xnor_gemm() method.
  */
  inline void xnor_gemm2( int m, int n, int k, BINARY_WORD *a, int lda, 
                                      BINARY_WORD *b, int ldb,
                                      float *c, int ldc )
  {
    int i, p, pb, ib;

    /* This time, we compute a mc x n block of C by a call to the InnerKernel */
    for ( p=0; p<k; p+=kc ){
      pb = std::min( kc, k-p );      
      for ( i=0; i<m; i+=mc ){
        ib = std::min( mc, m-i );
        xnor_gemm2_inner_kernel( ib, n, pb, &A(p, i), lda, &B(0, p), ldb, &C( 0, i ), ldc, i==0 );
      }
    }
  }


 /**
  * @brief binarize matrix column wise. 
  * Loop unroll and using register vars.
  * ~30% performance improvement without openmp
  * compared with get_binary_col() method.
  */
  inline void get_binary_col2(float* col, BINARY_WORD * b_col, int n, int k){        
    #pragma omp parallel for
    for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
      BINARY_WORD * y_col_pt = &b_col[y*k];
      #pragma omp parallel for
      for(int x=0; x < k; x+=4){          
        register BINARY_WORD rvalue0=0, rvalue1=0, rvalue2=0, rvalue3=0;
           
        for(int b=0; b<BITS_PER_BINARY_WORD; b+=4){
          register BINARY_WORD sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7,
          sign8, sign9, sign10, sign11, sign12, sign13, sign14, sign15;

          float* col_0 = &col[(y*BITS_PER_BINARY_WORD+b)*k + x];
          float* col_1 = &col[(y*BITS_PER_BINARY_WORD+b+1)*k + x];
          float* col_2 = &col[(y*BITS_PER_BINARY_WORD+b+2)*k + x];
          float* col_3 = &col[(y*BITS_PER_BINARY_WORD+b+3)*k + x];

          sign0 = (*col_0>=0);          
          sign1 = (*col_1>=0);          
          sign2 = (*col_2>=0);          
          sign3 = (*col_3>=0);          
         
          BIT_SET(rvalue0, b, sign0);
          BIT_SET(rvalue0, (b+1), sign1);
          BIT_SET(rvalue0, (b+2), sign2);
          BIT_SET(rvalue0, (b+3), sign3);

          sign4 = (*(col_0+1)>=0);          
          sign5 = (*(col_1+1)>=0);          
          sign6 = (*(col_2+1)>=0);          
          sign7 = (*(col_3+1)>=0);          
         
          BIT_SET(rvalue1, b, sign4);
          BIT_SET(rvalue1, (b+1), sign5);
          BIT_SET(rvalue1, (b+2), sign6);
          BIT_SET(rvalue1, (b+3), sign7);

          sign8 = (*(col_0+2)>=0);          
          sign9 = (*(col_1+2)>=0);          
          sign10 = (*(col_2+2)>=0);          
          sign11 = (*(col_3+2)>=0);          
         
          BIT_SET(rvalue2, b, sign8);
          BIT_SET(rvalue2, (b+1), sign9);
          BIT_SET(rvalue2, (b+2), sign10);
          BIT_SET(rvalue2, (b+3), sign11);

          sign12 = (*(col_0+3)>=0);          
          sign13 = (*(col_1+3)>=0);          
          sign14 = (*(col_2+3)>=0);          
          sign15 = (*(col_3+3)>=0);          
         
          BIT_SET(rvalue3, b, sign12);
          BIT_SET(rvalue3, (b+1), sign13);
          BIT_SET(rvalue3, (b+2), sign14);
          BIT_SET(rvalue3, (b+3), sign15);
        }
        BINARY_WORD * pnter = &y_col_pt[x];
        *pnter = rvalue0;   
        *(pnter+1) = rvalue1;        
        *(pnter+2) = rvalue2;        
        *(pnter+3) = rvalue3;        
      }
    }     
  }
  //========================= END optimized xnor GEMM ===============================//

} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
#endif //MXNET_XNOR_CPU_H
