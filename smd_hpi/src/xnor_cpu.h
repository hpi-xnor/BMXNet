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

  //uint32_t, uint64_t, __int128
  typedef uint32_t BINARY_WORD;
  const int BITS_PER_BINARY_WORD (sizeof(mxnet::op::xnor_cpu::BINARY_WORD) * CHAR_BIT);

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

    for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
      float * array = new float[BITS_PER_BINARY_WORD];

      #pragma omp parallel for
      for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
        array[j] = row[i+j];
      }

      b_row[i/BITS_PER_BINARY_WORD] = concatenate(array);
      delete[] array;
    }
  }

  /**
  * @brief binarize matrix column wise
  *
  */
  inline void get_binary_col(float* col, BINARY_WORD * b_col, int n, int k){

    #pragma omp parallel for collapse(2)
    for(int x=0; x < k; ++x){
      for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
        float * array = new float[BITS_PER_BINARY_WORD];
        #pragma omp parallel for
        for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
          array[b] = col[(y*BITS_PER_BINARY_WORD+b)*k + x];
        }

        b_col[y*k + x] = concatenate(array);
        delete[] array;
      }
    }
  }

  /**
   * @brief optimized gemm without multiplication but instead XOR and POPCNT
   * __builtin_popcountl suitable for both 32bit and 64bit 
   *
   */
  inline void xnor_gemm(int M, int K, int N,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
    int i,n,k;

    //#pragma omp parallel for collapse(2)
    for(i = 0; i < M; ++i){
      for(n = 0; n < N; ++n){
        BINARY_WORD A_PART = A[i*lda+n];
        //#pragma omp parallel for
        for(k = 0; k < K; ++k){
          C[i*ldc+k] += (float)__builtin_popcountl(~(A_PART ^ B[n*ldb+k]));

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
    for(i = 0; i < M; ++i){
      for(n = 0; n < N; ++n){
        float A_PART = A[i*lda+n];
        for(k = 0; k < K; ++k){
          C[i*ldc+k] += A_PART * B[n*ldb+k];
        }
      }
    }
  }

//  /**
// * binary gemm. instead of standard dot product
// * we apply binary_dot: _popcount( xnor() ) operators to perform the convolution
// *
// * params:
// * 	weights: (m x n)
// * 	col_input: inputs, unpacked via patch2col (NOT n x k, !BUT TRANSPOSED!: k x n)
// * 	output: (m x k)
// * 	m, n, k: size of matrices
// */
//
//  inline void binary_gemm(BINARY_WORD* weights, BINARY_WORD* col_input, float* output, int m, int n, int k) {
//    CHECK_EQ(n % 32, 0) << "!!! no masking yet, only input channel % 32==0";
//
//    int bitwords_per_row = n / BITS_PER_BINARY_WORD;
//
//    for (int mi = 0; mi < m; mi++) {
//      for (int ki = 0; ki < k; ki++) {
//        float accum = 0;
//        for (int bitword_index_in_row = 0; bitword_index_in_row < bitwords_per_row; bitword_index_in_row++) {
//          // masking or only 32bit support important cause !gaah!
//          BINARY_WORD pixel = col_input[ki * bitwords_per_row + bitword_index_in_row];
//          BINARY_WORD weight = weights[mi * bitwords_per_row + bitword_index_in_row];
//          accum += __builtin_popcount(~(pixel ^ weight));
//        }
//
//        output[mi * k + ki] = accum;
//      }
//    }
//  }
//
///**
// * binary convolution implementation. instead of standard dot product
// * we apply binary_dot: _popcount( xnor() ) operators to perform the convolution
// * on binary input(I) and weight matrix (W), the alpha is the scaling factor
// * for W, 2D_beta consist of all scaling factor beta for the input tensor.
// * The calculation follows the equation:
// * 		I * W ≈ (sign(I) (binary_dot) sign(W)) (dot) (2D_beta)(alpha)
// *
// * params:
// * 	output:output data array
// * 	input: input tensor
// * 	weights: weight filter
// */
//
//inline void binary_conv2D(float* output,  const BINARY_WORD *input,
//						  const BINARY_WORD *weights, int ix, int iy,
//						  int wx, int wy, int pad_x, int pad_y, int stride,
//						  int output_width, int output_height, int filter_iter_base) {
//    int r, rd, c, cd;
//    int wx_2 = wx / 2;
//    int wy_2 = wy / 2;
//
//    // Indexing for weights
//    int wsx, wex, wsy, wey;
//    wsx = -wx_2;				// weight start x
//    wsy = -wy_2;	 			// weight start y
//
//    if (wx % 2 == 1)  		// odd weights w
//        wex = wx_2 + 1;			// weight end x
//    else
//        wex = wx_2;
//    if (wy % 2 == 1)  		// odd weights h
//		wey = wy_2 + 1;			// weight end y
//	else
//		wey = wy_2;
//
//    // Indexing for input pixels. since stride can only be 1 now,
//    int sx = pad_x + wx_2;               // start x
//    int ex = ix + pad_x - wx_2;      // end x
//    int sy = pad_y + wy_2;               // start y
//    int ey = iy + pad_y - wy_2;      // end y
//
//    //padded input width
//    int px = ix + 2*pad_x;
//
//    for (r = sy; r < ey; ++r) { 					// slide in y on input
//        for (c = sx; c < ex; ++c) {				// slide in x on input
//            int accumulator = 0;
//            for (rd = wsy; rd < wey; ++rd) {		//	slide in y on weight filter
//                for (cd = wsx; cd < wex; ++cd) {	//	slide in x on weight filter
//
//                	// calculates the index of data in the input data array (y*width + x)
//                	int iidx = (r+rd)*px + (c+cd);
//                    BINARY_WORD pixel = input[iidx];
//
//                    // calculates the index of data in the weights data array (y*width + x)
//                    int widx = (rd + wy_2)*wx + (cd+wx_2);
//                    BINARY_WORD weight = weights[widx];
//
//                    // binary convolution operation
//                    accumulator += __builtin_popcount(~(pixel ^ weight));
//                }
//            }
//            // write to output, padded space
//            int oidx = (r-wy_2)*output_width + (c-wx_2);
//            oidx += filter_iter_base;
//            output[oidx] += (float) accumulator;
//        }
//    }
//};
//
///**
// * pointwise multiplication of two array
// */
//inline void pointwise_mul_mm(float *output, const float *input, int step_size){
//    int i = 0;
//
//    //!!!!! Why? !!!!!
//    /*while (i + 8 <= step_size) {
//        output[i+0] *= input[i+0];
//        output[i+1] *= input[i+1];
//        output[i+2] *= input[i+2];
//        output[i+3] *= input[i+3];
//        output[i+4] *= input[i+4];
//        output[i+5] *= input[i+5];
//        output[i+6] *= input[i+6];
//        output[i+7] *= input[i+7];
//
//        i += 8;
//    }*/
//
//    while (++i < step_size) // finish iteration leftover
//         output[i] *= input[i];
//};
//
///**
// * Performs a tiled pointwise matrix multiplication between two 2D tensors
// * Pre-conditions: wx < ix, and wy < iy
// */
//inline void pointwise_mul_mm_2D(float *output, const float *alpha,
//								int input_w, int input_h, int filter_w, int filter_h,
//								int pad_x, int pad_y){
//// Slower version
////      for (int y = 0; y < input_h; ++y)
////          for (int x = 0; x < input_w; x++)
////              output[y*input_w+x] *= input[(y % filter_h)*filter_w + (x % filter_w)];
//
//	int padded_input_w = input_w+2*pad_x;
//
//    // Stride prefetch optimized
//    for (int s = 0; s < filter_h; ++s) {  // for each strip
//
//    	const float *strip_ptr = &alpha[s*filter_w];
//
//        for (int y = pad_y; y < pad_y + (input_h / filter_h); ++y) {   //
//            int stride = y*(padded_input_w*filter_h) + s*padded_input_w;
//            float *output_ptr = &output[stride];
//
//            for (int x = 0; x < input_w; ++x) {
//                output_ptr[x] *= strip_ptr[x % filter_w];
//            }
//        }
//    }
//};
//
///**
// * Description: this function will perform the binary convolution for the input
// *  binary layer.
// * params:
// *  BinaryLayer: which contains structure and data that the binary convolution required.
// */
//inline void xnor_forward(std::unique_ptr<mxnet::op::BinaryLayer> const &binary_layer) {
//	CHECK(binary_layer->binary_input != nullptr) << "xnor_forward: must init layer input";
//	CHECK(binary_layer->binary_weights != nullptr) << "xnor_forward: must init layer weights";
//	CHECK(binary_layer->output != nullptr) << "xnor_forward: must set layer output";
//	CHECK(binary_layer->alpha != nullptr) << "xnor_forward: must init weight scaling factor alpha";
//	CHECK(binary_layer->beta != nullptr) << "xnor_forward: must init input scaling factor beta";
//
//
//	//======== TODO: able to support arbitrary channel size ==========//
//	CHECK_EQ(binary_layer->input_channels % 32, 0) << "Channel is not divisible by 32."
//												"before supporting arbitrary channel size. For now, "
//												"set the channel size to the nearest multiple of 32 "
//												"and ignore any ''extra'' channels unused.";
//
//	//smaller the input channel number, divided by 32, because we will process per word 32 bit number
//	//later.
//	int input_channels_mod_bits = binary_layer->input_channels / BITS_PER_BINARY_WORD;   // 32
//    //===============================================================//
//
//    // padded input size
//    int padded_w = (int) binary_layer->input_width + 2*binary_layer->padding_x;
//    int padded_h = (int) binary_layer->input_height + 2*binary_layer->padding_y;
//
//    BINARY_WORD *binary_weights = binary_layer->binary_weights;
//
//    // do forward calc
//    for (int z = 0; z < binary_layer->num_filters; ++z) {    // for each filter map
//        BINARY_WORD *binary_input = binary_layer->binary_input;
//        for (int c = 0; c < input_channels_mod_bits; ++c) {    // for each input channel
//        	binary_conv2D(binary_layer->output, binary_input, binary_weights,
//        						binary_layer->input_width, binary_layer->input_height,
//								binary_layer->kernel_width, binary_layer->kernel_height,
//								binary_layer->padding_x, binary_layer->padding_y, binary_layer->stride,
//								binary_layer->output_width, binary_layer->output_height,
//								z*binary_layer->output_width*binary_layer->output_height);
//
//        	// increment with next input image
//        	//length of binary_input: input_channels(original) * input_w * input_h / BITS_PER_BINARY_WORD
//            *binary_input += padded_w * padded_h;
//
//            //length of binary_weights: num_filters * input_channels(original) * kernel_width * kernel_heihgt / BITS_PER_BINARY_WORD
//            *binary_weights += binary_layer->kernel_width * binary_layer->kernel_height;
//
//            //====== !!NON-binary operations!! =======//
//            /*pointwise_mul_mm(binary_layer->output, binary_layer->beta, padded_w * padded_h);
//            pointwise_mul_mm_2D(binary_layer->output, binary_layer->alpha, binary_layer->output_width, binary_layer->output_height,
//            		binary_layer->kernel_width, binary_layer->kernel_height,
//					binary_layer->padding_x, binary_layer->padding_y);
//            *///=======================================//
//        }
//    }
//
//};

} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
#endif //MXNET_XNOR_CPU_H
