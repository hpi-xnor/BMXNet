/*!
 * Copyright (c) 2017 by Contributors
 * \file xnor_cpu.h
 * \brief implementation of xnor-gemm operator for cpu
 * \author HPI-DeepLearning
*/

#ifndef MXNET_XNOR_CPU_H
#define MXNET_XNOR_CPU_H

#include <dmlc/logging.h>
#include "binary_layer.h"
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

/**
 * binary gemm. instead of standard dot product
 * we apply binary_dot: _popcount( xnor() ) operators to perform the convolution
 *
 * params:
 * 	weights: (m x n)
 * 	col_input: inputs, unpacked via patch2col (NOT n x k, !BUT TRANSPOSED!: k x n)
 * 	output: (m x k)
 * 	m, n, k: size of matrices
 */

  inline void binary_gemm(BINARY_WORD* weights, BINARY_WORD* col_input, float* output, int m, int n, int k) {
    CHECK_EQ(n % 32, 0) << "!!! no masking yet, only input channel % 32==0";

    int bitwords_per_row = n / BITS_PER_BINARY_WORD;

    for (int mi = 0; mi < m; mi++) {
      for (int ki = 0; ki < k; ki++) {
        float accum = 0;
        for (int bitword_index_in_row = 0; bitword_index_in_row < bitwords_per_row; bitword_index_in_row++) {
          // masking or only 32bit support important cause !gaah!
          BINARY_WORD pixel = col_input[ki * bitwords_per_row + bitword_index_in_row];
          BINARY_WORD weight = weights[mi * bitwords_per_row + bitword_index_in_row];
          accum += __builtin_popcount(~(pixel ^ weight));
        }

        output[mi * k + ki] = accum;
      }
    }
  }

/**
 * binary convolution implementation. instead of standard dot product
 * we apply binary_dot: _popcount( xnor() ) operators to perform the convolution
 * on binary input(I) and weight matrix (W), the alpha is the scaling factor
 * for W, 2D_beta consist of all scaling factor beta for the input tensor.
 * The calculation follows the equation:
 * 		I * W ≈ (sign(I) (binary_dot) sign(W)) (dot) (2D_beta)(alpha)
 *
 * params:
 * 	output:output data array
 * 	input: input tensor
 * 	weights: weight filter
 */

inline void binary_conv2D(float* output,  const BINARY_WORD *input,
						  const BINARY_WORD *weights, int ix, int iy,
						  int wx, int wy, int pad_x, int pad_y, int stride,
						  int output_width, int output_height, int filter_iter_base) {
    int r, rd, c, cd;
    int wx_2 = wx / 2;
    int wy_2 = wy / 2;

    // Indexing for weights
    int wsx, wex, wsy, wey;
    wsx = -wx_2;				// weight start x
    wsy = -wy_2;	 			// weight start y

    if (wx % 2 == 1)  		// odd weights w
        wex = wx_2 + 1;			// weight end x
    else
        wex = wx_2;
    if (wy % 2 == 1)  		// odd weights h
		wey = wy_2 + 1;			// weight end y
	else
		wey = wy_2;

    // Indexing for input pixels. since stride can only be 1 now,
    int sx = pad_x + wx_2;               // start x
    int ex = ix + pad_x - wx_2;      // end x
    int sy = pad_y + wy_2;               // start y
    int ey = iy + pad_y - wy_2;      // end y

    //padded input width
    int px = ix + 2*pad_x;

    for (r = sy; r < ey; ++r) { 					// slide in y on input
        for (c = sx; c < ex; ++c) {				// slide in x on input
            int accumulator = 0;
            for (rd = wsy; rd < wey; ++rd) {		//	slide in y on weight filter
                for (cd = wsx; cd < wex; ++cd) {	//	slide in x on weight filter

                	// calculates the index of data in the input data array (y*width + x)
                	int iidx = (r+rd)*px + (c+cd);
                    BINARY_WORD pixel = input[iidx];

                    // calculates the index of data in the weights data array (y*width + x)
                    int widx = (rd + wy_2)*wx + (cd+wx_2);
                    BINARY_WORD weight = weights[widx];

                    // binary convolution operation
                    accumulator += __builtin_popcount(~(pixel ^ weight));
                }
            }
            // write to output, padded space
            int oidx = (r-wy_2)*output_width + (c-wx_2);
            oidx += filter_iter_base;
            output[oidx] += (float) accumulator;
        }
    }
};

/**
 * pointwise multiplication of two array
 */
inline void pointwise_mul_mm(float *output, const float *input, int step_size){
    int i = 0;

    //!!!!! Why? !!!!!
    /*while (i + 8 <= step_size) {
        output[i+0] *= input[i+0];
        output[i+1] *= input[i+1];
        output[i+2] *= input[i+2];
        output[i+3] *= input[i+3];
        output[i+4] *= input[i+4];
        output[i+5] *= input[i+5];
        output[i+6] *= input[i+6];
        output[i+7] *= input[i+7];

        i += 8;
    }*/

    while (++i < step_size) // finish iteration leftover
         output[i] *= input[i];
};

/**
 * Performs a tiled pointwise matrix multiplication between two 2D tensors
 * Pre-conditions: wx < ix, and wy < iy
 */
inline void pointwise_mul_mm_2D(float *output, const float *alpha,
								int input_w, int input_h, int filter_w, int filter_h,
								int pad_x, int pad_y){
// Slower version
//      for (int y = 0; y < input_h; ++y)
//          for (int x = 0; x < input_w; x++)
//              output[y*input_w+x] *= input[(y % filter_h)*filter_w + (x % filter_w)];

	int padded_input_w = input_w+2*pad_x;

    // Stride prefetch optimized
    for (int s = 0; s < filter_h; ++s) {  // for each strip

    	const float *strip_ptr = &alpha[s*filter_w];

        for (int y = pad_y; y < pad_y + (input_h / filter_h); ++y) {   //
            int stride = y*(padded_input_w*filter_h) + s*padded_input_w;
            float *output_ptr = &output[stride];

            for (int x = 0; x < input_w; ++x) {
                output_ptr[x] *= strip_ptr[x % filter_w];
            }
        }
    }
};

/**
 * Description: this function will perform the binary convolution for the input
 *  binary layer.
 * params:
 *  BinaryLayer: which contains structure and data that the binary convolution required.
 */
inline void xnor_forward(std::unique_ptr<mxnet::op::BinaryLayer> const &binary_layer) {
	CHECK(binary_layer->binary_input != nullptr) << "xnor_forward: must init layer input";
	CHECK(binary_layer->binary_weights != nullptr) << "xnor_forward: must init layer weights";
	CHECK(binary_layer->output != nullptr) << "xnor_forward: must set layer output";
	CHECK(binary_layer->alpha != nullptr) << "xnor_forward: must init weight scaling factor alpha";
	CHECK(binary_layer->beta != nullptr) << "xnor_forward: must init input scaling factor beta";


	//======== TODO: able to support arbitrary channel size ==========//
	CHECK_EQ(binary_layer->input_channels % 32, 0) << "Channel is not divisible by 32."
												"before supporting arbitrary channel size. For now, "
												"set the channel size to the nearest multiple of 32 "
												"and ignore any ''extra'' channels unused.";

	//smaller the input channel number, divided by 32, because we will process per word 32 bit number
	//later.
	int input_channels_mod_bits = binary_layer->input_channels / BITS_PER_BINARY_WORD;   // 32
    //===============================================================//

    // padded input size
    int padded_w = (int) binary_layer->input_width + 2*binary_layer->padding_x;
    int padded_h = (int) binary_layer->input_height + 2*binary_layer->padding_y;

    BINARY_WORD *binary_weights = binary_layer->binary_weights;

    // do forward calc
    for (int z = 0; z < binary_layer->num_filters; ++z) {    // for each filter map
        BINARY_WORD *binary_input = binary_layer->binary_input;
        for (int c = 0; c < input_channels_mod_bits; ++c) {    // for each input channel
        	binary_conv2D(binary_layer->output, binary_input, binary_weights,
        						binary_layer->input_width, binary_layer->input_height,
								binary_layer->kernel_width, binary_layer->kernel_height,
								binary_layer->padding_x, binary_layer->padding_y, binary_layer->stride,
								binary_layer->output_width, binary_layer->output_height,
								z*binary_layer->output_width*binary_layer->output_height);

        	// increment with next input image
        	//length of binary_input: input_channels(original) * input_w * input_h / BITS_PER_BINARY_WORD
            *binary_input += padded_w * padded_h;

            //length of binary_weights: num_filters * input_channels(original) * kernel_width * kernel_heihgt / BITS_PER_BINARY_WORD
            *binary_weights += binary_layer->kernel_width * binary_layer->kernel_height;

            //====== !!NON-binary operations!! =======//
            /*pointwise_mul_mm(binary_layer->output, binary_layer->beta, padded_w * padded_h);
            pointwise_mul_mm_2D(binary_layer->output, binary_layer->alpha, binary_layer->output_width, binary_layer->output_height,
            		binary_layer->kernel_width, binary_layer->kernel_height,
					binary_layer->padding_x, binary_layer->padding_y);
            *///=======================================//
        }
    }

};

} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
#endif //MXNET_XNOR_CPU_H
