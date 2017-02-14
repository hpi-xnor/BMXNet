/*!
 * Copyright (c) 2017 by Contributors
 * \file xnor_cpu.h
 * \brief implementation of xnor-gemm operator for cpu
 * \author HPI-DeepLearning
*/

#ifndef MXNET_XNOR_CPU_H
#define MXNET_XNOR_CPU_H

#include <dmlc/logging.h>
#include "./q_helper.h"

namespace mxnet {
namespace op {
namespace xnor_cpu {
/*
* Description: this function will perform the binary convolution for the input
*  binary layer.
* params:
*  BinaryLayer: which contains structure and data that the binary convolution required.
*/
template<typename Dtype>
inline void XnorForward(BinaryLayer* binary_layer) {
	CHECK_EQ(binary_layer->input, NULL) << "xnor_forward: must init layer input";
	CHECK_EQ(binary_layer->weights, NULL) << "xnor_forward: must init layer weights";
	CHECK_EQ(binary_layer->output, NULL) << "xnor_forward: must set layer output";
	CHECK_EQ(binary_layer->alpha, NULL) << "xnor_forward: must init weight scaling factor alpha";
	CHECK_EQ(binary_layer->beta, NULL) << "xnor_forward: must init input scaling factor beta";
	CHECK(binary_layer->channels % 32 != 0) << "Channel is not divisible by 32. Need to implement mask "
												"before supporting arbitrary channel size. For now, "
												"set the channel size to the nearest multiple of 32 "
												"and ignore any ''extra'' channels unused.";

	//todo:
	l->c /= BITS_PER_BINARY_WORD;   // For compensating with doing more work per word

    float *output = l->output;
    float *alpha = l->alpha;
    float *beta = l->beta;
    int px = l->px;
    int py = l->py;
    BINARY_WORD *binary_weights = l->binary_weights;

    for (int z = 0; z < l->batch; ++z) {    // for each filter map
        BINARY_WORD *binary_input = l->binary_input;
        for (int c = 0; c < l->c; ++c) {    // for each input channel
            ai2_bin_conv2D(output, binary_input, binary_weights, l->w, l->h, l->wx, l->wy, l->pad, l->stride);
            binary_input += px*py;   // increment with next 2D plane
            binary_weights += l->wx*l->wy;       // increment with next 2D plane

            ai2_pointwise_mul_mm(output, beta, px*py);
            ai2_pointwise_mul_mm_2d(output, alpha, l->w, l->h, l->wx, l->wy, l->pad);
        }
    }

};

} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
#endif //MXNET_XNOR_CPU_H
