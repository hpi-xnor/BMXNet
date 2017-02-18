/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_layer.h
 * \brief implementation of xnor-gemm operator for cpu
 * \author HPI-DeepLearning
*/

#include "binary_layer.h"
#include <cstdlib>

namespace mxnet {
namespace op {

#define SetBit(A,k)     ( A |= (1 << (k)) )
//#define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
//#define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )
//#define ClearBit(A,k)   ( A &= ~(1 << (k)) )
#define TestBit(A,k)    ( A & (1 << (k)) )
//#define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )


//  public  ------------------------------------------------------------------------------------------------------------

BinaryLayer::BinaryLayer(int input_channels, int input_width, int input_height, int num_filters, int kernel_width, int kernel_height, int padding_x, int padding_y):
        input_channels(input_channels),
        input_width(input_width),
        input_height(input_height),
        num_filters(num_filters),
        kernel_width(kernel_width),
        kernel_height(kernel_height),
        padding_x(padding_x),
        padding_y(padding_y) {
  CHECK_EQ(padding_x, padding_y) << "differing padding in x and y direction, unknown if supported";

  output_width = ((input_width - kernel_width + 2 * padding_x) / stride) + 1;
  output_height = ((input_height - kernel_height + 2 * padding_y) / stride) + 1;

  //CHECK_EQ(ceilf(output_w), output_w) << "invalid output size of binary convolution layer: " << output_w;

  // padded input size
  int input_width_padded = input_width + 2 * padding_x;
  int input_height_padded = input_height + 2 * padding_y;

  binary_input = (BINARY_WORD *) calloc(input_channels * input_width_padded * input_height_padded / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
  binary_weights = (BINARY_WORD *) calloc(num_filters * input_channels * kernel_width * kernel_height / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));

  output = (float *) calloc(num_filters * output_width * output_height, sizeof(float));
}

BinaryLayer::~BinaryLayer() {
  if (binary_input) free(binary_input);
  if (binary_weights) free(binary_weights);
  if (output) free(output);
  if (alpha) free(alpha);
  if (beta) free(beta);
}

void BinaryLayer::set_inputs(const mshadow::Tensor<cpu, 3, float> input) {
  float_to_binary(input, binary_input);

  if (beta) free(beta);
  beta = (float *) calloc (input.size(1) * input.size(2), sizeof(float));

  calculate_beta(beta, input);
}

void BinaryLayer::set_weights(const mshadow::Tensor<cpu, 3, float> &wmat) {
  float_to_binary(wmat, binary_weights);

  if (alpha) free(alpha);
  alpha = (float *) calloc (wmat.size(1) * wmat.size(2), sizeof(float));

  calculate_alpha(alpha, wmat);
}

void BinaryLayer::get_output(const mshadow::Tensor<cpu, 3, float> &out) {
  // @todo: what about padding?
  memcpy(out.dptr_, output, out.size(0) * out.size(1) * out.size(2) * sizeof(float));
  //binary_to_float(out);
}
//  private  -----------------------------------------------------------------------------------------------------------

/* @brief converts a 3d float tensor into a bitset
 *
 * @param input 3d float tensor
 * @param output pointer to zero-initialized memory for the bitset
 */
void BinaryLayer::float_to_binary(mshadow::Tensor<cpu, 3, float> input, BINARY_WORD *output) {
  // @todo: does this work and not run over the size of dptr_/miss some of it if it doesnt divide by 32?
  for (int i = 0; i < input.size(0) * input.size(1) * input.size(2); i += BITS_PER_BINARY_WORD) {
    BINARY_WORD tmp = 0x00000000;
    // @todo: why do we reverse the order inside one word? endianes?
    for (int x = 0; x < BITS_PER_BINARY_WORD; ++x) {
      if (std::signbit(input.dptr_[x + i]) == 0) SetBit(tmp, (BITS_PER_BINARY_WORD - 1) - x);
    }
    output[i / BITS_PER_BINARY_WORD] = tmp;
  }
}

/* @brief converts a bitset into a float tensor
 *
 * @param out a 3d output tensor
 */
void BinaryLayer::binary_to_float(const mshadow::Tensor<cpu, 3, float> &out) {
  CHECK(false) << "this method is untested!";
  int total_elements = out.size(0) * out.size(1) * out.size(2);

  for (int i = 0; i < total_elements; i += BITS_PER_BINARY_WORD) {
    BINARY_WORD tmp = (BINARY_WORD) output[i / BITS_PER_BINARY_WORD];
    for (int x = 0; x < BITS_PER_BINARY_WORD; ++x) {
      if (TestBit(tmp, (BITS_PER_BINARY_WORD - 1) - x)) {
        out.dptr_[i + x] = 1.f;
      } else {
        out.dptr_[i + x] = -1.f;
      }
    }
  }
  if (total_elements % BITS_PER_BINARY_WORD == 0) {
    return;
  }
  // there are some bits left
  BINARY_WORD tmp = (BINARY_WORD) output[total_elements / BITS_PER_BINARY_WORD];
  for (int i = total_elements - (total_elements % BITS_PER_BINARY_WORD); i < total_elements; i++) {
    if (TestBit(tmp, (BITS_PER_BINARY_WORD - 1) - i % BITS_PER_BINARY_WORD)) {
      out.dptr_[i] = 1.f;
    } else {
      out.dptr_[i] = -1.f;
    }
  }
}

/* @brief calculate mean of first dimension accross second and third dimension and save as 2d plane
 *
 * @param output_plane at every x,y this plane will contain the mean of all z values in input
 * @param input_volume expected layout: z,x,y
 */
void BinaryLayer::calculate_alpha(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume) {
  int depth = input_volume.size(0);
  int width = input_volume.size(1);
  int height = input_volume.size(2);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int out = y * width + x;
      float accum = 0.0;
      for (int z = 0; z < depth; ++z) {
        accum += fabs(input_volume.dptr_[out * depth + z]);
      }

      output_plane[out] = accum / depth;
    }
  }
}

// @todo ???
void BinaryLayer::calculate_beta(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume) {
  calculate_alpha(output_plane, input_volume);
}

}} // namespace mxnet { namespace op {

