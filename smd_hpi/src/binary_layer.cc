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

BinaryLayer::BinaryLayer(int input_channels, int input_width, int input_height, int num_filters, int kernel_width, int kernel_height, int padding_x, int padding_y, int m, int n, int k):
        input_channels(input_channels),
        input_width(input_width),
        input_height(input_height),
        num_filters(num_filters),
        kernel_width(kernel_width),
        kernel_height(kernel_height),
        padding_x(padding_x),
        padding_y(padding_y),
        m(m),
        n(n),
        k(k){
  CHECK_EQ(padding_x, padding_y) << "differing padding in x and y direction, unknown if supported";

  output_width = ((input_width - kernel_width + 2 * padding_x) / stride) + 1;
  output_height = ((input_height - kernel_height + 2 * padding_y) / stride) + 1;

  CHECK_EQ(m, num_filters);
  CHECK_EQ(n, kernel_width * kernel_height * input_channels);
  CHECK_EQ(k, output_width * output_height * 100);

  // padded input size
  int input_width_padded = input_width + 2 * padding_x;
  int input_height_padded = input_height + 2 * padding_y;

  binary_input = (BINARY_WORD *) calloc(n * k / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
  binary_weights = (BINARY_WORD *) calloc(m * n / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));

  output = (float *) calloc(m * k, sizeof(float));
}

BinaryLayer::~BinaryLayer() {
  if (binary_input) free(binary_input);
  if (binary_weights) free(binary_weights);
  if (output) free(output);
  if (alpha) free(alpha);
  if (beta) free(beta);
}

void BinaryLayer::set_input_as_col(const mshadow::Tensor<cpu, 2, float> &input) {
  float_to_binary(input, binary_input);

//  if (beta) free(beta);
//  beta = (float *) calloc (input.size(1) * input.size(2), sizeof(float));
//
//  calculate_beta(beta, input);
}

void BinaryLayer::set_weights(const mshadow::Tensor<cpu, 2, float> &wmat) {
  float_to_binary(wmat, binary_weights);

//      if (alpha) free(alpha);
//      alpha = (float *) calloc (wmat.size(1) * wmat.size(2), sizeof(float));
//
//      calculate_alpha(alpha, wmat);
}

void BinaryLayer::get_output(const mshadow::Tensor<cpu, 2, float> &out) {
  // @todo: what about padding?
  memcpy(out.dptr_, output, out.size(0) * out.size(1) * sizeof(float));
  //binary_to_float(out);
}

std::string BinaryLayer::weights_as_string() {
  std::ostringstream output_stream;

  output_stream << "WARN: not sure if order inside filters is correct!\n";

  for (int filter = 0; filter < num_filters; filter++) {
    output_stream << "filter[" << filter << "] ";
    for (int input_channel = 0; input_channel < input_channels; input_channel++) {
      output_stream << "channel[" << input_channel << "] <";
      for (int kx = 0; kx < kernel_width; kx++) {
        for (int ky = 0; ky < kernel_height; ky++) {
          int position = filter * m +
                         input_channel * kernel_height * kernel_width +
                         kx * kernel_height + ky;
          BINARY_WORD tmp = binary_weights[position / 32];
          if (TestBit(tmp, position % 32)) {
            output_stream << "1";
          } else {
            output_stream << "0";
          }
        }
      }
      output_stream << "> ";
    }
    output_stream << "\n";
  }

  return output_stream.str();
}
//  private  -----------------------------------------------------------------------------------------------------------

/* @brief converts a 3d float tensor into a bitset
*
* @param input 3d float tensor
* @param output pointer to zero-initialized memory for the bitset
*/
void BinaryLayer::float_to_binary(const mshadow::Tensor<cpu, 2, float> &input, BINARY_WORD *output) {
  // @todo: does this work and not run over the size of dptr_/miss some of it if it doesnt divide by 32?
  int total_elements = input.size(0) * input.size(1);
  for (int i = 0; i < total_elements; i += BITS_PER_BINARY_WORD) {
    BINARY_WORD tmp = 0x00000000;
    int step_end = std::min<int>(total_elements, i + BITS_PER_BINARY_WORD);
    // @todo: why do we reverse the order inside one word? endianes?
    for (int x = i; x < step_end; ++x) {
      if (std::signbit(input.dptr_[x]) == 0) SetBit(tmp, (BITS_PER_BINARY_WORD - 1) - x % BITS_PER_BINARY_WORD);
    }
    output[i / BITS_PER_BINARY_WORD] = tmp;
  }
}

/* @brief converts a bitset into a float tensor
*
* @param out a 3d output tensor
*/
void BinaryLayer::binary_to_float(BINARY_WORD *input, const mshadow::Tensor<cpu, 2, float> &out) {
  int total_elements = out.size(0) * out.size(1);
  for (int i = 0; i < (total_elements + (BITS_PER_BINARY_WORD - 1)); i += BITS_PER_BINARY_WORD) {
    BINARY_WORD tmp = (BINARY_WORD) input[i / BITS_PER_BINARY_WORD];
    int step_end = std::min<int>(total_elements, i + BITS_PER_BINARY_WORD);
    for (int x = i; x < step_end; ++x) {
      if (TestBit(tmp, (BITS_PER_BINARY_WORD - 1) - x % BITS_PER_BINARY_WORD)) {
        out.dptr_[x] = 1.f;
      } else {
        out.dptr_[x] = -1.f;
      }
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
        accum += input_volume.dptr_[out * depth + z];
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

