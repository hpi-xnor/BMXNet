/*!
 * Copyright (c) 2017 by Contributors
 * \file q_helper.h
 * \brief Quantization helper function
 * \author HPI-DeepLearning
*/

#ifndef MXNET_Q_HELPER_H
#define MXNET_Q_HELPER_H

#include "../../src/operator/mshadow_op.h"
#include <mshadow/tensor.h>
#include <mshadow/expression.h>
#include <cstdlib>

typedef uint32_t BINARY_WORD;
#define BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT)

namespace mxnet {
  namespace op {
    namespace helper {
        using mshadow::expr::F;
        using mshadow::expr::ScalarExp;
        using mshadow::expr::scalar;


        // @todo naive implementation |==> this needs to be implemented nicely and with gpu support (see nvidia pdf on reduction with cuda)
        // GPU (includes copy to CPU)
        template<int dim, typename DType>
        inline DType amax(const mshadow::Tensor<gpu, dim, DType> &tensor) {
          mshadow::Tensor<cpu, 1, DType> tensor_cpu = mshadow::NewTensor<cpu>(tensor.shape_, DType(1.0));

          mshadow::Copy(tensor_cpu, tensor, tensor.stream_);

          DType max = 0;
          for (index_t i = 0; i < tensor_cpu.size(0); ++i) {
            if (tensor_cpu[i] > max) {
              max = tensor_cpu[i];
            }
          }

          mshadow::FreeSpace(&tensor_cpu);

          return max;
        }

        // CPU only
        template<int dim, typename DType>
        inline DType amax(const mshadow::Tensor<cpu, dim, DType> &tensor) {
          DType max = 0;
          for (index_t i = 0; i < tensor.size(0); ++i) {
            if (tensor[i] > max) {
              max = tensor[i];
            }
          }

          return max;
        }


        template<typename xpu, typename DType>
        inline void quantize(mshadow::Tensor<xpu, 1, DType> &weights, mshadow::Tensor<xpu, 1, DType> &workspace,
                             unsigned int act_bit) {
          if (act_bit == 1) {
            real_t scaling_factor = 1;
            weights = F<mshadow_op::det_sign>(weights / ScalarExp<DType>(scaling_factor)) *
                      ScalarExp<DType>(scaling_factor);
          } else if (act_bit < 32) {
            workspace = F<mshadow_op::abs>(F<mshadow_op::tanh>(weights));

            DType max = amax(workspace);

            weights = scalar(DType(2.0)) *
                      F<mshadow_op::quantize>(
                              F<mshadow_op::tanh>(weights) / scalar(DType(2.0) * max) + scalar(DType(0.5)),
                              scalar(DType(act_bit)))
                      - scalar(DType(1.0));
          }
        };

        class BinaryLayer {
        public:
          BINARY_WORD *binary_input = nullptr;
          BINARY_WORD *binary_weights = nullptr;
          float *output = nullptr;
          float *alpha = nullptr;
          float *beta = nullptr;

          int input_channels;
          int input_width;
          int input_height;
          int num_filters;
          int kernel_width;
          int kernel_height;
          int padding_x;
          int padding_y;
          int stride = 1;

          BinaryLayer(int input_channels, int input_width, int input_height, int num_filters, int kernel_width, int kernel_height, int padding_x, int padding_y):
                  input_channels(input_channels),
                  input_width(input_width),
                  input_height(input_height),
                  num_filters(num_filters),
                  kernel_width(kernel_width),
                  kernel_height(kernel_height),
                  padding_x(padding_x),
                  padding_y(padding_y) {

            CHECK_EQ(padding_x, padding_y) << "differing padding in x and y direction, unknown if supported";

            float output_size = ((input_width - kernel_width + 2 * padding_x) / stride) + 1;

            CHECK_EQ(ceilf(output_size), output_size) << "invalid output size of binary convolution layer: " << output_size;

            // padded input size
            int input_width_padded = input_width + 2 * padding_x;
            int input_height_padded = input_height + 2 * padding_y;

            binary_input = (BINARY_WORD *) calloc(input_channels * input_width_padded * input_height_padded / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
            binary_weights = (BINARY_WORD *) calloc(num_filters * input_channels * kernel_width * kernel_height / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));

            output = (float *) calloc(input_channels * input_width_padded * input_height_padded, sizeof(float));
          }

          ~BinaryLayer() {
            if (binary_input) free(binary_input);
            if (binary_weights) free(binary_weights);
            if (output) free(output);
            if (alpha) free(alpha);
            if (beta) free(beta);
          }

#define SetBit(A,k)     ( A |= (1 << (k)) )
#define ClearBit(A,k)   ( A &= ~(1 << (k)) )
#define TestBit(A,k)    ( A & (1 << (k)) )

          void float_to_binary(mshadow::Tensor<cpu, 3, float> input, BINARY_WORD *output) {
            for (int i = 0; i < input.size(0) * input.size(1) * input.size(2); i += BITS_PER_BINARY_WORD) {
              BINARY_WORD tmp = 0x00000000;
              for (int x = 0; x < BITS_PER_BINARY_WORD; ++x) {
                float tmp_flt = input.dptr_[x + i];
                if (signbit(input.dptr_[x + i]) == 0) SetBit(tmp, (BITS_PER_BINARY_WORD - 1) - x);
              }
              output[i / BITS_PER_BINARY_WORD] = tmp;
            }
          }

          // calculate mean of first dimension accross second and third dimension and save as 2d plane
          void calculate_alpha(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume) {
            //layout of input_volume: depth|x|y
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

          // ???
          void calculate_beta(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume) {
            calculate_alpha(output_plane, input_volume);
          }

          void set_inputs(mshadow::Tensor<cpu, 3, float> input) {
            float_to_binary(input, binary_input);

            if (beta) free(beta);
            beta = (float *) calloc (input.size(1) * input.size(2), sizeof(float));

            calculate_beta(beta, input);
          }

          void set_weights(const mshadow::Tensor<cpu, 3, float> &wmat) {
            float_to_binary(wmat, binary_weights);

            if (alpha) free(alpha);
            alpha = (float *) calloc (wmat.size(1) * wmat.size(2), sizeof(float));

            calculate_alpha(alpha, wmat);
          }

          void get_output(const mshadow::Tensor<cpu, 3, float> &out) {

          }
        };

//        void  SetBit( int A[],  int k )
//        {
//          int i = k/32;        //gives the corresponding index in the array A
//          int pos = k%32;      //gives the corresponding bit position in A[i]
//
//          unsigned int flag = 1;   // flag = 0000.....00001
//
//          flag = flag << pos;      // flag = 0000...010...000   (shifted k positions)
//
//          A[i] = A[i] | flag;      // Set the bit at the k-th position in A[i]
//        }
    }
  }
}
#endif //MXNET_Q_HELPER_H
