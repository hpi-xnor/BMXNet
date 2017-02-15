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
          BINARY_WORD *binary_input;
          BINARY_WORD *binary_weights;
          float *output;
          float *alpha;
          float *beta;

          int input_channels;
          int input_width;
          int input_height;
          int num_filters;
          int kernel_width;
          int kernel_height;
          int padding_x;
          int padding_y;
          int stride = 1;

          BinaryLayer(int channels, int input_width, int input_height, int num_filters, int kernel_width, int kernel_height, int padding_x, int padding_y):
                  input_channels(channels),
                  input_width(input_width),
                  input_height(input_height),
                  num_filters(num_filters),
                  kernel_width(kernel_width),
                  kernel_height(kernel_height),
                  padding_x(padding_x),
                  padding_y(padding_y)
            {
            //malloc etc
            float output_size = ((input_width - input_height + padding_x + padding_y) / stride) + 1;
            //assert(false);
          }

          ~BinaryLayer() {
            //free
          }

          void set_inputs() {

          }

          void set_weights() {

          }

          void get_output() {

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
