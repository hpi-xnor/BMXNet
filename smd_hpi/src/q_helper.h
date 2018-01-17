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

namespace mxnet {
  namespace op {
    namespace helper {
        using mshadow::expr::F;
        using mshadow::expr::ScalarExp;
        using mshadow::expr::scalar;

        //wrapper function for CUDA kernel        
        extern "C" float launch_max_reduce(float*, int);
        
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

        // Launches CUDA max_reduce kernel.
        // Note that currently only "float" supported 
        template<int dim>
        inline float amax(const mshadow::Tensor<gpu, dim, float> &tensor) {           
          int tensor_size = tensor.size(0);
          float * input = tensor.dptr_;
          float max = launch_max_reduce(input, tensor_size);
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

        /*
         * Quantization for dataflow.
         * bit width > 1:
         *  Forward: w_i = 2 * quantize_k-bit(tanh(x_i)/2*max(|tanh(x_i)|) + 1/2) - 1
         */
        template<typename xpu, int dim, typename DType>
        inline void quantize_weights(mshadow::Tensor<xpu, dim, DType> &dataflow, unsigned int bit_width) {
          if (bit_width == 1) {
            real_t scaling_factor = 1;
            dataflow = F<mshadow_op::det_sign>(dataflow / ScalarExp<DType>(scaling_factor)) *
                      ScalarExp<DType>(scaling_factor);
          } else if (bit_width < 32) {
            mshadow::Tensor<xpu, 1, DType> workspace = mshadow::NewTensor<xpu>(dataflow.shape_, DType(1.0), true, dataflow.stream_);
            workspace = F<mshadow_op::abs>(F<mshadow_op::tanh>(dataflow));

            DType max = amax(workspace);            

            dataflow = scalar(DType(2.0)) *
                      F<mshadow_op::quantize>(
                              F<mshadow_op::tanh>(dataflow) / scalar(DType(2.0) * max) + scalar(DType(0.5)),
                              scalar(DType(bit_width))) 
                              - scalar(DType(1.0));            

            mshadow::FreeSpace(&workspace);
          }
        };

        /*
         * Quantization for activations (inputs).
         */
        template<typename xpu, int dim, typename DType>
        inline void quantize_activations(mshadow::Tensor<xpu, dim, DType> &dataflow, unsigned int bit_width) {
          if (bit_width == 1) {
            dataflow = F<mshadow_op::det_sign>(dataflow);
          } else {         
            dataflow = F<mshadow_op::quantize>(
                                                F<mshadow_op::maximum>(
                                                  F<mshadow_op::minimum>(
                                                    dataflow, 
                                                    scalar(DType(1))), 
                                                  scalar(DType(0))), //clip to [0, 1]
                                                scalar(DType(bit_width))
                                              );
          }
        };        
    }
  }
}
#endif //MXNET_Q_HELPER_H
