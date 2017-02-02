/*!
 * Copyright (c) 2017 by Contributors
 * \file q_helper.h
 * \brief Quantization helper function
 * \author HPI-DeepLearning
*/

#ifndef MXNET_Q_HELPER_H
#define MXNET_Q_HELPER_H

#include "./mshadow_op.h"
#include <mshadow/tensor.h>
#include <mshadow/expression.h>

namespace mxnet {
  namespace op {
    namespace helper {
        using mshadow::expr::F;
        using mshadow::expr::ScalarExp;
        using mshadow::expr::scalar;

        template<typename DType>
        inline void quantize(mshadow::Tensor<cpu, 1, DType> &weights, mshadow::Tensor<cpu, 1, DType> &workspace,
                             unsigned int act_bit) {
          if (act_bit == 1) {
            real_t scaling_factor = 1;
            weights = F<mshadow_op::det_sign>(weights / ScalarExp<DType>(scaling_factor)) *
                      ScalarExp<DType>(scaling_factor);
          } else if (act_bit < 32) {
            workspace = F<mshadow_op::abs>(F<mshadow_op::tanh>(weights));

            DType max = 0;
            for (index_t i = 0; i < workspace.size(0); ++i) {
              if (workspace[i] > max) {
                max = workspace[i];
              }
            }

            weights = scalar(DType(2.0)) *
                      F<mshadow_op::quantize>(
                              F<mshadow_op::tanh>(weights) / scalar(DType(2.0) * max) + scalar(DType(0.5)),
                              scalar(DType(act_bit)))
                      - scalar(DType(1.0));
          }
        };
    }
  }
}
#endif //MXNET_Q_HELPER_H
