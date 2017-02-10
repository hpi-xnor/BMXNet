/*!
 * Copyright (c) 2016 by Contributors
 * \file q_activation-inl.h
 * \brief Quantized Activation operator
 * \author HPI-DeepLearning
*/
#include "./q_activation-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(QActivationParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QActivationOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet

