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
Operator *CreateOp<cpu>(QActivationParam param, int dtype) {
  Operator *op = NULL;
  CHECK(param.act_bit==1 || param.act_bit==2 || param.act_bit==4 || param.act_bit==8 || param.act_bit==16 || param.act_bit==32);
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QActivationOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QActivationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(QActivationParam);

MXNET_REGISTER_OP_PROPERTY(QActivation, QActivationProp)
.describe(R"(Quantized activation function.

The following quantized/binarized activation are supported (operations are applied elementwisely to each
scalar of the input tensor):

- `1 bit`: using deteministic sign() function to generate binary activation
- `2-32 bit`: using quantization function 

)")
.add_argument("data", "NDArray-or-Symbol", "Input array to q_activation function.") 
.add_arguments(QActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

