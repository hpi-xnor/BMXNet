/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
#include "./q_fully_connected-inl.h"
#include "./xnor_cpu.h"


namespace mshadow {

    inline void QFullyConnectedForward(const Tensor<cpu, 2, float> &data,
                                    const Tensor<cpu, 2, float> &wmat,
                                    const Tensor<cpu, 2, float> &out,
                                    const mxnet::op::QFullyConnectedParam &param) {
    	CHECK(false) << "not implemented";
    }

    template<typename DType>
    inline void QFullyConnectedForward(const Tensor<cpu, 2, DType> &data,
                                    const Tensor<cpu, 2, DType> &wmat,
                                    const Tensor<cpu, 2, DType> &out,
                                    const mxnet::op::QFullyConnectedParam &param) {
      CHECK(false) << "only float supported";
    }
}

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(QFullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QFullyConnectedOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QFullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(QFullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(QFullyConnected, QFullyConnectedProp)
.describe(R"(Apply matrix multiplication to input then add a bias.
It maps the input of shape `(batch_size, input_dim)` to the shape of
`(batch_size, num_hidden)`. Learnable parameters include the weights
of the linear transform and an optional bias vector.)")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(QFullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
