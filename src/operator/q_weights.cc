/*!
 * Copyright (c) 2016 by Contributors
 * \file q_weights.cc
 * \brief Quantize Weights operator
 * \author HPI-DeepLearning
*/
#include "./q_weights-inl.h"

namespace mxnet {
    namespace op {
        template<>
        Operator *CreateOp<cpu>(QWeightsParam param, int dtype) {
            Operator *op = NULL;
            CHECK(param.act_bit==1 || param.act_bit==2 || param.act_bit==4 || param.act_bit==8 || param.act_bit==16 || param.act_bit==32);
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new QWeightsOp<cpu, DType>(param);
            });
            return op;
        }

// DO_BIND_DISPATCH comes from operator_common.h
        Operator *QWeightsProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                    std::vector<int> *in_type) const {
            std::vector<TShape> out_shape, aux_shape;
            std::vector<int> out_type, aux_type;
            CHECK(InferType(in_type, &out_type, &aux_type));
            CHECK(InferShape(in_shape, &out_shape, &aux_shape));
            DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
        }

        DMLC_REGISTER_PARAMETER(QWeightsParam);

        MXNET_REGISTER_OP_PROPERTY(QWeights, QWeightsProp)
        .describe(R"(Quantize weights function.

The following quantization methods of weights are supported:

- `1 bit`: using deteministic sign() function to generate binary activation
- `2-32 bit`: using quantization function

)")
        .add_arguments(QActivationParam::__FIELDS__());

    }  // namespace op
}  // namespace mxnet

