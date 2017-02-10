/*!
 * Copyright (c) 2016 by Contributors
 * \file q_weights.cu
 * \brief Quantize Weights operator
 * \author HPI-DeepLearning
*/
#include "./q_weights-inl.h"

namespace mxnet {
    namespace op {
        template<>
        Operator *CreateOp<gpu>(QWeightsParam param, int dtype) {
            Operator *op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new QWeightsOp<gpu, DType>(param);
            });
            return op;
        }
    }  // namespace op
}  // namespace mxnet

