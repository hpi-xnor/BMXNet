/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include <vector>

namespace mshadow {
    template<typename Dtype>
    inline void QConvolutionForward(const Tensor<gpu, 4, Dtype> &data,
                                    const Tensor<gpu, 3, Dtype> &wmat,
                                    const Tensor<gpu, 4, Dtype> &out,
                                    const mxnet::op::QConvolutionParam &param) {
      LOG(FATAL) << "binary cuda convolution not supported yet";
    }
}

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(QConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

