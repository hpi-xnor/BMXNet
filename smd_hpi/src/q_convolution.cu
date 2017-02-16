/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include "../../src/operator/convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "../../src/operator/cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

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
#if MXNET_USE_CUDNN == 1
  if (param.dilate.Size() == 1 && !param.cudnn_off) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      ConvolutionParam conv_param;
      conv_param.kernel = param.kernel;
      conv_param.stride = param.stride;
      conv_param.dilate = param.dilate;
      conv_param.pad = param.pad;
      conv_param.num_filter = param.num_filter;
      conv_param.num_group = param.num_group;
      conv_param.workspace = param.workspace;
      conv_param.no_bias = param.no_bias;
      conv_param.cudnn_tune = param.cudnn_tune;
      conv_param.cudnn_off = param.cudnn_off;
      conv_param.layout = param.layout;

      op = new CuDNNConvolutionOp<DType>(conv_param, *in_shape, *out_shape, ctx);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new QConvolutionOp<gpu, DType>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet

