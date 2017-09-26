/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include <mshadow/tensor.h>
#if MXNET_USE_CUDNN == 1
#include "./q_cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mshadow {

	inline void QConvolutionForward(int m, int n, int k,
									mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
		CHECK(false) << "cuda with pre-binarized weights not implemented";
	}

	inline void QConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, float> &wmat,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
    CHECK(false) << "!deprecated! will be removed later";
		//cuda::QConvolutionForward(wmat, in_col, temp_dst);
	}

	template<typename DType>
	inline void QConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, DType> &wmat,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}

	template<typename DType>
	inline void QConvolutionForward(int m, int n, int k,
									mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}
} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(QConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
	Operator *op = NULL;
	// If 1D convolution, use MXNet implementation
	if (param.kernel.ndim() == 1) {
		MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new QConvolutionOp<gpu, DType>(param);
		})
		return op;
	}
#if MXNET_USE_CUDNN == 1
	// The NVIDIA Pascal architecture was the first to include 16-bit ALUs.
  // Thus, when the framework is compiled with MSHADOW_USE_PASCAL == 1, we
  // perform the convolution calculation in 16-bit when the tensor type is
  // also 16-bit.  For NVIDIA architectures earlier than Pascal (so Maxwell
  // and Kepler), the computation precision is always at least 32-bits.
#if MSHADOW_USE_PASCAL == 1
  // true fp16
  int desired_forward_compute_type = dtype;
  int desired_backward_compute_type = dtype;
#else
  // pseudo fp16
  int desired_forward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
  int desired_backward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
#endif  // MSHADOW_USE_PASCAL == 1

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      op = new QConvolutionOp<gpu, DType>(param);
    } else {
      int forward_compute_type = desired_forward_compute_type;
      int backward_compute_type = desired_backward_compute_type;
      bool convolutionIsSupported = QCuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);

      // If cuDNN can't handle this case with fp16 backprop kernels, try fp32 backprop.
      if (!convolutionIsSupported && backward_compute_type == mshadow::kFloat16) {
        backward_compute_type = mshadow::kFloat32;
        convolutionIsSupported = QCuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);
      }

      // If cuDNN can't handle this case with fp16 forward kernels, try fp32
      if (!convolutionIsSupported && forward_compute_type == mshadow::kFloat16) {
        forward_compute_type = mshadow::kFloat32;
        convolutionIsSupported = QCuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);
      }
      if (!convolutionIsSupported) {
        LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
        op = new QConvolutionOp<gpu, DType>(param);
      } else {
        if ((forward_compute_type != desired_forward_compute_type) ||
            (backward_compute_type != desired_backward_compute_type))
          LOG(WARNING) << "True fp16 convolution by cudnn not supported in this configuration.  " <<
                       "Falling back to pseudo fp16.";
        op = new QCuDNNConvolutionOp<DType>(param,
                                         forward_compute_type,
                                         backward_compute_type,
                                         *in_shape, *out_shape, ctx);
      }
    }
  })
#else
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
			op = new QConvolutionOp<gpu, DType>(param);
	})
#endif  // MXNET_USE_CUDNN
	return op;

}

}  // namespace op
}  // namespace mxnet

