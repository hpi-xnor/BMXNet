/*!
 * Copyright (c) 2016 by Contributors
 * \file q_convolution.cu
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution_v1-inl.h"
#include <mshadow/tensor.h>

namespace mshadow {

	inline void QConvolutionV1Forward(int m, int n, int k,
									mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
		CHECK(false) << "cuda with pre-binarized weights not implemented";
	}

	inline void QConvolutionV1Forward(int m, int n, int k,
									const Tensor<gpu, 2, float> &wmat,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
		//!deprecated! will be removed later
		//cuda::QConvolutionForward(wmat, in_col, temp_dst);
	}

	template<typename DType>
	inline void QConvolutionV1Forward(int m, int n, int k,
									const Tensor<gpu, 2, DType> &wmat,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}

	template<typename DType>
	inline void QConvolutionV1Forward(int m, int n, int k,
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
Operator* CreateOp<gpu>(QConvolutionV1Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionV1Op<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

