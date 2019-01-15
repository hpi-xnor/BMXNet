/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution_v1-inl.h"

using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

namespace mshadow {
    using namespace mxnet::op::xnor_cpu;

    inline void _QConvolutionV1Forward(int m, int n, int k,
									 BINARY_WORD* binary_weights_row,
									 Tensor<cpu, 1, float> &workspace,
									 const Tensor<cpu, 2, float> &in_col,
									 Tensor<cpu, 2, float> &temp_dst) {
  			CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * k);
  			BINARY_WORD* binary_col = (BINARY_WORD*) workspace.dptr_;

  			get_binary_col_unrolled(in_col.dptr_, binary_col, n, k);
  			
  			temp_dst = 0;
      	
    		xnor_gemm(m, k, n/BITS_PER_BINARY_WORD,
                    binary_weights_row, n/BITS_PER_BINARY_WORD,
                    binary_col, k,
                    temp_dst.dptr_, k);
    }


    inline void QConvolutionV1Forward(int m, int n, int k,
									BINARY_WORD* wmat_binarized,
									Tensor<cpu, 1, float> &workspace,
                                    const Tensor<cpu, 2, float> &in_col,
                                    Tensor<cpu, 2, float> &temp_dst) {

		    _QConvolutionV1Forward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
    }

	inline void QConvolutionV1Forward(int m, int n, int k,
									const Tensor<cpu, 2, float> &wmat,
									Tensor<cpu, 1, float> &workspace,
									const Tensor<cpu, 2, float> &in_col,
									Tensor<cpu, 2, float> &temp_dst) {
      	BINARY_WORD binary_row[m * n/BITS_PER_BINARY_WORD];
      	get_binary_row(wmat.dptr_, &binary_row[0], m*n);
		    _QConvolutionV1Forward(m, n, k, binary_row, workspace, in_col, temp_dst);
	}


    template<typename DType>
    inline void QConvolutionV1Forward(int m, int n, int k,
                                    const Tensor<cpu, 2, DType> &wmat,
									Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }

    template<typename DType>
    inline void QConvolutionV1Forward(int m, int n, int k,
									BINARY_WORD* wmat_binarized,
									Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }
}

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(QConvolutionV1Param);

template<>
Operator* CreateOp<cpu>(QConvolutionV1Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionV1Op<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QConvolutionV1Prop::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(QConvolution_v1, QConvolutionV1Prop)
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(QConvolutionV1Param::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
