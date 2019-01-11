/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"

using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

namespace mshadow {
    using namespace mxnet::op::xnor_cpu;

    inline void _QConvolutionForward(int m, int n, int k,
									 BINARY_WORD* binary_weights_row,
									 Tensor<cpu, 1, float> &workspace,
									 const Tensor<cpu, 2, float> &in_col,
									 Tensor<cpu, 2, float> &temp_dst) {
  			CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * k);
  			BINARY_WORD* binary_col = (BINARY_WORD*) workspace.dptr_;

  			get_binary_col_unrolled(in_col.dptr_, binary_col, k, n);
  			
  			temp_dst = 0;
      	
    		xnor_gemm(m, n, k/BITS_PER_BINARY_WORD,
                    binary_weights_row, k/BITS_PER_BINARY_WORD,
                    binary_col, n,
                    temp_dst.dptr_, n);
    }


    inline void QConvolutionForward(int m, int n, int k,
									BINARY_WORD* wmat_binarized,
									Tensor<cpu, 1, float> &workspace,
                                    const Tensor<cpu, 2, float> &in_col,
                                    Tensor<cpu, 2, float> &temp_dst) {

		    _QConvolutionForward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
    }

	inline void QConvolutionForward(int m, int n, int k,
									const Tensor<cpu, 2, float> &wmat,
									Tensor<cpu, 1, float> &workspace,
									const Tensor<cpu, 2, float> &in_col,
									Tensor<cpu, 2, float> &temp_dst) {
      	BINARY_WORD binary_row[m * k/BITS_PER_BINARY_WORD];
      	get_binary_row(wmat.dptr_, &binary_row[0], m*k);
		    _QConvolutionForward(m, n, k, binary_row, workspace, in_col, temp_dst);
	}


    template<typename DType>
    inline void QConvolutionForward(int m, int n, int k,
                                    const Tensor<cpu, 2, DType> &wmat,
									Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }

    template<typename DType>
    inline void QConvolutionForward(int m, int n, int k,
									BINARY_WORD* wmat_binarized,
									Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }
}

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(QConvolutionParam);

template<>
Operator* CreateOp<cpu>(QConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(QConvolution, QConvolutionProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the Q_ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(QConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
