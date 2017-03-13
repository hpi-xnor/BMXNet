/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/

#include "./q_convolution-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <memory>
#include "./xnor_cpu.h"
# include <chrono>
using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

namespace mshadow {
    using namespace mxnet::op::xnor_cpu;

    inline void QConvolutionForward(const Tensor<cpu, 4, float> &data,
                                    const Tensor<cpu, 2, float> &wmat,
                                    const Tensor<cpu, 2, float> &in_col,
                                    const Tensor<cpu, 2, float> &temp_dst,
                                    const Tensor<cpu, 4, float> &out,
                                    const mxnet::op::QConvolutionParam &param) {

			///*
		int m = wmat.size(0);
		int n = wmat.size(1);
		int k = in_col.size(1);
		int batch_size = data.size(0);
		int input_width = data.size(2);
		int input_height = data.size(3);
		int input_depth = data.size(1);
		int output_width = (input_width - param.kernel[0] + 2 * 0/*padding*/) / 1/*stride*/ + 1;
		int output_height = (input_height - param.kernel[1] + 2 * 0/*padding*/) / 1/*stride*/ + 1;
		BINARY_WORD* binary_row = (BINARY_WORD*) malloc(m * n/BITS_PER_BINARY_WORD * sizeof(BINARY_WORD));
		BINARY_WORD* binary_col = (BINARY_WORD*) malloc(n * k/BITS_PER_BINARY_WORD * sizeof(BINARY_WORD));
		float *alpha_plane = nullptr;
		float *A_planes = nullptr;
		float *K_planes = nullptr;

		if (param.scaling_factor) {
			CHECK_EQ(param.stride[0], 1) << "binary convolution currently only supported with stride==1";
			CHECK_EQ(param.stride[1], 1) << "binary convolution currently only supported with stride==1";
			CHECK_EQ(param.pad[0], 0) << "cant create beta scaling factor with padded input yet";
			CHECK_EQ(param.pad[1], 0) << "cant create beta scaling factor with padded input yet";
			alpha_plane = (float *) malloc(param.num_filter * sizeof(float));
			A_planes = (float *) malloc(input_width * input_height * batch_size * sizeof(float));
			K_planes = (float *) malloc(output_width * output_height * batch_size * sizeof(float));
			// alpha
			get_alpha_plane(alpha_plane, wmat.dptr_, param.num_filter, param.kernel[0], param.kernel[1], input_depth);
			// beta
			get_A_planes(A_planes, data.dptr_, input_depth, input_width, input_height, batch_size);
			get_K_planes(K_planes, A_planes, input_width, input_height, param.kernel[0], param.kernel[1], batch_size);
		}

		get_binary_row(wmat.dptr_, binary_row, m*n);
		get_binary_col(in_col.dptr_, binary_col, n, k);

	    //#pragma omp parallel for
	    for (int i = 0; i < temp_dst.shape_.Size(); ++i) {
	      temp_dst.dptr_[i] = 0;
	    }
		//auto start = std::chrono::high_resolution_clock::now();

		///*
		xnor_gemm(m, k, n/BITS_PER_BINARY_WORD,
				binary_row, n/BITS_PER_BINARY_WORD,
				binary_col, k,
				temp_dst.dptr_, k);
		//*/

		/*
		//test using baseline gemm kernel
		baseline_gemm(m, k, n,
							wmat.dptr_, n,
					in_col.dptr_, k,
					temp_dst.dptr_, k);
		*/
		//auto finish = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<double> elapsed = finish - start;
		//std::cout << "xnor Elapsed time: " << elapsed.count() << " s\n";

		if (param.scaling_factor) {
			// apply alpha and beta scaling factor (filter-wise)
			// std::cout << "random values from alpha: " << alpha_plane[3] << " " << alpha_plane[30] << " and from K plane: " << K_planes[3] << " " << K_planes[130] << " " << K_planes[1000] << std::endl;
			for (int i = 0; i < param.num_filter; i++) {
				pointwise_mul_scalar(temp_dst[i].dptr_, alpha_plane[i], batch_size * output_height * output_width);
				pointwise_mul_mm(temp_dst[i].dptr_, K_planes, output_width * output_height * batch_size);
			}
			if (alpha_plane) free(alpha_plane);
			if (A_planes) free(A_planes);
			if (K_planes) free(K_planes);
		}

		free(binary_row);
		free(binary_col);
    }

    template<typename DType>
    inline void QConvolutionForward(const Tensor<cpu, 4, DType> &data,
                                    const Tensor<cpu, 2, DType> &wmat,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    const Tensor<cpu, 2, DType> &temp_dst,
                                    const Tensor<cpu, 4, DType> &out,
                                    const mxnet::op::QConvolutionParam &param) {
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
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(QConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
