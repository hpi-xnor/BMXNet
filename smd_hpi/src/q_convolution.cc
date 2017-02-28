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
#include "./binary_layer.h"
#include "./xnor_cpu.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../../src/operator/mkl/mkl_memory-inl.h"
#include "../../src/operator/mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "../../src/operator/nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK
# include <chrono>
using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

namespace mshadow {
	inline float get_alpha(float* weight, int width, int height, int depth) {
		float accum = 0.0f;
		for (int z = 0; z < depth; ++z) {
			for (int x = 0; x < width; ++x) {
				for (int y = 0; y < height; ++y) {
					accum += weight[z * (width * height) + x * height + y]; //@todo: abs?
				}
			}
		}
		return accum / (float) (width * height * depth);
	}

	inline void get_alpha_plane(float* alpha_plane_out, float* weights,
															int num_weights,
															int kernel_width, int kernel_height,
															int input_depth) {
		for (int i = 0; i < num_weights; i++) {
			alpha_plane_out[i] = get_alpha(&weights[i * kernel_height * kernel_width * input_depth], kernel_height, kernel_width, input_depth);
		}
	}

	inline void get_A_planes(float* A_planes_out, float* input,
													 int input_depth, int input_width, int input_height,
													 int batch_size) {
		for (int i = 0; i < batch_size; i++) {
			for (int x = 0; x < input_width; ++x) {
				for (int y = 0; y < input_height; ++y) {
					float accum = 0.0f;
					for (int z = 0; z < input_depth; ++z) {
						accum += input[i * (input_depth * input_width * input_height) +
													 z * (input_width * input_height) +
													 x * input_height +
													 y]; //@todo: abs?
					}
					A_planes_out[i * input_width * input_height +
										  x * input_height +
											y] = accum / (float) input_depth;
				}
			}
		}
	}

	inline void get_K_planes(float* K_planes_out, float* A_planes,
													 int input_width, int input_height,
													 int kernel_width, int kernel_height,
													 int batch_size) {
		int K_width = (input_width - kernel_width + 2 * 0/*padding*/) / 1/*stride*/ + 1;
		int K_height = (input_height - kernel_height + 2 * 0/*padding*/) / 1/*stride*/ + 1;

		//@todo: super naive conv
		for (int i = 0; i < batch_size; i ++) {
			// for every batch
			for (int kx = 0; kx < K_width; kx++) {
				for (int ky = 0; ky < K_height; ky++) {
					// for every kx, ky in our output plane
					int accum = 0;
					// we do collect the sum of all values covered by the kernel
					for (int ix = kx; ix < kx + kernel_width; ix++) {
						for (int iy = ky; iy < ky + kernel_height; iy++) {
							accum += A_planes[i * input_width * input_height +
																ix * input_height +
																iy];
						}
					}
					// and multiply them with 1/(w * h)
					K_planes_out[i * K_width * K_height +
											 kx * K_height +
											 ky] = accum / ((float) kernel_height * kernel_width);
				}
			}
		}
	}

	inline void pointwise_mul_mm(float *output, const float *input, int size){
		for (int i = 0; i < size; i++) {
			output[i] *= input[i];
		}
	}

	inline void pointwise_mul_scalar(float *output, const float scalar, int size){
		for (int i = 0; i < size; i++) {
			output[i] *= scalar;
		}
	}

	inline BINARY_WORD concatenate(float* array)
	{
		BINARY_WORD rvalue=0;
		BINARY_WORD sign;

		#pragma omp parallel for
		for (int i = 0; i < BITS_PER_BINARY_WORD; i++)
		{
			sign = (array[i]>=0);
			rvalue = rvalue | (sign<< (i));
		}

		return rvalue;
	}
	inline void get_binary_row(float* row, BINARY_WORD * b_row, int size){

		for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
			float * array = new float[BITS_PER_BINARY_WORD];

			#pragma omp parallel for
			for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
				array[j] = row[i+j];
			}

			b_row[i/BITS_PER_BINARY_WORD] = concatenate(array);
			delete[] array;
		}
	}

	inline void get_binary_col(float* col, BINARY_WORD * b_col, int n, int k){

		for(int x=0; x < k; ++x){
			for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
				float * array = new float[BITS_PER_BINARY_WORD];

				#pragma omp parallel for
				for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
					array[b] = col[(y*BITS_PER_BINARY_WORD+b)*k + x];
				}

				b_col[y*k + x]=concatenate(array);
				delete[] array;
			}
		}
	}

	inline void xnor_gemm(int M, int K, int N,
							BINARY_WORD *A, int lda,
							BINARY_WORD *B, int ldb,
							float *C, int ldc){
	    int i,n,k;

		#pragma omp parallel for
	    for(i = 0; i < M; ++i){
			#pragma omp parallel for
	        for(n = 0; n < N; ++n){
	        	BINARY_WORD A_PART = A[i*lda+n];
				#pragma omp parallel for
	            for(k = 0; k < K; ++k){
	            	C[i*ldc+k] += (float)__builtin_popcount(~(A_PART ^ B[n*ldb+k]));
	            }
	        }
	    }
	}


	inline void baseline_gemm(int M, int K, int N,
							float *A, int lda,
							float *B, int ldb,
							float *C, int ldc){
	    int i,n,k;
	    for(i = 0; i < M; ++i){
	        for(n = 0; n < N; ++n){
	        	float A_PART = A[i*lda+n];
	            for(k = 0; k < K; ++k){
	            	C[i*ldc+k] += A_PART * B[n*ldb+k];
	            }
	        }
	    }
	}

    inline void QConvolutionForward(const Tensor<cpu, 4, float> &data,
                                    const Tensor<cpu, 2, float> &wmat,
                                    const Tensor<cpu, 2, float> &in_col,
                                    const Tensor<cpu, 2, float> &temp_dst,
                                    const Tensor<cpu, 4, float> &out,
                                    const mxnet::op::QConvolutionParam &param) {

      CHECK_EQ(param.stride[0], 1) << "binary convolution currently only supported with stride==1";
      CHECK_EQ(param.stride[1], 1) << "binary convolution currently only supported with stride==1";
			CHECK_EQ(param.pad[0], 0) << "cant create beta scaling factor with padded input yet";
			CHECK_EQ(param.pad[1], 0) << "cant create beta scaling factor with padded input yet";

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
			float* alpha_plane = (float *) malloc (param.num_filter * sizeof(float));
			float* A_planes = (float *) malloc (input_width * input_height * batch_size * sizeof(float));
			float* K_planes = (float *) malloc (output_width * output_height * batch_size * sizeof(float));

			get_binary_row(wmat.dptr_, binary_row, m*n);
			get_binary_col(in_col.dptr_, binary_col, n, k);

			// alpha
			get_alpha_plane(alpha_plane, wmat.dptr_, param.num_filter, param.kernel[0], param.kernel[1], input_depth);

			// beta
			get_A_planes(A_planes, data.dptr_, input_depth, input_width, input_height, batch_size);
			get_K_planes(K_planes, A_planes, input_width, input_height, param.kernel[0], param.kernel[1], batch_size);

			auto start = std::chrono::high_resolution_clock::now();
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
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			std::cout << "xnor Elapsed time: " << elapsed.count() << " s\n";

			free(binary_row);
			free(binary_col);
			free(alpha_plane);
			free(A_planes);
			free(K_planes);
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
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
      LOG(FATAL) << "QConvolution not supported with MKL";
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolutionOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConvolutionOp<cpu, double>(param);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConvolutionOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2 && (!param.no_bias)
      && param.num_group == 1 && (batch_size == 1 ||
      ((batch_size > 1) && (param.stride[0] == 1) &&
      (param.stride[1] == 1)))) {
      LOG(FATAL) << "QConvolution not supported with NNPACK";
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKConvolutionOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif
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
