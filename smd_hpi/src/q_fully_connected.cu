/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cu
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
#include "./q_fully_connected-inl.h"
#include <mshadow/tensor.h>
#include "./xnor_kernels.h"

namespace mshadow {
namespace cuda {

/*
 *  m: batch size
 *  n: input_dim e.g. 1024
 *  k: hidden_num e.g. 1000
 */
inline void _BinaryInferenceFullyConnectedForward(int m, int n, int k,
                                                 const Tensor<gpu, 2, float> &data,
                                                 Tensor<gpu, 1, float> &workspace,
                                                 mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
                                                 Tensor<gpu, 2, float> &out) {
                                      
  CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * m);         
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);

  //set memory
  float *fA = data.dptr_; 
  // float *fB = wmat.dptr_;
  float *fC = out.dptr_;  
  xnor_cuda::BINARY_WORD* binary_row = (xnor_cuda::BINARY_WORD*) workspace.dptr_;
      
  //concatinates matrix (m x n) -> (m x n/32)
  int threads_per_block = xnor_cuda::get_next_block_dim(m*n/xnor_cuda::BITS_PER_BINARY_WORD);
  dim3 conc_block(threads_per_block, 1, 1);
  dim3 conc_grid(m*n/(threads_per_block*xnor_cuda::BITS_PER_BINARY_WORD)+1,1);
  xnor_cuda::concatenate_rows_kernel<<<conc_grid, conc_block, 0, stream>>>(fA, binary_row, m*n/xnor_cuda::BITS_PER_BINARY_WORD);

  //get block size  
  threads_per_block = xnor_cuda::get_next_block_dim(m, n/xnor_cuda::BITS_PER_BINARY_WORD, k);
  // Shared memory used to store Asub and Bsub respectively
  int memsize = threads_per_block*threads_per_block*sizeof(xnor_cuda::BINARY_WORD)*2;
  //perform xnor gemm
  dim3 block(threads_per_block, threads_per_block);
  dim3 grid(k/threads_per_block + 1, m/threads_per_block + 1);
  xnor_cuda::xnor_gemm<<<grid, block, memsize, stream>>>(binary_row, (xnor_cuda::BINARY_WORD*)wmat_binarized, fC, 
                                                m, n/xnor_cuda::BITS_PER_BINARY_WORD, k, 
                                                threads_per_block);   
  cudaDeviceSynchronize();      
}
}  // namespace cuda

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
                                     Tensor<gpu, 2, float> &out) {
    cuda::_BinaryInferenceFullyConnectedForward(m, n, k, data, workspace, wmat_binarized, out);
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     const Tensor<gpu, 2, float> &wmat,
                                     Tensor<gpu, 2, float> &out) {
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    float *fB = wmat.dptr_; //note that the weight matrix 'wmat' here should be transposed (w.T) one.
    mxnet::op::xnor_cpu::BINARY_WORD *wmat_binarized;
    cudaMalloc(&wmat_binarized, n*k/xnor_cuda::BITS_PER_BINARY_WORD*sizeof(mxnet::op::xnor_cpu::BINARY_WORD));

    //concatinates matrix (n x k) -> (n/32 x k)
    int threads_per_block = xnor_cuda::get_next_block_dim(k);;
    int blocks_per_grid = k / threads_per_block + 1;
    dim3 conc_block(threads_per_block,1,1);
    dim3 conc_grid(blocks_per_grid,1);
    xnor_cuda::concatenate_cols_kernel<<<conc_grid, conc_block, 0, stream>>>(fB, (xnor_cuda::BINARY_WORD*)wmat_binarized, n, k);
    cudaDeviceSynchronize();
    cuda::_BinaryInferenceFullyConnectedForward(m, n, k, data, workspace, wmat_binarized, out);
    cudaFree(wmat_binarized);
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     const Tensor<gpu, 2, DType> &wmat,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }
} // namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(QFullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QFullyConnectedOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
