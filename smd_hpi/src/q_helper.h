/*!
 * Copyright (c) 2017 by Contributors
 * \file q_helper.h
 * \brief Quantization helper function
 * \author HPI-DeepLearning
*/

#ifndef MXNET_Q_HELPER_H
#define MXNET_Q_HELPER_H

#include "../../src/operator/mshadow_op.h"
#include <mshadow/tensor.h>
#include <mshadow/expression.h>

namespace mxnet {
namespace op {    
namespace q_helper {    

using mshadow::expr::F;
using mshadow::expr::ScalarExp;
using mshadow::expr::scalar;    

//wrapper function for CUDA kernel        
extern "C" float launch_max_reduce(float*, int);
extern "C" double launch_max_reduce_d(double*, int);
extern "C" float launch_mean_reduce(float*, int);
extern "C" double launch_mean_reduce_d(double*, int);


// CPU amean
template<int dim, typename DType>
inline DType amean(const mshadow::Tensor<cpu, dim, DType> &tensor) {
  if(tensor.shape_.Size() == 0) return DType(0);

  DType mean = 0;
  #pragma omp parallel for
  for (index_t i = 0; i < tensor.shape_.Size(); ++i) {
    mean += tensor.dptr_[i];
  }
  mean = mean / static_cast<DType>(tensor.shape_.Size());
  return mean;
}

// CPU amax
template<int dim, typename DType>
inline DType amax(const mshadow::Tensor<cpu, dim, DType> &tensor) {
  DType max = 0;
  for (index_t i = 0; i < tensor.shape_.Size(); ++i) {
    if (tensor.dptr_[i] > max) {
      max = tensor.dptr_[i];
    }
  }
  return max;
}


#if MXNET_USE_CUDA
//wrapper function for CUDA kernel        
//extern "C" float launch_max_reduce(DType*, int);
// @todo naive implementation |==> this needs to be implemented nicely and with gpu support (see nvidia pdf on reduction with cuda)
// GPU (includes copy to CPU)
template<int dim, typename DType>
inline DType amax(const mshadow::Tensor<gpu, dim, DType> &tensor) {
  mshadow::Tensor<cpu, dim, DType> tensor_cpu = mshadow::NewTensor<cpu>(tensor.shape_, DType(0.0));
  mshadow::Copy(tensor_cpu, tensor, tensor.stream_);

  DType max = 0;
  for (index_t i = 0; i < tensor_cpu.shape_.Size(); ++i) {
    if (tensor_cpu.dptr_[i] > max) {
      max = tensor_cpu.dptr_[i];
    }
  }
  mshadow::FreeSpace(&tensor_cpu);
  std::cout << "INFO: amax gpu -> cpu" << std::endl;

  return max;
}

// Launches CUDA max_reduce kernel.
// float
template<int dim>
inline float amax(const mshadow::Tensor<gpu, dim, float> &tensor) {
  int tensor_size = tensor.shape_.Size();
  float * input = tensor.dptr_;
  float max = launch_max_reduce(input, tensor_size);

/* stuff for verify gpu and cpu results
  mshadow::Tensor<cpu, dim, float> tensor_cpu = mshadow::NewTensor<cpu>(tensor.shape_, 0.0f);
  mshadow::Copy(tensor_cpu, tensor, tensor.stream_);

  float cpumax = .0f;
  for (index_t i = 0; i < tensor_cpu.shape_.Size(); ++i) {
    cpumax = cpumax >tensor_cpu.dptr_[i] ? cpumax : tensor_cpu.dptr_[i];
  }  
  mshadow::FreeSpace(&tensor_cpu);
  printf("cpu max: %f \n gpu max float: %f \n", cpumax, max);
*/

  return max;
}
// Launches CUDA max_reduce kernel.
// double
template<int dim>
inline double amax(const mshadow::Tensor<gpu, dim, double> &tensor) {
  int tensor_size = tensor.shape_.Size();
  double * input = tensor.dptr_;
  double max = launch_max_reduce_d(input, tensor_size);
  return max;
}


// GPU (includes copy to CPU)
template<int dim, typename DType>
inline DType amean(const mshadow::Tensor<gpu, dim, DType> &tensor) {
  if(tensor.shape_.Size() == 0) return DType(0);

  mshadow::Tensor<cpu, dim, DType> tensor_cpu = mshadow::NewTensor<cpu>(tensor.shape_, DType(0.0));
  mshadow::Copy(tensor_cpu, tensor, tensor.stream_);

  DType mean = 0;
  #pragma omp parallel for
  for (index_t i = 0; i < tensor_cpu.shape_.Size(); ++i) {
    mean += tensor_cpu.dptr_[i];
  }
  mean = mean/(DType)tensor_cpu.shape_.Size();
  mshadow::FreeSpace(&tensor_cpu);
  std::cout << "INFO: amean gpu -> cpu" << std::endl;
  return mean;
}

/*
 * GPU 
 * Double
 * calculates the global mean of input array using mean reduce kernel
 */
template<int dim>
inline double amean(const mshadow::Tensor<gpu, dim, double> &tensor) {
  if(tensor.shape_.Size() == 0) return .0;
  int tensor_size = tensor.shape_.Size();
  double * input = tensor.dptr_;
  double mean = launch_mean_reduce_d(input, tensor_size);
  return mean;
}

// Float
template<int dim>
inline float amean(const mshadow::Tensor<gpu, dim, float> &tensor) {
  if(tensor.shape_.Size() == 0) return .0f;
  int tensor_size = tensor.shape_.Size();
  float * input = tensor.dptr_;
  float mean = launch_mean_reduce(input, tensor_size);
  
  /* stuff for verify gpu and cpu results
  //testing code
  mshadow::Tensor<cpu, dim, float> tensor_cpu = mshadow::NewTensor<cpu>(tensor.shape_, 0.0f);
  mshadow::Copy(tensor_cpu, tensor, tensor.stream_);
  float cpumean = 0.0f;
  for (index_t i = 0; i < tensor_size; ++i) {
    cpumean += (float)tensor_cpu.dptr_[i];
  }
  cpumean = cpumean/(float)tensor_size;
  mshadow::FreeSpace(&tensor_cpu);



  printf("cpu mean: %f \n gpu mean float: %f \n", cpumean, mean);
  */
  return mean;
}
#endif


/*
 * Quantization for dataflow.
 * bit width > 1:
 *  Forward: w_i = 2 * quantize_k-bit(tanh(x_i)/2*max(|tanh(x_i)|) + 1/2) - 1
 */
template<int dim, typename xpu, typename DType>
inline void quantize_weights(mshadow::Tensor<xpu, dim, DType> &dataflow, unsigned int bit_width) {
  if (bit_width == 1) {
    real_t scaling_factor = 1;
    dataflow = F<mshadow_op::det_sign>(dataflow / ScalarExp<DType>(scaling_factor)) *
              ScalarExp<DType>(scaling_factor);
  } else if (bit_width < 32) {
    mshadow::Tensor<xpu, dim, DType> workspace = mshadow::NewTensor<xpu>(dataflow.shape_, DType(1.0), true, dataflow.stream_);
    workspace = F<mshadow_op::abs>(F<mshadow_op::tanh>(dataflow));

    DType max = amax(workspace);            

    dataflow = scalar(DType(2.0)) *
              F<mshadow_op::quantize>(
                      F<mshadow_op::tanh>(dataflow) / scalar(DType(2.0) * max) + scalar(DType(0.5)),
                      scalar(DType(bit_width))) 
                      - scalar(DType(1.0));            

    mshadow::FreeSpace(&workspace);
  }
}

/*
 * Quantization for activations (inputs).
 */
template<int dim, typename xpu, typename DType>
inline void quantize_activations(mshadow::Tensor<xpu, dim, DType> &dataflow, unsigned int bit_width) {
  if (bit_width == 1) {
    dataflow = F<mshadow_op::det_sign>(dataflow);
  } else {         
    dataflow = F<mshadow_op::quantize>(
                                        F<mshadow_op::maximum>(
                                          F<mshadow_op::minimum>(
                                            dataflow, 
                                            scalar(DType(1))), 
                                          scalar(DType(0))), //clip to [0, 1]
                                        scalar(DType(bit_width))
                                      );
  }
}


/*
 * Calculates the mean of absolute values for the dataflow.
 * E_x = amean( abs(x) )
 */
template<int dim, typename xpu, typename DType>
inline DType get_scaling_scalar(const mshadow::Tensor<xpu, dim, DType> &dataflow) {
  mshadow::Tensor<xpu, dim, DType> workspace = mshadow::NewTensor<xpu>(dataflow.shape_, DType(1.0), true, dataflow.stream_);
  workspace = F<mshadow_op::abs>(dataflow);
  DType mean = amean(workspace);                    
  mshadow::FreeSpace(&workspace);
  return mean;
}

/*
 * return tensor * scalar
 */
template<int dim, typename xpu, typename DType>
inline void tensor_mul_scalar(mshadow::Tensor<xpu, dim, DType>& dataflow, DType scal) {
  dataflow = dataflow * mshadow::expr::scalar(DType(scal));
}


} // namespace q_helper

}//op
}//mxnet
#endif //MXNET_Q_HELPER_H
