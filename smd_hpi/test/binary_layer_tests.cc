/*!
 * Copyright (c) 2017 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
//#include <cstdio>
#include <gtest/gtest.h>

#include <mxnet/base.h>
#include <dmlc/logging.h>

namespace mxnet {
namespace op {


std::string print_tensor(mshadow::Tensor<mshadow::cpu, 2, float> tensor_cpu) {
  std::ostringstream ss;
  for (int y = 0; y < tensor_cpu.shape_[0]; y++) {
    for (int x = 0; x < tensor_cpu.shape_[1]; x++) {
        int index = y * tensor_cpu.shape_[1] + x;
        ss << tensor_cpu.dptr_[index] << " | ";
      }
      ss << "\n";
  }
  ss << "-----\n";
  return ss.str();
}

template<int dim>
void init_tensor(mshadow::Tensor<mshadow::cpu, dim, float> &tensor) {
  int total_elements= 1;
  for (int i = 0; i < dim; i++) {
    total_elements *= tensor.size(i);
  }
  for (int i = 0; i < total_elements; i++) {
    i % 3 == 0 ? tensor.dptr_[i] = +1.0f : tensor.dptr_[i] = -1.0f;
  }
}

template<int dim>
void expect_eq(mshadow::Tensor<mshadow::cpu, dim, float> &a, mshadow::Tensor<mshadow::cpu, dim, float> &b) {
  int total_elements= 1;
  for (int i = 0; i < dim; i++) {
    total_elements *= a.size(i);
    ASSERT_EQ(a.size(i), b.size(i));
  }
  for (int i = 0; i < total_elements; i++) {
    EXPECT_EQ(a.dptr_[i], b.dptr_[i]);
  }
}


//TEST(binary_layer, float_to_binary_transposed) {
//  mshadow::Shape<2> shape;
//  shape.shape_[0] = 32;
//  shape.shape_[1] = 2;
//
//  mshadow::Tensor<mshadow::cpu, 2, float> tensor_cpu = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
//  mshadow::Tensor<mshadow::cpu, 2, float> tensor_output = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
//  BINARY_WORD* binary = (BINARY_WORD *) calloc(shape.shape_[0] * shape.shape_[1] / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
//
//  // init with values
//  init_tensor(tensor_cpu);
//
//  mxnet::op::BinaryLayer::float_to_binary_transposed(tensor_cpu, binary);
//  mxnet::op::BinaryLayer::binary_transposed_to_float(binary, tensor_output);
//
//  expect_eq(tensor_cpu, tensor_output);
//
//  mshadow::FreeSpace(&tensor_output);
//  mshadow::FreeSpace(&tensor_cpu);
//  free(binary);
//}
//
//TEST(binary_layer, float_to_binary) {
//  mshadow::Shape<2> shape;
//  shape.shape_[0] = 3;
//  shape.shape_[1] = 2;
//
//  mshadow::Tensor<mshadow::cpu, 2, float> tensor_cpu = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
//  mshadow::Tensor<mshadow::cpu, 2, float> tensor_output = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
//  BINARY_WORD* binary = (BINARY_WORD *) calloc(shape.shape_[0] * shape.shape_[1] / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
//
//  // init with values
//  init_tensor(tensor_cpu);
//
////  LOG(INFO) << "\n" << print_tensor(tensor_cpu);
//  mxnet::op::BinaryLayer::float_to_binary(tensor_cpu, binary);
//
//  mxnet::op::BinaryLayer::binary_to_float(binary, tensor_output);
////  LOG(INFO) << "\n" << print_tensor(tensor_output);
//
//  expect_eq(tensor_cpu, tensor_output);
//
//  mshadow::FreeSpace(&tensor_output);
//  mshadow::FreeSpace(&tensor_cpu);
//  free(binary);
//}}}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
