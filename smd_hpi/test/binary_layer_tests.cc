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
#include "../src/binary_layer.h"

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


TEST(binary_layer, flt_to_bin) {
  mshadow::Shape<2> shape;
  shape.shape_[0] = 5;
  shape.shape_[1] = 2;

  mshadow::Tensor<mshadow::cpu, 2, float> tensor_cpu = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
  mshadow::Tensor<mshadow::cpu, 2, float> tensor_output = mshadow::NewTensor<mshadow::cpu>(shape, 1.0f);
  BINARY_WORD* binary = (BINARY_WORD *) calloc(shape.shape_[0] * shape.shape_[1] / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));

  // init with values
  for (int y = 0; y < tensor_cpu.shape_[0]; y++) {
    for (int x = 0; x < tensor_cpu.shape_[1]; x++) {
      int index = y * tensor_cpu.shape_[1] + x;
      (x + y) % 3 == 0 ? tensor_cpu.dptr_[index] = +1.0f : tensor_cpu.dptr_[index] = -1.0f ;
    }
  }

  mxnet::op::BinaryLayer::float_to_binary(tensor_cpu, binary);

  mxnet::op::BinaryLayer::binary_to_float(binary, tensor_output);

  for (int y = 0; y < tensor_cpu.shape_[0]; y++) {
    for (int x = 0; x < tensor_cpu.shape_[1]; x++) {
      int index = y * tensor_cpu.shape_[1] + x;
      EXPECT_EQ(tensor_cpu.dptr_[index], tensor_output.dptr_[index]);
    }
  }

  mshadow::FreeSpace(&tensor_output);
  mshadow::FreeSpace(&tensor_cpu);
  free(binary);
}}}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
