/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_layer.h
 * \brief representation of binary
 * \author HPI-DeepLearning
*/

#ifndef MXNET_BINARYLAYER_H
#define MXNET_BINARYLAYER_H

#include <mshadow/tensor.h>
#include <mshadow/base.h>

typedef uint32_t BINARY_WORD;
#define BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT)

namespace mxnet {
namespace op {
    using mshadow::Tensor;
    using mshadow::cpu;

    class BinaryLayer {
    public:
        BinaryLayer(int input_channels, int input_width, int input_height, int num_filters, int kernel_width, int kernel_height, int padding_x, int padding_y,
                    int m, int n, int k);
        ~BinaryLayer();

        void set_input_as_col(const mshadow::Tensor<cpu, 2, float> &input);
        void set_weights(const mshadow::Tensor<cpu, 2, float> &wmat);
        void get_output(const mshadow::Tensor<cpu, 2, float> &out);

        std::string weights_as_string();

        BINARY_WORD *binary_input = nullptr;
        BINARY_WORD *binary_weights = nullptr;
        float *output = nullptr;
        float *alpha = nullptr;
        float *beta = nullptr;

        int input_channels;
        int input_width;
        int input_height;
        int num_filters;
        int kernel_width;
        int kernel_height;
        int padding_x;
        int padding_y;
        int stride = 1;
        int output_width;
        int output_height;
        int m;
        int n;
        int k;
        static void float_to_binary(const mshadow::Tensor<cpu, 2, float> &input, BINARY_WORD *output);
        static void binary_to_float(BINARY_WORD *input, const mshadow::Tensor<cpu, 2, float> &out);

    private:
        void calculate_alpha(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume);
        void calculate_beta(float *output_plane, mshadow::Tensor<cpu, 3, float> input_volume);
    };

  }}

#endif //MXNET_BINARYLAYER_H
