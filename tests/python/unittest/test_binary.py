import mxnet as mx
import numpy as np
import mxnet.ndarray as F
import pytest

def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)

def test_binary_inference_consistency_fc_gpu_cpu():
    '''
    compares the outputs of binary_inference_fc from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        input_shapes = [(100,2048), (10,1024), (1,512), (1,64), (1,32)]# [(batch size, num of input channels)...]
        hidden_nums = [1000, 512, 100, 13, 2]        
        bits_binary_word = 32

        for input_shape in input_shapes:
            for hidden_num in hidden_nums:                        
                weight_shape = (hidden_num, (int)(input_shape[1]/bits_binary_word))
                
                # create input tensor using gpu
                input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
                # create a copy on cpu
                input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))
                
                # create weights
                weight_np = np.random.randint(0, 2, size=weight_shape)    
                weight_g = mx.nd.array(weight_np, dtype='int32', ctx=mx.gpu(gpu_num))
                weight_c = mx.nd.array(weight_np, dtype='int32', ctx=mx.cpu(0))

                # binary inferece forward
                result_g = mx.ndarray.QFullyConnected(data=input_g, weight=weight_g, num_hidden=hidden_num, binarized_weights_only = True)
                result_c = mx.ndarray.QFullyConnected(data=input_c, weight=weight_c, num_hidden=hidden_num, binarized_weights_only = True)
                
                np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())        

def test_binary_inference_consistency_conv_gpu_cpu():
    '''
    compares the outputs of binary_inference_conv from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        bits_binary_word = 32
        filter_nums = [5, 32, 64, 128, 512]
        input_shapes = [(10,64,8,8), (10,256,8,8),(10,1024,8,8), (10,512,8,8),
                        (10,256,32,32),(10,32,10,10)]# [(batch size, num of input channels, h, w)...]
        conv_kernels = [(7,7), (3,3), (1,1)]

        for input_shape in input_shapes:
            for conv_kernel in conv_kernels:
                for filter_num in filter_nums:
                    weight_shape = (filter_num,
                                    (int)(input_shape[1]/bits_binary_word),  # num input channels / bits
                                    conv_kernel[0], 
                                    conv_kernel[1])
                    
                    # create input tensor using gpu
                    input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
                    # create a copy on cpu
                    input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))
                    
                    # create weights
                    weight_np = np.random.randint(0, 2, size=weight_shape)    
                    weight_g = mx.nd.array(weight_np, dtype='int32', ctx=mx.gpu(gpu_num))
                    weight_c = mx.nd.array(weight_np, dtype='int32', ctx=mx.cpu(0))

                    # binary inferece forward
                    result_g = mx.nd.QConvolution(data=input_g, weight=weight_g, 
                                                kernel=conv_kernel, num_filter=filter_num, binarized_weights_only = True)
                    result_c = mx.nd.QConvolution(data=input_c, weight=weight_c, 
                                                kernel=conv_kernel, num_filter=filter_num, binarized_weights_only = True)
                    
                    np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())        


def test_qfc_train_output_consistency_gpu_cpu():
    '''
    compares the outputs of binary_inference_fc from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        input_shapes = [(100,2048), (10,1024), (1,512), (1,64), (1,32)]# [(batch size, num of input channels)...]
        hidden_nums = [1000, 512, 100, 13, 2]        

        for input_shape in input_shapes:
            for hidden_num in hidden_nums:                        
                weight_shape = (hidden_num, input_shape[1])
                
                # create input tensor using gpu
                input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
                # create a copy on cpu
                input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))
                
                # create weights
                weight_np = np.random.uniform(-1, 1, size=weight_shape)    
                weight_g = mx.nd.array(weight_np, dtype='float32', ctx=mx.gpu(gpu_num))
                weight_c = mx.nd.array(weight_np, dtype='float32', ctx=mx.cpu(0))

                # binary inferece forward
                result_g = mx.ndarray.QFullyConnected(data=input_g, weight=weight_g, num_hidden=hidden_num)
                result_c = mx.ndarray.QFullyConnected(data=input_c, weight=weight_c, num_hidden=hidden_num)
                
                np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())    


def test_qconv_train_output_consistency_gpu_cpu():
    '''
    compares the outputs of binary_inference_conv from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        filter_nums = [64,8,128]
        input_shapes = [(10,64,8,8), (10,512,8,8),(10,1024,8,8),(10,32,10,10)]# [(batch size, num of input channels, h, w)...]
        conv_kernels = [(7,7), (3,3), (1,1)]

        for input_shape in input_shapes:
            for conv_kernel in conv_kernels:
                for filter_num in filter_nums:
                    weight_shape = (filter_num,
                                    input_shape[1],  # num input channels / bits
                                    conv_kernel[0], 
                                    conv_kernel[1])
                    
                    # create input tensor using gpu
                    input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
                    # create a copy on cpu
                    input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))
                    
                    # create weights
                    weight_np = np.random.uniform(-1, 1, size=weight_shape)    
                    weight_g = mx.nd.array(weight_np, dtype='float32', ctx=mx.gpu(gpu_num))
                    weight_c = mx.nd.array(weight_np, dtype='float32', ctx=mx.cpu(0))

                    # binary inferece forward
                    result_g = mx.nd.QConvolution(data=input_g, weight=weight_g, 
                                                kernel=conv_kernel, num_filter=filter_num)
                    result_c = mx.nd.QConvolution(data=input_c, weight=weight_c, 
                                                kernel=conv_kernel, num_filter=filter_num)
                    
                    np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())        