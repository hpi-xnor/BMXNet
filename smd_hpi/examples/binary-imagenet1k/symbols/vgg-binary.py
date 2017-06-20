"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import mxnet as mx
BIT = 1

def get_symbol(num_classes, **kwargs):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    bn2_1 = mx.sym.BatchNorm(data=pool1, fix_gamma=False, eps=2e-5)
    act2_1 = mx.sym.QActivation(data=bn2_1, act_bit=BIT, backward_only=True)    
    conv2_1 = mx.symbol.QConvolution(
        data=act2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    bn2_2 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=2e-5)
    pool2 = mx.symbol.Pooling(
        data=bn2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    act3_1 = mx.sym.QActivation(data=pool2, act_bit=BIT, backward_only=True)            
    conv3_1 = mx.symbol.QConvolution(
        data=act3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    
    bn3_2 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=2e-5)
    act3_2 = mx.sym.QActivation(data=bn3_2, act_bit=BIT, backward_only=True)            
    conv3_2 = mx.symbol.QConvolution(
        data=act3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    
    bn3 = mx.sym.BatchNorm(data=conv3_2, fix_gamma=False, eps=2e-5)
    pool3 = mx.symbol.Pooling(
        data=bn3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    
    # group 4
    act4_1 = mx.sym.QActivation(data=pool3, act_bit=BIT, backward_only=True)
    conv4_1 = mx.symbol.QConvolution(
        data=act4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")

    bn4_2 = mx.sym.BatchNorm(data=conv4_1, fix_gamma=False, eps=2e-5)
    act4_2 = mx.sym.QActivation(data=bn4_2, act_bit=BIT, backward_only=True)            
    conv4_2 = mx.symbol.QConvolution(
        data=act4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")

    bn4 = mx.sym.BatchNorm(data=conv4_2, fix_gamma=False, eps=2e-5)
    pool4 = mx.symbol.Pooling(
        data=bn4, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")

    # group 5
    act5_1 = mx.sym.QActivation(data=pool4, act_bit=BIT, backward_only=True)
    conv5_1 = mx.symbol.QConvolution(
        data=act5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")

    bn5_2 = mx.sym.BatchNorm(data=conv5_1, fix_gamma=False, eps=2e-5)
    act5_2 = mx.sym.QActivation(data=bn5_2, act_bit=BIT, backward_only=True)  
    conv5_2 = mx.symbol.QConvolution(
        data=act5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")

    bn5 = mx.sym.BatchNorm(data=conv5_2, fix_gamma=False, eps=2e-5)
    pool5 = mx.symbol.Pooling(
        data=bn5, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
