"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx

eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False

def get_symbol(num_classes, **kwargs):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=32)
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=32)
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu2 = mx.symbol.Activation(data=bn2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")

    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(5, 5), pad=(2, 2), num_filter=64)
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu3 = mx.symbol.Activation(data=bn3, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu3, kernel=(3, 3), stride=(2, 2), pool_type="max")    

    # stage 4
    conv4 = mx.symbol.Convolution(
        data=pool3, kernel=(5, 5), pad=(2, 2), num_filter=96)
    bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu4 = mx.symbol.Activation(data=bn4, act_type="relu")
    pool4 = mx.symbol.Pooling(data=relu4, kernel=(3, 3), pool_type="max")    
    
    # stage 5
    flatten = mx.symbol.Flatten(data=pool4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)

    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax
