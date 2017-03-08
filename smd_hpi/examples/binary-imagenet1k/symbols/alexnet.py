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
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu2 = mx.symbol.Activation(data=bn2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")

    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu3 = mx.symbol.Activation(data=bn3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu4 = mx.symbol.Activation(data=bn4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    bn5 = mx.sym.BatchNorm(data=conv5, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu5 = mx.symbol.Activation(data=bn5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    bn6 = mx.sym.BatchNorm(data=fc1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu6 = mx.symbol.Activation(data=bn6, act_type="relu")

    # stage 5
    fc2 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096)
    bn7 = mx.sym.BatchNorm(data=fc2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu7 = mx.symbol.Activation(data=bn7, act_type="relu")
    
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=relu7, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
