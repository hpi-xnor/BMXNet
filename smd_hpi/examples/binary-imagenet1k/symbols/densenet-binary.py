"""
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu
Referenced https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
Original author bamos
Referenced https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
Original author andreasveit
Referenced https://github.com/Nicatio/Densenet/blob/master/mxnet/symbol_densenet.py
Original author Nicatio
Implemented the following paper:     DenseNet-BC
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"
Coded by Lin Xiong Mar-1, 2017
"""
import mxnet as mx
import math
import logging
logging.basicConfig(level=logging.DEBUG)

BITW = -1 # set in get_symbol
BITA = -1 # set in get_symbol


def BasicBlock(data, growth_rate, stride, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return BaiscBlock Unit symbol for building DenseBlock
    Parameters
    ----------
    data : str
        Input data
    growth_rate : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    # import pdb
    # pdb.set_trace()

    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.QActivation(data=bn1, backward_only=True, name=name + '_relu1', act_bit=BITA)
        conv1 = mx.sym.QConvolution(data=act1, num_filter=int(growth_rate * 4), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=1, workspace=workspace, name=name + '_conv1', act_bit=BITW)
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.QActivation(data=bn2, backward_only=True, name=name + '_relu2', act_bit=BITA)
        conv2 = mx.sym.QConvolution(data=act2, num_filter=int(growth_rate), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=1, workspace=workspace, name=name + '_conv2', act_bit=BITW)
        if drop_out > 0:
            conv2 = mx.symbol.Dropout(data=conv2, p=drop_out, name=name + '_dp2')
        # return mx.symbol.Concat(data, conv2, name=name + '_concat0')
        return conv2
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.QActivation(data=bn1, backward_only=True, name=name + '_relu1', act_bit=BITA)
        conv1 = mx.sym.QConvolution(data=act1, num_filter=int(growth_rate), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=1, workspace=workspace, name=name + '_conv1', act_bit=BITW)
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        # return mx.symbol.Concat(data, conv1, name=name + '_concat0')
        return conv1


def DenseBlock(units_num, data, growth_rate, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return DenseBlock Unit symbol for building DenseNet
    Parameters
    ----------
    units_num : int
        the number of BasicBlock in each DenseBlock
    data : str
        Input data
    growth_rate : int
        Number of output channels
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    # import pdb
    # pdb.set_trace()

    for i in range(units_num):
        Block = BasicBlock(data, growth_rate=growth_rate, stride=(1, 1), name=name + '_unit%d' % (i + 1),
                           bottle_neck=bottle_neck, drop_out=drop_out,
                           bn_mom=bn_mom, workspace=workspace)
        data = mx.symbol.Concat(data, Block, name=name + '_concat%d' % (i + 1))
    return data


def TransitionBlock(num_stage, data, num_filter, stride, name, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return TransitionBlock Unit symbol for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    data : str
        Input data
    num : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    name : str
        Base name of the operators
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.QActivation(data=bn1, backward_only=True, name=name + '_relu1', act_bit=BITA)
    conv1 = mx.sym.QConvolution(data=act1, num_filter=num_filter,
                               kernel=(1, 1), stride=stride, pad=(0, 0), no_bias=1,
                               workspace=workspace, name=name + '_conv1', act_bit=BITW)
    if drop_out > 0:
        conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
    return mx.symbol.Pooling(conv1, global_pool=False, kernel=(2, 2), stride=(2, 2), pool_type='avg',
                             name=name + '_pool%d' % (num_stage + 1))


def DenseNet(units, num_stage, growth_rate, num_class, data_type, reduction=0.5, drop_out=0., bottle_neck=True,
             bn_mom=0.9, workspace=512):
    """Return DenseNet symbol of imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    growth_rate : int
        Number of output channels
    num_class : int
        Ouput size of symbol
    data_type : str
        the type of dataset
    reduction : float
        Compression ratio. Default = 0.5
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    init_channels = 2 * growth_rate
    n_channels = init_channels
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=growth_rate * 2, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    elif data_type == 'vggface':
        body = mx.sym.Convolution(data=data, num_filter=growth_rate * 2, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    elif data_type == 'msface':
        body = mx.sym.Convolution(data=data, num_filter=growth_rate * 2, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=1, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage - 1):
        body = DenseBlock(units[i], body, growth_rate=growth_rate, name='DBstage%d' % (i + 1), bottle_neck=bottle_neck,
                          drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        n_channels += units[i] * growth_rate
        n_channels = int(math.floor(n_channels * reduction))
        body = TransitionBlock(i, body, n_channels, stride=(1, 1), name='TBstage%d' % (i + 1), drop_out=drop_out,
                               bn_mom=bn_mom, workspace=workspace)
    body = DenseBlock(units[num_stage - 1], body, growth_rate=growth_rate, name='DBstage%d' % (num_stage),
                      bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    # mx.visualization.print_summary(fc1, shape={'data':(512,3,224,224)})
    # digraph = mx.visualization.plot_network(ret, save_format='jpg')
    # digraph.render()
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, bn_mom=0.9, drop_out=0.0, reduction=0.5,
               data_type="imagenet", growth_rate=32, bits_w=1, bits_a=1, use_bottle_neck=False, **kwargs):
    global BITW, BITA
    BITW = bits_w
    BITA = bits_a

    logging.info("Created binary densenet with bit_w={} and bit_a={} and bottleneck={}.".format(BITW, BITA, use_bottle_neck))

    if data_type == "imagenet":
        if num_layers == 21:
            units = [2, 2, 2, 2]
        elif num_layers == 45:
            units = [2, 4, 8, 6]
        elif num_layers == 85:
            units = [4, 8, 16, 12]
        elif num_layers == 121:
            units = [6, 12, 24, 16]
        elif num_layers == 169:
            units = [6, 12, 32, 32]
        elif num_layers == 201:
            units = [6, 12, 48, 32]
        elif num_layers == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(num_layers))
        DenseNet(units=units, num_stage=4, growth_rate=64 if num_layers == 161 else growth_rate, num_class=num_classes, 
                            data_type="imagenet", reduction=reduction, drop_out=drop_out, bottle_neck=use_bottle_neck,
                            bn_mom=bn_mom, workspace=conv_workspace)
        return 
    elif data_type == "vggface":
        if num_layers   == 121:
            units = [6, 12, 24, 16]
        elif num_layers == 169:
            units = [6, 12, 32, 32]
        elif num_layers == 201:
            units = [6, 12, 48, 32]
        elif num_layers == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(num_layers))
        return DenseNet(units=units, num_stage=4, growth_rate=64 if num_layers == 161 else growth_rate, num_class=num_classes, 
                            data_type="vggface", reduction=reduction, drop_out=drop_out, bottle_neck=use_bottle_neck,
                            bn_mom=bn_mom, workspace=conv_workspace)
    elif data_type == "msface":
        if num_layers   == 121:
            units = [6, 12, 24, 16]
        elif num_layers == 169:
            units = [6, 12, 32, 32]
        elif num_layers == 201:
            units = [6, 12, 48, 32]
        elif num_layers == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(num_layers))
        return DenseNet(units=units, num_stage=4, growth_rate=64 if num_layers == 161 else growth_rate, num_class=num_classes, 
                            data_type="msface", reduction=reduction, drop_out=drop_out, bottle_neck=use_bottle_neck,
                            bn_mom=bn_mom, workspace=conv_workspace)
    else:
        raise ValueError("do not support {} yet".format(data_type))
