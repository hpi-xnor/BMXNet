# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding:utf-8 -*-
__author__ = 'zhangshuai'
modified_date = '16/7/5'
__modify__ = 'anchengwu'
modified_date = '17/2/22'

'''
Inception v4 , suittable for image with around 299 x 299

Reference:
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke
    arXiv.1602.07261
'''
import mxnet as mx
import numpy as np

BITW = -1 # set in get_symbol
BITA = -1 # set in get_symbol

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_Qconv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=Qconv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def QConv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.QActivation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix), act_bit=BITA, backward_only=True) 
    Qconv = mx.sym.QConvolution(data=act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_Qconv2d' %(name, suffix), 
                              act_bit=BITA, weight_bit=BITW, cudnn_off=cudnn_off)
    bn2 = mx.sym.BatchNorm(data=Qconv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True, eps=2e-5, momentum=0.9)
    return bn2

def QInception_stem(data, name= None):
    c = Conv(data, 32, kernel=(3, 3), stride=(2, 2), name='%s_Qconv1_3*3' %name)
    c = QConv(c, 32, kernel=(3, 3), name='%s_Qconv2_3*3' %name)
    c = QConv(c, 64, kernel=(3, 3), pad=(1, 1), name='%s_Qconv3_3*3' %name)

    p1 = mx.sym.Pooling(c, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)
    c2 = QConv(c, 96, kernel=(3, 3), stride=(2, 2), name='%s_Qconv4_3*3' %name)
    concat = mx.sym.Concat(*[p1, c2], name='%s_concat_1' %name)

    c1 = QConv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_Qconv5_1*1' %name)
    c1 = QConv(c1, 96, kernel=(3, 3), name='%s_Qconv6_3*3' %name)

    c2 = QConv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_Qconv7_1*1' %name)
    c2 = QConv(c2, 64, kernel=(7, 1), pad=(3, 0), name='%s_Qconv8_7*1' %name)
    c2 = QConv(c2, 64, kernel=(1, 7), pad=(0, 3), name='%s_Qconv9_1*7' %name)
    c2 = QConv(c2, 96, kernel=(3, 3), pad=(0, 0), name='%s_Qconv10_3*3' %name)

    concat = mx.sym.Concat(*[c1, c2], name='%s_concat_2' %name)

    c1 = QConv(concat, 192, kernel=(3, 3), stride=(2, 2), name='%s_Qconv11_3*3' %name)
    p1 = mx.sym.Pooling(concat, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_2' %name)

    concat = mx.sym.Concat(*[c1, p1], name='%s_concat_3' %name)

    return concat


def QInceptionA(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = QConv(p1, 96, kernel=(1, 1), pad=(0, 0), name='%s_Qconv1_1*1' %name)

    c2 = QConv(input, 96, kernel=(1, 1), pad=(0, 0), name='%s_Qconv2_1*1' %name)

    c3 = QConv(input, 64, kernel=(1, 1), pad=(0, 0), name='%s_Qconv3_1*1' %name)
    c3 = QConv(c3, 96, kernel=(3, 3), pad=(1, 1), name='%s_Qconv4_3*3' %name)

    c4 = QConv(input, 64, kernel=(1, 1), pad=(0, 0), name='%s_Qconv5_1*1' % name)
    c4 = QConv(c4, 96, kernel=(3, 3), pad=(1, 1), name='%s_Qconv6_3*3' % name)
    c4 = QConv(c4, 96, kernel=(3, 3), pad=(1, 1), name='%s_Qconv7_3*3' %name)

    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)

    return concat


def QReductionA(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)

    c2 = QConv(input, 384, kernel=(3, 3), stride=(2, 2), name='%s_Qconv1_3*3' %name)

    c3 = QConv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_Qconv2_1*1' %name)
    c3 = QConv(c3, 224, kernel=(3, 3), pad=(1, 1), name='%s_Qconv3_3*3' %name)
    c3 = QConv(c3, 256, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name='%s_Qconv4_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat

def QInceptionB(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = QConv(p1, 128, kernel=(1, 1), pad=(0, 0), name='%s_Qconv1_1*1' %name)

    c2 = QConv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_Qconv2_1*1' %name)

    c3 = QConv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_Qconv3_1*1' %name)
    c3 = QConv(c3, 224, kernel=(1, 7), pad=(0, 3), name='%s_Qconv4_1*7' %name)
    #paper wrong
    c3 = QConv(c3, 256, kernel=(7, 1), pad=(3, 0), name='%s_Qconv5_1*7' %name)

    c4 = QConv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_Qconv6_1*1' %name)
    c4 = QConv(c4, 192, kernel=(1, 7), pad=(0, 3), name='%s_Qconv7_1*7' %name)
    c4 = QConv(c4, 224, kernel=(7, 1), pad=(3, 0), name='%s_Qconv8_7*1' %name)
    c4 = QConv(c4, 224, kernel=(1, 7), pad=(0, 3), name='%s_Qconv9_1*7' %name)
    c4 = QConv(c4, 256, kernel=(7, 1), pad=(3, 0), name='%s_Qconv10_7*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)

    return concat

def QReductionB(input,name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)

    c2 = QConv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_Qconv1_1*1' %name)
    c2 = QConv(c2, 192, kernel=(3, 3), stride=(2, 2), name='%s_Qconv2_3*3' %name)

    c3 = QConv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_Qconv3_1*1' %name)
    c3 = QConv(c3, 256, kernel=(1, 7), pad=(0, 3), name='%s_Qconv4_1*7' %name)
    c3 = QConv(c3, 320, kernel=(7, 1), pad=(3, 0), name='%s_Qconv5_7*1' %name)
    c3 = QConv(c3, 320, kernel=(3, 3), stride=(2, 2), name='%s_Qconv6_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat


def QInceptionC(input, name=None):
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = QConv(p1, 256, kernel=(1, 1), pad=(0, 0), name='%s_Qconv1_1*1' %name)

    c2 = QConv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_Qconv2_1*1' %name)

    c3 = QConv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_Qconv3_1*1' %name)
    c3_1 = QConv(c3, 256, kernel=(1, 3), pad=(0, 1), name='%s_Qconv4_3*1' %name)
    c3_2 = QConv(c3, 256, kernel=(3, 1), pad=(1, 0), name='%s_Qconv5_1*3' %name)

    c4 = QConv(input, 384, kernel=(1, 1), pad=(0, 0), name='%s_Qconv6_1*1' %name)
    c4 = QConv(c4, 448, kernel=(1, 3), pad=(0, 1), name='%s_Qconv7_1*3' %name)
    c4 = QConv(c4, 512, kernel=(3, 1), pad=(1, 0), name='%s_Qconv8_3*1' %name)
    c4_1 = QConv(c4, 256, kernel=(3, 1), pad=(1, 0), name='%s_Qconv9_1*3' %name)
    c4_2 = QConv(c4, 256, kernel=(1, 3), pad=(0, 1), name='%s_Qconv10_3*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3_1, c3_2, c4_1, c4_2], name='%s_concat' %name)

    return concat


def get_symbol(num_classes=1000, dtype='float32', bits_w=1, bits_a=1, **kwargs):
    global BITW, BITA
    BITW = bits_w
    BITA = bits_a

    data = mx.sym.Variable(name="data")
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    x = QInception_stem(data, name='in_stem')

    #4 * QInceptionA
    # x = QInceptionA(x, name='in1A')
    # x = QInceptionA(x, name='in2A')
    # x = QInceptionA(x, name='in3A')
    # x = QInceptionA(x, name='in4A')

    for i in range(4):
        x = QInceptionA(x, name='in%dA' %(i+1))

    #Reduction A
    x = QReductionA(x, name='re1A')

    #7 * QInceptionB
    # x = QInceptionB(x, name='in1B')
    # x = QInceptionB(x, name='in2B')
    # x = QInceptionB(x, name='in3B')
    # x = QInceptionB(x, name='in4B')
    # x = QInceptionB(x, name='in5B')
    # x = QInceptionB(x, name='in6B')
    # x = QInceptionB(x, name='in7B')

    for i in range(7):
        x = QInceptionB(x, name='in%dB' %(i+1))

    #QReductionB
    x = QReductionB(x, name='re1B')

    #3 * QInceptionC
    # x = QInceptionC(x, name='in1C')
    # x = QInceptionC(x, name='in2C')
    # x = QInceptionC(x, name='in3C')

    for i in range(3):
        x = QInceptionC(x, name='in%dC' %(i+1))

    #Average Pooling
    x = mx.sym.Pooling(x, kernel=(8, 8), pad=(1, 1), pool_type='avg', name='global_avgpool')

    #Dropout
    x = mx.sym.Dropout(x, p=0.2)

    flatten = mx.sym.Flatten(x, name='flatten')
    fc1 = mx.sym.FullyConnected(flatten, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(fc1, name='softmax')

    return softmax
