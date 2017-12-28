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

import mxnet as mx

BN_mom=0.9
BN_eps=2e-5

BITW = -1 # set in get_symbol
BITA = -1 # set in get_symbol

cudnn_off = False

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, 
    			stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def QConv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, 
    			eps=BN_eps, momentum=BN_mom)
    act = mx.sym.QActivation(data=bn, act_bit=BITA, backward_only=True)
    conv = mx.sym.QConvolution(data=act, num_filter=num_filter, kernel=kernel, num_group=num_group, 
    		stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix), act_bit=BITW, cudnn_off=cudnn_off)      
	relu = mx.symbol.LeakyReLU(data=conv, act_type="leaky")
    return relu

def get_symbol(num_classes, bits_w=1, bits_a=1, **kwargs):
    global BITW, BITA
    BITW = bits_w
    BITA = bits_a

    data = mx.symbol.Variable(name="data") # 224
    conv_1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1") # 224/112
    conv_2_dw = QConv(conv_1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112/112
    conv_2 = QConv(conv_2_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112/112
    conv_3_dw = QConv(conv_2, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 112/56
    conv_3 = QConv(conv_3_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56/56
    conv_4_dw = QConv(conv_3, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56/56
    conv_4 = QConv(conv_4_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56/56
    conv_5_dw = QConv(conv_4, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 56/28
    conv_5 = QConv(conv_5_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28/28
    conv_6_dw = QConv(conv_5, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28/28
    conv_6 = QConv(conv_6_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28/28
    conv_7_dw = QConv(conv_6, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 28/14
    conv_7 = QConv(conv_7_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14/14

    conv_8_dw = QConv(conv_7, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14/14
    conv_8 = QConv(conv_8_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14/14
    conv_9_dw = QConv(conv_8, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14/14
    conv_9 = QConv(conv_9_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14/14
    conv_10_dw = QConv(conv_9, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14/14
    conv_10 = QConv(conv_10_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14/14
    conv_11_dw = QConv(conv_10, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14/14
    conv_11 = QConv(conv_11_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14/14
    conv_12_dw = QConv(conv_11, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14/14
    conv_12 = QConv(conv_12_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14/14

    conv_13_dw = QConv(conv_12, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 14/7
    conv_13 = QConv(conv_13_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7/7
    conv_14_dw = QConv(conv_13, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7/7
    conv_14 = QConv(conv_14_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7

    pool = mx.sym.Pooling(data=conv_14, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool")
    flatten = mx.sym.Flatten(data=pool, name="flatten")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
