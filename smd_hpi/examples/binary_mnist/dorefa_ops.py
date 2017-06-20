import mxnet as mx
import numpy as np
from math_ops import *

def get_dorefa(nbit_w, nbit_a, nbit_g):
	""" 
	implements a dorefa style quantization functions fw, fa, fg, for weights,
	activations and gradients respectively
	param:
		nbit_w: bit of weights
		nbit_a: bit of activation
		nbit_g: bit of gradient
	"""    

	def quantize(x, k):
		"""
		Implements k-bit quatization function
		x: input tensor
		k: k-bit quatization
		"""
		n = float(2**k-1)
		x_q = mx.sym.Custom(data=x*n, op_type='around') / n		
		return x_q

	def binary_sign(x):
		"""
		- clip input tensor to [0, 1]
		- round it to {0, 1}
		- convert to {1, -1}
		"""
		x_1_0 = mx.sym.Custom(data=x, op_type='clip_by_0_1')
		x_round = mx.sym.Custom(data=x_1_0, op_type='around')
		binary_w = x_round * 2 - 1
		return binary_w

	def qua_w(x):
		"""
		quantization function for weights
			x: input tensor
		"""
		#32 bit
		if nbit_w == 32:
			return x
		# 1 bit
		if nbit_w == 1:   
				#with scaling factor E
				E = mx.sym.Custom(data=mx.sym.abs(x), op_type='pro_channel_reduce_mean')	
				binary_w = binary_sign(x/E)*E
				#binary_w = binary_sign(x)
				#binary_w = mx.sym.Custom(data=binary_w, op_type='debug')
				return binary_w
		# otherwise
		x = mx.sym.Activation(data=x, act_type="tanh")  	
		x = x / mx.sym.Custom(data=mx.sym.abs(x), op_type='amax') * 0.5 + 0.5
		return 2 * quantize(x, nbit_w) - 1

	def qua_a(x):
		if nbit_a == 32:
			return x
		if nbit_a == 1:
			return binary_sign(x)
		return quantize(x, nbit_a)

	def qua_g(x):
		#if nbit_g == 32:
		return x
	return qua_w, qua_a, qua_g

	