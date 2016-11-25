import mxnet as mx
import numpy as np


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
		return mx.sym.round(x * n) / n

	def qua_w(x):
		"""
		quantization function for weights
			x: input tensor
		"""
		#32 bit
		if nbit_w == 32:
			return x
		# 1 bit
		if nbit_w == 1:   # BWN TODO: implement custom operators
				E = mx.sym.mean(mx.sym.abs(x))
				return mx.sym.sign(x / E) * E
		# otherwise
		x = tf.tanh(x)
		x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
		return 2 * quantize(x, bitW) - 1

	def qua_a(x):
		if bitA == 32:
			return x
		return quantize(x, bitA)

	global GRAD_DEFINED
	if not GRAD_DEFINED:
		@tf.RegisterGradient("FGGrad")
		def grad_fg(op, x):
			rank = x.get_shape().ndims
			assert rank is not None
			maxx = tf.reduce_max(tf.abs(x), list(range(1,rank)), keep_dims=True)
			x = x / maxx
			n = float(2**bitG-1)
			x = x * 0.5 + 0.5 + tf.random_uniform(
					tf.shape(x), minval=-0.5/n, maxval=0.5/n)
			x = tf.clip_by_value(x, 0.0, 1.0)
			x = quantize(x, bitG) - 0.5
			return x * maxx * 2
	GRAD_DEFINED = True

	def qua_g(x):
		if bitG == 32:
			return x
		with G.gradient_override_map({"Identity": "FGGrad"}):
			return tf.identity(x)
	return fw, fa, fg