import os
import pdb

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np

#=========== math reduce_mean ============#
class ChannelReduceMean(mx.operator.CustomOp):
	"""
	can only process array up to 4 dimontions
	"""
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0].asnumpy()				
		#pdb.set_trace()		
		y = x
		n = x.ndim

		#create mean for each channel
		if n > 2:
			y = np.mean(y, axis=2, keepdims=True)
		if n == 4:
			y = np.mean(y, axis=3, keepdims=True)
		
		#do this if we want to get a global scalar mean across all channels			
		#if n > 1:
		#	y = np.mean(y, axis=1, keepdims=True)
		
		y_nd = mx.nd.array(y)
		y_o = y_nd.broadcast_to(x.shape)
		#pdb.set_trace()		
		self.assign(out_data[0], req[0], y_o)
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])	

@mx.operator.register("pro_channel_reduce_mean")
class ChannelReduceMeanProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(ChannelReduceMeanProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return ChannelReduceMean()
#========= end math mean ==========#

#=========== math tanh ============#
class Tanh(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0].asnumpy()
		y = np.tanh(x)	
		#pdb.set_trace()
		#print y.asnumpy()
		self.assign(out_data[0], req[0], mx.nd.array(y))
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("tanh")
class TanhProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(TanhProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return Tanh()
#=========== end tanh ============#

#=========== math max ============#
class Amax(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0].asnumpy()
		y = np.amax(x)
		self.assign(out_data[0], req[0], y)
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("amax")
class AmaxProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(AmaxProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return Amax()
#=========== end tanh ============#

#=========== math around ============#
class Around(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0]
		y = mx.nd.round(x)
		#pdb.set_trace()
		#print y.asnumpy()
		self.assign(out_data[0], req[0], y)
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("around")
class AroundProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(AroundProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return Around()
#========= end around ==========#

#=========== math clip ============#
class Clip(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0]
		y = mx.nd.clip(x, 0, 1)
		#pdb.set_trace()
		self.assign(out_data[0], req[0], y)
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("clip_by_0_1")
class ClipProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(ClipProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return Clip()
#========= end math clip ==========#   

#=========== debug data ============#
class debug(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0].asnumpy()
		#pdb.set_trace()
		print x
		#print y.asnumpy()
		self.assign(out_data[0], req[0], in_data[0])
	
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("debug")
class debugProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(debugProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def create_operator(self, ctx, shapes, dtypes):
		return debug()
#=========== end debug data ============#