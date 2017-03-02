import mxnet as mx
import logging
import numpy as np
import matplotlib.pyplot as plt
import pdb
from dorefa_ops import get_dorefa
from math_ops import *

logging.getLogger().setLevel(logging.DEBUG)

BITW = 1
BITA = 1
BITG = 6 # TODO: we don't have binarized gradient implementation yet.

# get quantized functions
f_w, f_a, f_g = get_dorefa(BITW, BITA, BITG)

def to4d(img):
	return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size):	
	train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
	val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)
	return train_iter, val_iter

def get_lenet():
	"""
	original lenet
	"""
	data = mx.symbol.Variable('data')
	# first conv layer
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=32)
	tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")	
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	# second conv layer
	conv2 = mx.sym.QConvolution(data=pool1, kernel=(5,5), num_filter=50, act_bit=BITW)	
	#conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)	
	#conv2 = mx.sym.Custom(data=conv2, op_type='debug')

	tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
	pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
	# first fullc layer
	flatten = mx.sym.Flatten(data=pool2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
	tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
	# second fullc
	fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
	# softmax loss
	lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
	return lenet

def nonlin(x):
	if BITA == 32:
		return mx.sym.Activation(data=x, act_type="tanh")    # still use tanh for 32bit cases
	return mx.sym.Custom(data=x, op_type='clip_by_0_1')

def activate(x):
	return f_a(nonlin(x))


def get_binary_lenet():
	data = mx.symbol.Variable('data')

	# first conv layer
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=32)
	bn1 = mx.sym.BatchNorm(data=conv1)
	tanh1 = mx.sym.Activation(data=bn1, act_type="tanh")

	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

	# second conv layer
	conv2 = mx.sym.QConvolution(data=pool1, kernel=(5,5), num_filter=100, act_bit=BITW, scaling_factor=False)

	#conv2 = mx.sym.Custom(data=conv2, op_type='debug')

	bn2 = mx.sym.BatchNorm(data=conv2)

	tanh2 = mx.sym.Activation(data=bn2, act_type="tanh")

	pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
	# first fullc layer
	flatten = mx.sym.Flatten(data=pool2)	
	fc1 = mx.symbol.QFullyConnected(data=flatten, num_hidden=500, act_bit=BITW)
	#fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
	
	#fc1 = mx.sym.Custom(data=fc1, op_type='debug')

	bn3 = mx.sym.BatchNorm(data=fc1)

	tanh3 = mx.sym.Activation(data=bn3, act_type="tanh")

	# second fullc
	fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
	# softmax loss
	lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

	print 'using quantized lenet with bitwidth %d (weights), %d (activations) and %d (gradients)' % (BITW, BITA, BITG)
	return lenet


def train(train_img, val_img, train_lbl, val_lbl, batch_size, epochs, gpu_id=0):
	lenet = get_lenet()
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	device = mx.cpu()
	if gpu_id >= 0:
		device = mx.gpu(gpu_id)
	model = mx.model.FeedForward(
		ctx = device,     # use GPU 0 for training, others are same as before
		symbol = lenet,   		  # network structure    
		num_epoch = epochs,     	  # number of data passes for training 
		learning_rate = 0.1)
	model.fit(
		X=train_iter,  			# training data
		eval_data=val_iter, 	# validation data
		batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
	) 
	return model

def val(model_prefix, epoch_num, train_img, val_img, train_lbl, val_lbl, batch_size):
	print('Preparing data for validation...')
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	print('Loading model...')
	model = mx.model.FeedForward.load(model_prefix, epoch_num)
	print('Evaluating...')
	print 'Validation accuracy: %f%%' % (model.score(val_iter)*100,)

def classify(val_img, model_prefix, epoch_num):
	model = mx.model.FeedForward.load(model_prefix, epoch_num)
	plt.imshow(val_img[0], cmap='Greys_r')
	plt.axis('off')
	plt.show()
	prob = model.predict(to4d(val_img[0:1]))[0]
	print 'Classified as %d with probability %f' % (prob.argmax(), max(prob))

def train_binary(train_img, val_img, train_lbl, val_lbl, batch_size, epochs, gpu_id=0):
	lenet = get_binary_lenet()
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	device = mx.cpu()
	if gpu_id >= 0:
		device = mx.gpu(gpu_id)
	#model = mx.model.FeedForward(
	# 	ctx = device,     # use GPU 0 for training, others are same as before
	# 	symbol = lenet,   		  # network structure
	# 	num_epoch = epochs,     	  # number of data passes for training
	# 	optimizer='Adam')
    
	#model.fit(
	# 	X=train_iter,  			# training data
	# 	eval_data=val_iter, 	# validation data
	# 	batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
	#)

	model = mx.mod.Module(lenet, context = device)

	model.fit(
		train_iter,  			# training data
		eval_data=val_iter, 	# validation data
		optimizer='Adam',
		num_epoch=epochs,
		initializer = mx.initializer.Xavier(),
		batch_end_callback = mx.callback.Speedometer(batch_size, 5) # output progress for each 200 data batches
	)

	return model