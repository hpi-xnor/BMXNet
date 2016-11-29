import mxnet as mx
import logging
import numpy as np
import matplotlib.pyplot as plt
import pdb
from dorefa_ops import get_dorefa
from math_ops import *

logging.getLogger().setLevel(logging.DEBUG)

BITW = 1
BITA = 8
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
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
	tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	# second conv layer
	conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
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
		return mx.sym.Activation(data=x, act_type="relu")    # still use relu for 32bit cases
	return mx.sym.Custom(data=x, op_type='clip_by_0_1')

def activate(x):
	return f_a(nonlin(x))


def get_binary_lenet():
	data = mx.symbol.Variable('data')

	# first conv layer
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)	
	conv1_q = f_w(conv1)
	#conv1_q = mx.sym.Custom(data=conv1_q, op_type='debug')
	tanh1 = activate(conv1_q)
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	# second conv layer
	conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
	conv2_q = f_w(conv2)
	tanh2 = activate(conv2_q)
	pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
	# first fullc layer
	flatten = mx.sym.Flatten(data=pool2)
	flatten = f_w(flatten)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
	fc1_q = f_w(fc1)
	tanh3 = activate(fc1_q)
	# second fullc
	fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
	# softmax loss
	lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

	return lenet


def train(train_img, val_img, train_lbl, val_lbl, batch_size, gpu_id=0):
	lenet = get_lenet()
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)

	model = mx.model.FeedForward(
		ctx = mx.gpu(gpu_id),     # use GPU 0 for training, others are same as before
		symbol = lenet,   		  # network structure    
		num_epoch = 10,     	  # number of data passes for training 
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

def train_binary(train_img, val_img, train_lbl, val_lbl, batch_size, gpu_id=0):
	lenet = get_binary_lenet()
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)

	model = mx.model.FeedForward(
		ctx = mx.gpu(gpu_id),     # use GPU 0 for training, others are same as before
		symbol = lenet,   		  # network structure    
		num_epoch = 10,     	  # number of data passes for training 
		learning_rate = 0.1,
		optimizer='Adam')

	model.fit(
		X=train_iter,  			# training data
		eval_data=val_iter, 	# validation data
		batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
	) 
	return model