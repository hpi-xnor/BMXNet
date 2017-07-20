import mxnet as mx
import logging
import numpy as np
import matplotlib.pyplot as plt
import pdb
from dorefa_ops import get_dorefa
from math_ops import *
from random import randint

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
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=64)
	tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")	
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	bn1 = mx.sym.BatchNorm(data=pool1)

	# second conv layer
	conv2 = mx.sym.Convolution(data=bn1, kernel=(5,5), num_filter=64)	
	#conv2 = mx.sym.Custom(data=conv2, op_type='debug')
	bn2 = mx.sym.BatchNorm(data=conv2)

	tanh2 = mx.sym.Activation(data=bn2, act_type="tanh")
	pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
	
	# first fullc layer
	flatten = mx.sym.Flatten(data=pool2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1000)
	bn3 = mx.sym.BatchNorm(data=fc1)
	tanh3 = mx.sym.Activation(data=bn3, act_type="tanh")

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
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=64)
	tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	bn1 = mx.sym.BatchNorm(data=pool1)

	# second conv layer
	ba1 = mx.sym.QActivation(data=bn1, act_bit=BITA, backward_only=True)
	conv2 = mx.sym.QConvolution(data=ba1, kernel=(5,5), num_filter=64, act_bit=BITW)
	bn2 = mx.sym.BatchNorm(data=conv2)
	pool2 = mx.sym.Pooling(data=bn2, pool_type="max", kernel=(2,2), stride=(2,2))
	
	# first fullc layer
	flatten = mx.sym.Flatten(data=pool2)	
	ba2 = mx.sym.QActivation(data=flatten,  act_bit=BITA, backward_only=True)	
	fc1 = mx.symbol.QFullyConnected(data=ba2, num_hidden=1000, act_bit=BITW)
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

def val(model_prefix, epoch_num, train_img, val_img, train_lbl, val_lbl, batch_size, gpu_id=0):
	device = mx.cpu()
	if gpu_id >= 0:
		device = mx.gpu(gpu_id)
	print('Preparing data for validation...')
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	print('Loading model...')	
	model = mx.mod.Module.load(model_prefix, epoch_num, context = device)
	
	model.bind(data_shapes=val_iter.provide_data,
	         label_shapes=val_iter.provide_label, for_training=False)  # create memory by given input shapes
	model.init_params()  # initial parameters with the default random initializer
	print('Evaluating...')
	metric = mx.metric.Accuracy()
	score = model.score(val_iter, metric)
	print score
	#print 'Validation accuracy: %f%%' % (score*100)

def classify(val_img, model_prefix, epoch_num, train_img, train_lbl, val_lbl, batch_size, gpu_id=0):
	device = mx.cpu()
	if gpu_id >= 0:
		device = mx.gpu(gpu_id)
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	model = mx.mod.Module.load(model_prefix, epoch_num, context = device)
		
	model.bind(data_shapes=val_iter.provide_data,
         	   label_shapes=val_iter.provide_label, for_training=False)  # create memory by given input shapes
	model.init_params()  # initial parameters with the default random initializer
	n = randint(0,100)
	#plt.imshow(val_img[n], cmap='Greys_r')
	#plt.axis('off')
	#plt.show()
	prob = model.predict(eval_data=val_iter, num_batch=1)[n].asnumpy() 
	print 'Classified as %d[%d] with probability %f' % (prob.argmax(), val_lbl[n], max(prob))

def train_binary(train_img, val_img, train_lbl, val_lbl, batch_size, epochs, gpu_id=0):
	lenet = get_binary_lenet()
	train_iter, val_iter = prepair_data(train_img, val_img, train_lbl, val_lbl, batch_size)
	device = mx.cpu()
	if gpu_id >= 0:
		device = mx.gpu(gpu_id)

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
