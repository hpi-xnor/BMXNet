import argparse
from common import find_mxnet
import mxnet as mx
import time
import os
import logging
import sys


def score(model_prefix, epoch, data_val, metrics, gpus, batch_size, rgb_mean,
          image_shape='3,224,224', data_nthreads=4):
    # create data iterator
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    data = mx.io.ImageRecordIter(
        path_imgrec        = data_val,
        label_width        = 1,
        mean_r             = rgb_mean[0],
        mean_g             = rgb_mean[1],
        mean_b             = rgb_mean[2],
        preprocess_threads = data_nthreads,
        batch_size         = batch_size,
        data_shape         = data_shape,
        rand_crop          = False,
        rand_mirror        = False)

    # download model
#    dir_path = os.path.dirname(os.path.realpath(__file__))
#    (prefix, epoch) = modelzoo.download_model(
#        model, os.path.join(dir_path, 'model'))

    # create module
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    logging.info('Info: model scoring started...')
    total_bat = 0
    num = 0
    tic = time.time()
    for batch in data:        
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        #if num >= 1000:
        #    total_bat = time.time() - tic
        #    logging.info('%f second per image, total time: %f', total_bat/num, total_bat)
        #    break
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model-prefix', type=str, required=True,
                        help = 'the model prefix.')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939')
    parser.add_argument('--data-val', type=str, required=True)
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    parser.add_argument('--epoch', type=int, default=0,
                        help='epoch of the model')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(metrics = metrics, **vars(args))
    logging.info('Finished with %f images per second', speed)
    for m in metrics:
        logging.info(m.get())
