import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def add_binary_args(parser):
    parser.add_argument('--bits-w', type=int, default=1,
                       help='number of bits for weights')
    parser.add_argument('--bits-a', type=int, default=1,
                       help='number of bits for activations')

if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser(description="train cifar100",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 3)

    parser.add_argument('--pretrained', type=str,
                help='the pre-trained model')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                    help='save training log to file')
    add_binary_args(parser)

    parser.set_defaults(
        # network
        network        = 'cifar10',
        num_layers     = 8,
        # data
        data_train     = '/data/cifar/cifar100/train.rec',
        data_val       = '/data/cifar/cifar100/test.rec',
        num_classes    = 100,
        num_examples   = 50000,
        image_shape    = '3,32,32',
        #pad_size       = 4,
        # train
        batch_size     = 256,
        num_epochs     = 200,
        lr_step_epochs = '50,100,150',
        optimizer        = 'nadam',
        disp_batches     = 10,
        lr               = 0.1,
        top_k            = 5,
    )
    args = parser.parse_args()

    # set up logger    
    log_file = args.log_file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    #load pretrained
    args_params=None
    auxs_params=None
    
    # train
    if args_params and auxs_params:
        fit.fit(
            args, 
            sym, 
            data.get_rec_iter, 
            arg_params=args_params, 
            aux_params=auxs_params)
    else:
        fit.fit(
            args, 
            sym, 
            data.get_rec_iter)

