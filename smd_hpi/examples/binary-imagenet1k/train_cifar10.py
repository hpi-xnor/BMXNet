import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def add_binary_args(parser):
    parser.add_argument('--bits-w', type=int, default=1,
                       help='number of bits for weights')
    parser.add_argument('--bits-a', type=int, default=1,
                       help='number of bits for activations')

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 3)
    parser.add_argument('--pretrained', type=str,
                help='the pre-trained model')
    add_binary_args(parser)

    parser.set_defaults(
        # network
        network        = 'cifar10',
        num_layers     = 18,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples   = 50000,
        image_shape    = '3,32,32',
        # train
        batch_size     = 128,
        num_epochs     = 300,
        lr_step_epochs = '50,200,250',
        optimizer        = 'Nadam',
        disp_batches     = 100,
        lr               = 0.01,
        lr_factor        = 0.1
    )

    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                    help='save training log to file')
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
