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

if __name__ == '__main__':
    # download data
#    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'cifar10',
        num_layers     = 5,
        # data
        data_train     = '/data/haojin/cifar/cifar10/cifar10_train.rec',
        data_val       = '/data/haojin/cifar/cifar10/cifar10_val.rec',
        num_classes    = 10,
        num_examples   = 50000,
        image_shape    = '3,32,32',        
        # train
        batch_size     = 256,
        num_epochs     = 100,
        lr_step_epochs = '10,20,30,40,50,60,70,80,90',
        optimizer        = 'sgd',
        disp_batches     = 10,
        lr               = 0.5,
	    lr_factor        = 0.5
    )
    args = parser.parse_args()

    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                    help='save training log to file')

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
