import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet1K",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 3)
    parser.add_argument('--pretrained', type=str,
                    help='the pre-trained model')

    parser.set_defaults(
        # network
        network          = 'inception-bn-binary',
        num_layers       = 22,
        # data
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 60,
        lr_step_epochs   = '20,30,40,50',
        lr               = 0.01,
        batch_size     = 32,
        optimizer        = 'Adam',
        disp_batches     = 10,
        top_k            = 5,
        data_train       = '/data/haojin/imagenet1k/imagenet1k-val',
        data_val         = '/data/haojin/imagenet1k/imagenet1k-val'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    #load pretrained
    _, args_params, auxs_params = mx.model.load_checkpoint(args.pretrained, 126)#inception-bn-0126.params

    # train
    fit.fit(args, sym, data.get_rec_iter, arg_params=args_params, aux_params=auxs_params)
