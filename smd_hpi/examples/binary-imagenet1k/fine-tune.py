import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
  
    parser.set_defaults(
        # network
        network          = 'inception-bn-binary',
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
        batch_size       = 32,
        optimizer        = 'sgd',
        disp_batches     = 10,
        top_k            = 5,
        data_train       = '/data/haojin/imagenet1k/imagenet1k-train',
        data_val         = '/data/haojin/imagenet1k/imagenet1k-val'
    )
    args = parser.parse_args()

    #load pretrained model
    if args.pretrained_model:
        sym, args_params, aux_params = mx.model.load_checkpoint(args.pretrained_model, 126)#inception-bn

    # save model
    checkpoint = _save_model(args, kv.rank)
    
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    initializer   = mx.init.Xavier(
       rnd_type='gaussian', factor_type="in", magnitude=2)

    # train
    fit.fit(args               = args,
            network            = sym,
            data_loader        = data.get_rec_iter,
            arg_params         = args_params,
            aux_params         = aux_params,
            begin_epoch        = args.load_epoch if args.load_epoch else 0,
            optimizer          = args.optimizer,
            optimizer_params   = optimizer_params,
            initializer        = initializer,
            arg_params         = arg_params,
            aux_params         = aux_params,
            epoch_end_callback = checkpoint,
            allow_missing      = True,
            monitor            = monitor)
