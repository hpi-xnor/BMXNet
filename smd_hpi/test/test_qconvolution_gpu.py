import mxnet as mx
import numpy as np
from mxnet.test_utils import check_consistency


# Apply N symbols against each of M contexts, checking that all NxM combinations match.
def check_consistency_NxM(sym_list, ctx_list, arg_params=None):
    # e.g. if sym_list=[sym1, sym2] and ctx_list=[ctx1, ctx2, ctx3], then resulting lists are:
    # sym_list=[sym1, sym1, sym1, sym2, sym2, sym2] and ctx_list=[ctx1, ctx2, ctx3, ctx1, ctx2, ctx3]
    check_consistency(np.repeat(sym_list, len(ctx_list)), ctx_list * len(sym_list), arg_params=arg_params)

def test_convolution_options():
    # 2D convolution
    ctx_list = [
        # {'ctx': mx.gpu(0), 'conv_data': (2, 64, 7, 7), 'type_dict': {'conv_data': np.float64}},
        {'ctx': mx.gpu(0), 'conv_data': (2, 64, 7, 7), 'type_dict': {'conv_data': np.float32}},
        # {'ctx': mx.gpu(0), 'conv_data': (2, 64, 7, 7), 'type_dict': {'conv_data': np.float16}},
        # {'ctx': mx.cpu(0), 'conv_data': (2, 64, 7, 7), 'type_dict': {'conv_data': np.float64}},
        {'ctx': mx.cpu(0), 'conv_data': (2, 64, 7, 7), 'type_dict': {'conv_data': np.float32}},
    ]

    use_fixes_test_case = True
    if use_fixes_test_case:
        _rng = np.random.RandomState(1234)
        arg_params = {
            'conv_data': _rng.normal(size=(2, 64, 7, 7)),
            'conv_weight': _rng.normal(size=(3, 64, 1, 1)),
        }
    else:
        arg_params = None

    # Pad > 0
    sym = mx.sym.QConvolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    sym_no_cudnn = mx.sym.QConvolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    #sym_fp = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv', no_bias=True)
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # Stride > 1
    sym = mx.sym.QConvolution(num_filter=3, kernel=(3,3), stride=(2,2), name='conv')
    sym_no_cudnn = mx.sym.QConvolution(num_filter=3, kernel=(3,3), stride=(2,2), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # # Dilate > 1
    # sym = mx.sym.QConvolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='conv')
    # sym_no_cudnn = mx.sym.QConvolution(num_filter=3, kernel=(3,3), dilate=(2,2), cudnn_off=True, name='conv')
    # check_consistency_NxM([sym, sym_no_cudnn], ctx_list)

    # 1x1 convolution
    sym = mx.sym.QConvolution(num_filter=3, kernel=(1,1), pad=(0,0), name='conv')
    sym_no_cudnn = mx.sym.QConvolution(num_filter=3, kernel=(1,1), pad=(0,0), cudnn_off=True, name='conv')
    check_consistency_NxM([sym, sym_no_cudnn], ctx_list, arg_params=arg_params)
