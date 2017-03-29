import mxnet as mx


class BinaryInitializer(mx.init.Initializer):
    """
    Customized initializer for binary weights
    """
    def __init__(self):
        pass

    def _init_default(self, name, arr):
        if name.endswith("binarized"):
            self._init_zero(name, arr)
        else:
            raise ValueError('Unknown initialization pattern for %s' % name)
