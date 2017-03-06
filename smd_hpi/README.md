### quantized and binarized operations in mxnet

The structure should be fairly self explanatory.

- ``src`` quantized operators that get compiled into the mxnet library
- ``test`` unit tests for those operators
- ``examples`` several projects demonstrating the binarized/quantized operators
    - [binary_mnist](examples/binary_mnist) train and predict with a LeNet on the MNIST dataset
    - [ios-prediction](examples/ios-prediction) small ios app that can load and apply an mxnet model
