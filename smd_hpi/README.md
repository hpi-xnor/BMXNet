### Quantized and binarized operations in mxnet

The structure should be fairly self explanatory.

- ``src`` quantized operators that get compiled into the mxnet library
- ``tools`` tools developed for working with binarized networks
    - **model-converter**: a tool to pack a trained binary model so that each weight just uses 1 bit of storage.
    - **docker**: a simple Dockerfile to setup a container for mxnet with all dependencies, build it with ``docker build -t mxnet``, then run it with ``docker run -t -i mxnet``
    - ``amalgamate_mxnet_mac.sh``: this script will amalgamate the mxnet library into a single file and perform some modifications needed to compile on macOS and iOS
- ``test`` unit tests for those operators // @todo
- ``examples`` several projects demonstrating the binarized/quantized operators
    - [binary_mnist](examples/binary_mnist) train and predict with a LeNet on the MNIST dataset
    - [binary-imagenet1k](examples/binary-imagenet1k) train and predict with a ResNet18 on the imagenet or cifar10 dataset
- ``binary_models`` a collection of pre-trained binarized models over MNIST, CIFAR-10 and ImageNet dataset. The model accuracy has been described in our paper.
