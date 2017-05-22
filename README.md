# HPI-DeepLearning fork of mxnet 

A fork of the deep learning framework [mxnet](http://mxnet.io) to study and implement quantization and binarization in neural networks.

Our current efforts are focused on binarizing the inputs and weights of convolutional layers, enabling the use of performant bit operations instead of expensive matrix multiplications as described in:

[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)

# Setup

We use ``cmake`` to build the project. Make sure to install all the dependencies described [here](http://mxnet.io/get_started/install.html) in the ``install from source`` section. Adjust settings in cmake (build-type ``Release`` or ``Debug``, enable CUDA, OpenMP etc.)  

```shell
$ git clone --recursive https://github.com/hpi-xnor/mxnet.git # remember to include the --recursive
$ mkdir build && cd build
$ ccmake .. # or cmake, or GUI cmake
$ make -j `nproc`
```

This will generate the mxnet library. To be able to use it from python, be sure to add the location of the libray to your ``LD_LIBRARY_PATH`` as well as the mxnet python folder to your ``PYTHONPATH``:
```shell
$ export LD_LIBRARY_PATH=<mxnet-root>/build
$ export PYTHONPATH=<mxnet-root>/python
```
# Usage

Our main contribution are drop-in replacements for the Convolution and Activation layers of mxnet called **QConvoluion** and **QActivation**.

These can be used when specifying a model. They extend the parameters of the original Convolution layer of mxnet.

## Quantization

Set the QConvolution parameter ``act_bit`` to a value between 1 and 32 to quantize the weights and activation to that bitwidth.

The quantization on bitwidths ranging from 2 to 31 bit is mainly for scientific purpose. There is no speed or memory gain as the quantized values are still stored in full precision ``float`` variables.

### Binarization

To binarize the weights first set ``act_bit=1``. Then train your network (you can use CUDA). The resulting .params file will still contain binarized weights, but still store a single weight in one float. 

To convert your trained and saved network, call the model converter with your ``.params`` file: 
```shell
$ <mxnet-root>smd_hpi/tools/model_converter mnist-0001.params
```

This will generate a ``.params`` and ``.json`` file with prepended ``binarized_``. This model file will use only 1 bit of runtime memory and storage for every weight in the convolutional layers.

We have example python scripts to train and validate [resnet18](smd_hpi/examples/binary-imagenet1k) (cifar10, imagenet) and [lenet](md_hpi/examples/binary_mnist) (mnist) neural networks with binarized layers.

There are example applications running on iOS and Android that can utilize binarized networks. Find them in the following repos:
- [Android image classification](https://github.com/hpi-xnor/android-image-classification)
- [iOS image classification](https://github.com/hpi-xnor/ios-image-classification)
- [iOS handwritten digit detection](https://github.com/hpi-xnor/ios-mnist)

Have a look at our [source, tools and examples](smd_hpi) to find out more.

