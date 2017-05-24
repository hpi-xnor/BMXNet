FROM ubuntu
MAINTAINER Martin Fritzsche <martin.fritzsche@student.hpi.de>

RUN apt-get -yqq update && \
    apt-get -yqq upgrade && \
    apt-get -yqq install git sudo g++ make

RUN apt-get -yqq install cmake

RUN apt-get -yqq install libopenblas-dev && \
    apt-get -yqq install python python-tk python-pip

RUN pip install --upgrade pip && \
    pip install numpy && \
	pip install matplotlib

RUN git clone --recursive https://github.com/hpi-xnor/mxnet.git # get repo

RUN mkdir /mxnet/release && \
    cd /mxnet/release && \
	echo "CMAKE_BUILD_TYPE:STRING=Release" >> CMakeCache.txt && \
	echo "USE_CUDA:BOOL=OFF" >> CMakeCache.txt && \
    echo "USE_OPENCV:BOOL=OFF" >> CMakeCache.txt && \
    cmake /mxnet

RUN cd /mxnet/release && \
    make -j `nproc`

ENV LD_LIBRARY_PATH=/mxnet/release
ENV PYTHONPATH=/mxnet/python

ENTRYPOINT ["/bin/bash"]
