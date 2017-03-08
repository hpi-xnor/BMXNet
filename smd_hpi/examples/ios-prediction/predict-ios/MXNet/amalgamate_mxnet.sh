#!/bin/bash
# Copyright (c) 2017 by Contributors
# \file amalgamate_mxnet.h
# \brief prepare mxnet for usage in xcode project
# \author HPI-DeepLearning

PROJECT_DIR=${PROJECT_DIR:-`pwd`}
MXNET_ROOT=${MXNET_ROOT:-`pwd`/../../..}

set -e # fail on error

echo "> Launching amalgamation scripts"
cd $MXNET_ROOT/amalgamation
make mxnet_predict-all.cc

echo "> Copying amalgamated mxnet and prediction library header to xcode project"
cp $MXNET_ROOT/amalgamation/mxnet_predict-all.cc $PROJECT_DIR/predict-ios/MXNet
cp $MXNET_ROOT/include/mxnet/c_predict_api.h $PROJECT_DIR/predict-ios/MXNet

echo "> Changing include from cblas.h to Accelerate.h"
sed -i.backup 's/#include <cblas.h>/#include <Accelerate\/Accelerate.h>/g' $PROJECT_DIR/predict-ios/MXNet/mxnet_predict-all.cc
rm $PROJECT_DIR/predict-ios/MXNet/mxnet_predict-all.cc.backup

echo "> Removing emmintrin.h include"
sed -i.backup 's/#include <emmintrin.h>/ /g' $PROJECT_DIR/predict-ios/MXNet/mxnet_predict-all.cc
rm $PROJECT_DIR/predict-ios/MXNet/mxnet_predict-all.cc.backup
