#!/bin/bash
# Copyright (c) 2017 by Contributors
# \file amalgamate_mxnet.h
# \brief prepare mxnet for usage in xcode project
# \author HPI-DeepLearning

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MXNET_ROOT=${MXNET_ROOT:-$SCRIPT_DIR/../../../../..}

set -e # fail on error

echo "> Launching amalgamation scripts"
cd $MXNET_ROOT/amalgamation
make clean
make mxnet_predict-all.cc

echo "> Copying amalgamated mxnet and prediction library header to xcode project"
cp $MXNET_ROOT/amalgamation/mxnet_predict-all.cc $SCRIPT_DIR
cp $MXNET_ROOT/include/mxnet/c_predict_api.h $SCRIPT_DIR

echo "> Changing include from cblas.h to Accelerate.h"
sed -i.backup 's/#include <cblas.h>/#include <Accelerate\/Accelerate.h>/g' $SCRIPT_DIR/mxnet_predict-all.cc
rm $SCRIPT_DIR/mxnet_predict-all.cc.backup

echo "> Removing emmintrin.h include"
sed -i.backup 's/#include <emmintrin.h>/ /g' $SCRIPT_DIR/mxnet_predict-all.cc
rm $SCRIPT_DIR/mxnet_predict-all.cc.backup
