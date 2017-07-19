export MXNET_ROOT=`pwd`/..

# set this to root folder of ndk toolchain
export NDK_TOOLCHAIN_ROOT=/home/christian/workspace/android/arm-toolchain

# Change this to path or specify in make command
ifndef OPENBLAS_ROOT
    export OPENBLAS_ROOT=/usr/local/opt/openblas
endif

# Whether use minimum build without blas and SSE, this will make the library super slow
ifndef MIN
	export MIN=0
	DEFS=-DMSHADOW_USE_CBLAS=1
else
	DEFS=-DMSHADOW_USE_CBLAS=0
endif

ifndef ANDROID
    export ANDROID=0
else
    DEFS+=-DMSHADOW_USE_SSE=0
endif

# Use locally installed emscripten if not specified
ifndef EMCC
    EMCC=emcc
endif

ifeq ($(OPENMP), 1)
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
else
    export OPENMP=0
    DEFS+=-DDISABLE_OPENMP=1
endif

.PHONY: all clean

# definitions for binarization
DEFS += -DBINARY_WORD_32=0 -DBINARY_WORD_64=1

DEFS+=-DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_MKL=0 -DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DDMLC_LOG_STACK_TRACE=0
DEFS+=-DMSHADOW_FORCE_STREAM -DMXNET_USE_OPENCV=0 -DMXNET_PREDICT_ONLY=1
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall $(DEFS)
ifneq ($(MIN), 1)
	CFLAGS += -I${OPENBLAS_ROOT} -I${OPENBLAS_ROOT}/include
	LDFLAGS+= -L${OPENBLAS_ROOT} -L${OPENBLAS_ROOT}/lib
        
	# Define which blas is installed. Uses OpenBLAS by default.
	ifeq ($(USE_BLAS), atlas)
                LDFLAGS += -lcblas
        else ifeq ($(USE_BLAS), blas)
                LDFLAGS += -lblas
        else
		LDFLAGS += -lopenblas
	endif
endif

ifeq ($(ANDROID), 1)
    CFLAGS += -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=softfp
    CFLAGS += -I${NDK_TOOLCHAIN_ROOT}/include/c++/4.9.x
    LDFLAGS += -march=armv7-a

    ifeq ($(CC), arm-linux-androideabi-clang)
        LDFLAGS += -L${NDK_TOOLCHAIN_ROOT}/arm-linux-androideabi/lib/armv7-a
    endif

    ifeq ($(CC), arm-linux-androideabi-gcc)
        DEFS+=-isystem ${NDK_TOOLCHAIN_ROOT}/include/c++/4.9.x
    endif
endif

all: android libmxnet_predict.a ${MXNET_ROOT}/lib/libmxnet_predict.so

nnvm.d:
	./prep_nnvm.sh $(ANDROID) $(OPENMP)

dmlc.d: dmlc-minimum0.cc
	${CXX} ${CFLAGS} -M -MT dmlc-minimum0.o \
	-I ${MXNET_ROOT}/dmlc-core/include \
	-D__MIN__=$(MIN) $+ > dmlc.d


mxnet_predict0.d: mxnet_predict0.cc nnvm.d dmlc.d
	${CXX} ${CFLAGS} -M -MT mxnet_predict0.o \
	-I ${MXNET_ROOT}/ -I ${MXNET_ROOT}/mshadow/ -I ${MXNET_ROOT}/dmlc-core/include -I ${MXNET_ROOT}/dmlc-core/src \
	-I ${MXNET_ROOT}/nnvm/include \
	-I ${MXNET_ROOT}/dlpack/include \
	-I ${MXNET_ROOT}/include \
	-D__MIN__=$(MIN) mxnet_predict0.cc mxnet_hpi0.cc > mxnet_predict0.d
	cat dmlc.d >> mxnet_predict0.d
	cat nnvm.d >> mxnet_predict0.d

mxnet_predict-all.cc:  mxnet_predict0.d dmlc-minimum0.cc nnvm.cc mxnet_predict0.cc mxnet_hpi0.cc
	@echo "Generating amalgamation to " $@
	python2 ./amalgamation.py $+ $@ $(MIN) $(ANDROID) $(OPENMP)

mxnet_predict-all.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libmxnet_predict.a: mxnet_predict-all.o
	${AR} rcs libmxnet_predict.a $+

jni_libmxnet_predict.o: mxnet_predict-all.cc jni/predictor.cc 
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor.cc

jni_libmxnet_predict.so: jni_libmxnet_predict.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

ifneq ($(ANDROID), 1)
        android:
else
        CFLAGS+=  -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -O3
        LDFLAGS+=  -Wl,--no-warn-mismatch -lm_hard
        android: jni_libmxnet_predict.so
endif

libmxnet_predict.js: mxnet_predict-all.cc
	${EMCC} -std=c++11 -O2 $(DEFS) -DMSHADOW_USE_SSE=0 -D__MXNET_JS__  -o $@ $+ \
	-s EXPORTED_FUNCTIONS="['_MXPredCreate', \
	                        '_MXPredGetOutputShape', \
	                        '_MXPredSetInput', \
	                        '_MXPredForward', \
	                        '_MXPredPartialForward', \
	                        '_MXPredGetOutput', \
	                        '_MXPredFree', \
	                        '_MXNDListCreate', \
	                        '_MXNDListGet', \
	                        '_MXNDListFree']" \
	-s ALLOW_MEMORY_GROWTH=1

${MXNET_ROOT}/lib/libmxnet_predict.so:  mxnet_predict-all.o
	@mkdir -p ${MXNET_ROOT}/lib
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	ls -alh $@

clean:
	rm -f *.d *.o *.so *.a *.js *.js.mem mxnet_predict-all.cc nnvm.cc
