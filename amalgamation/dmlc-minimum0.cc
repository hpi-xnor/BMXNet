/*!
 * Copyright 2015 by Contributors.
 * \brief Mininum DMLC library Amalgamation, used for easy plugin of dmlc lib.
 *  Normally this is not needed.
 */
#if defined(__ANDROID__)
	#include <cstdio>
	#define fopen64 std::fopen
#endif

#include "../dmlc-core/src/io/line_split.cc"
#include "../dmlc-core/src/io/recordio_split.cc"
#include "../dmlc-core/src/io/input_split_base.cc"
#include "../dmlc-core/src/io/local_filesys.cc"
#include "../dmlc-core/src/data.cc"
#include "../dmlc-core/src/io.cc"
#include "../dmlc-core/src/recordio.cc"


