
#if defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#if !defined(__WIN32__)
#include <sys/stat.h>
#include <sys/types.h>

#if !defined(__ANDROID__) && (!defined(MSHADOW_USE_SSE) || MSHADOW_USE_SSE == 1)
//#include <emmintrin.h>
#endif

#endif

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <Accelerate/Accelerate.h>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
//#include <cuda.h>
//#include <cuda_fp16.h>
#include <deque>
//#include <emmintrin.h>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <istream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <streambuf>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

//===== EXPANDIND: mxnet_predict0.cc =====

// mexnet.cc

#define MSHADOW_FORCE_STREAM
#ifndef MSHADOW_USE_CBLAS
#define MSHADOW_USE_CBLAS 	1
#endif
#define MSHADOW_USE_CUDA 	0
#define MSHADOW_USE_MKL 	0
#define MSHADOW_RABIT_PS 	0
#define MSHADOW_DIST_PS 	0

#define MSHADOW_USE_SSE     0

#define MXNET_USE_OPENCV 	0
#define MXNET_PREDICT_ONLY 	1
#define DISABLE_OPENMP 1

//===== EXPANDIND: ../src/ndarray/unary_function.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file unary_function.cc
 * \brief CPU Implementation of unary function.
 */
// this will be invoked by gcc and compile CPU version
//===== EXPANDIND: ../src/ndarray/unary_function-inl.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file unary-function-inl.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
#define MXNET_NDARRAY_UNARY_FUNCTION_INL_H_

//===== EXPANDIND: ../src/common/tblob_op_registry.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file tblob_op_registry.h
 * \brief Helper registry to make registration of simple unary binary math function easy.
 * Register to this registry will enable both symbolic operator and NDArray operator in client.
 *
 * More complicated operators can be registered in normal way in ndarray and operator modules.
 */
#ifndef MXNET_COMMON_TBLOB_OP_REGISTRY_H_
#define MXNET_COMMON_TBLOB_OP_REGISTRY_H_

//===== EXPANDIND: ../dmlc-core/include/dmlc/registry.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.h
 * \brief Registry utility that helps to build registry singletons.
 */
#ifndef DMLC_REGISTRY_H_
#define DMLC_REGISTRY_H_

//===== EXPANDIND: ../dmlc-core/include/dmlc/base.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros
 */
#ifndef DMLC_BASE_H_
#define DMLC_BASE_H_

/*! \brief whether use glog for logging */
#ifndef DMLC_USE_GLOG
#define DMLC_USE_GLOG 0
#endif

/*!
 * \brief whether throw dmlc::Error instead of
 *  directly calling abort when FATAL error occured
 *  NOTE: this may still not be perfect.
 *  do not use FATAL and CHECK in destructors
 */
#ifndef DMLC_LOG_FATAL_THROW
#define DMLC_LOG_FATAL_THROW 1
#endif

/*!
 * \brief whether always log a message before throw
 * This can help identify the error that cannot be catched.
 */
#ifndef DMLC_LOG_BEFORE_THROW
#define DMLC_LOG_BEFORE_THROW 1
#endif

/*!
 * \brief Whether to use customized logger,
 * whose output can be decided by other libraries.
 */
#ifndef DMLC_LOG_CUSTOMIZE
#define DMLC_LOG_CUSTOMIZE 0
#endif

/*! \brief whether compile with hdfs support */
#ifndef DMLC_USE_HDFS
#define DMLC_USE_HDFS 0
#endif

/*! \brief whether compile with s3 support */
#ifndef DMLC_USE_S3
#define DMLC_USE_S3 0
#endif

/*! \brief whether or not use parameter server */
#ifndef DMLC_USE_PS
#define DMLC_USE_PS 0
#endif

/*! \brief whether or not use c++11 support */
#ifndef DMLC_USE_CXX11
#define DMLC_USE_CXX11 (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
                        __cplusplus >= 201103L || defined(_MSC_VER))
#endif

/// check if g++ is before 4.6
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 6
#pragma message("Will need g++-4.6 or higher to compile all"           \
                "the features in dmlc-core, "                           \
                "compile without c++0x, some features may be disabled")
#undef DMLC_USE_CXX11
#define DMLC_USE_CXX11 0
#endif
#endif

/*!
 * \brief Enable std::thread related modules,
 *  Used to disable some module in mingw compile.
 */
#ifndef DMLC_ENABLE_STD_THREAD
#define DMLC_ENABLE_STD_THREAD DMLC_USE_CXX11
#endif


/*!
 * \brief Disable copy constructor and assignment operator.
 *
 * If C++11 is supported, both copy and move constructors and
 * assignment operators are deleted explicitly. Otherwise, they are
 * only declared but not implemented. Place this macro in private
 * section if C++11 is not available.
 */
#ifndef DISALLOW_COPY_AND_ASSIGN
#  if DMLC_USE_CXX11
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&) = delete; \
       T(T&&) = delete; \
       T& operator=(T const&) = delete; \
       T& operator=(T&&) = delete
#  else
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&); \
       T& operator=(T const&)
#  endif
#endif

///
/// code block to handle optionally loading
///
#if !defined(__GNUC__)
#define fopen64 std::fopen
#endif
#ifdef _MSC_VER
#if _MSC_VER < 1900
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#define snprintf sprintf_s
#define vsnprintf vsprintf_s
#endif
#else
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#pragma message("Warning: FILE OFFSET BITS defined to be 32 bit")
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 std::fopen
#endif

extern "C" {
}
#endif

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#endif

/*! \brief namespace for dmlc */
namespace dmlc {
/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a vector
 * \param str input string
 * \return beginning address of a string
 */
inline char* BeginPtr(std::string &str) {  // NOLINT(*)
  if (str.length() == 0) return NULL;
  return &str[0];
}
/*!
 * \brief get the beginning address of a vector
 * \param str input string
 * \return beginning address of a string
 */
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define constexpr const
#define alignof __alignof
#endif

#endif  // DMLC_BASE_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/base.h =====

//===== EXPANDIND: ../dmlc-core/include/dmlc/logging.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of dmlc
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef DMLC_LOGGING_H_
#define DMLC_LOGGING_H_

namespace dmlc {
/*!
 * \brief exception class that will be thrown by
 *  default logger if DMLC_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept(a)
#endif

#if DMLC_USE_CXX11
#define DMLC_THROW_EXCEPTION noexcept(false)
#else
#define DMLC_THROW_EXCEPTION
#endif

#if DMLC_USE_GLOG

namespace dmlc {
inline void InitLogging(const char* argv0) {
  google::InitGoogleLogging(argv0);
}
}  // namespace dmlc

#else
// use a light version of glog

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#endif

namespace dmlc {
inline void InitLogging(const char* argv0) {
  // DO NOTHING
}

// Always-on checking
#define CHECK(x)                                           \
  if (!(x))                                                \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check "  \
      "failed: " #x << ' '
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x) \
  ((x) == NULL ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)
// Debug-only checking.
#ifdef NDEBUG
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

#if DMLC_LOG_CUSTOMIZE
#define LOG_INFO dmlc::CustomLogMessage(__FILE__, __LINE__)
#else
#define LOG_INFO dmlc::LogMessage(__FILE__, __LINE__)
#endif
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LOG_INFO
#define LOG_FATAL dmlc::LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

// Poor man version of VLOG
#define VLOG(x) LOG_INFO.stream()

#define LOG(severity) LOG_##severity.stream()
#define LG LOG_INFO.stream()
#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)

#ifdef NDEBUG
#define LOG_DFATAL LOG_ERROR
#define DFATAL ERROR
#define DLOG(severity) true ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#else
#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)
#endif

// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm *pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value);  // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d",
             pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
    return buffer_;
  }

 private:
  char buffer_[9];
};

class LogMessage {
 public:
  LogMessage(const char* file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() { log_stream_ << '\n'; }
  std::ostream& stream() { return log_stream_; }

 protected:
  std::ostream& log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};

// customized logger that can allow user to define where to log the message.
class CustomLogMessage {
 public:
  CustomLogMessage(const char* file, int line) {
    log_stream_ << "[" << DateLogger().HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~CustomLogMessage() {
    Log(log_stream_.str());
  }
  std::ostream& stream() { return log_stream_; }
  /*!
   * \brief customized logging of the message.
   * This function won't be implemented by libdmlc
   * \param msg The message to be logged.
   */
  static void Log(const std::string& msg);

 private:
  std::ostringstream log_stream_;
};

#if DMLC_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n";
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() DMLC_THROW_EXCEPTION {
    // throwing out of destructor is evil
    // hopefully we can do it here
    // also log the message before throw
#if DMLC_LOG_BEFORE_THROW
    LOG(ERROR) << log_stream_.str();
#endif
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

}  // namespace dmlc

#endif
#endif  // DMLC_LOGGING_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/logging.h =====

//===== EXPANDIND: ../dmlc-core/include/dmlc/parameter.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file parameter.h
 * \brief Provide lightweight util to do parameter setup and checking.
 */
#ifndef DMLC_PARAMETER_H_
#define DMLC_PARAMETER_H_

//===== EXPANDIND: ../dmlc-core/include/dmlc/json.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file json.h
 * \brief Lightweight JSON Reader/Writer that read save into C++ data structs.
 *  This includes STL composites and structures.
 */
#ifndef DMLC_JSON_H_
#define DMLC_JSON_H_

//===== EXPANDIND: ../dmlc-core/include/dmlc/type_traits.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file type_traits.h
 * \brief type traits information header
 */
#ifndef DMLC_TYPE_TRAITS_H_
#define DMLC_TYPE_TRAITS_H_

#if DMLC_USE_CXX11
#endif

namespace dmlc {
/*!
 * \brief whether a type is pod type
 * \tparam T the type to query
 */
template<typename T>
struct is_pod {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_pod<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};


/*!
 * \brief whether a type is integer type
 * \tparam T the type to query
 */
template<typename T>
struct is_integral {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_integral<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is floating point type
 * \tparam T the type to query
 */
template<typename T>
struct is_floating_point {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_floating_point<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is arithemetic type
 * \tparam T the type to query
 */
template<typename T>
struct is_arithmetic {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_arithmetic<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = (dmlc::is_integral<T>::value ||
                             dmlc::is_floating_point<T>::value);
#endif
};

/*!
 * \brief the string representation of type name
 * \tparam T the type to query
 * \return a const string of typename.
 */
template<typename T>
inline const char* type_name() {
  return "";
}

/*!
 * \brief whether a type have save/load function
 * \tparam T the type to query
 */
template<typename T>
struct has_saveload {
  /*! \brief the value of the traits */
  static const bool value = false;
};

/*!
 * \brief template to select type based on condition
 * For example, IfThenElseType<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam The typename to be returned if cond is false
*/
template<bool cond, typename Then, typename Else>
struct IfThenElseType;

/*! \brief macro to quickly declare traits information */
#define DMLC_DECLARE_TRAITS(Trait, Type, Value)       \
  template<>                                          \
  struct Trait<Type> {                                \
    static const bool value = Value;                  \
  }

/*! \brief macro to quickly declare traits information */
#define DMLC_DECLARE_TYPE_NAME(Type, Name)            \
  template<>                                          \
  inline const char* type_name<Type>() {              \
    return Name;                                      \
  }

//! \cond Doxygen_Suppress
// declare special traits when C++11 is not available
#if DMLC_USE_CXX11 == 0
DMLC_DECLARE_TRAITS(is_pod, char, true);
DMLC_DECLARE_TRAITS(is_pod, int8_t, true);
DMLC_DECLARE_TRAITS(is_pod, int16_t, true);
DMLC_DECLARE_TRAITS(is_pod, int32_t, true);
DMLC_DECLARE_TRAITS(is_pod, int64_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint8_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint16_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint32_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint64_t, true);
DMLC_DECLARE_TRAITS(is_pod, float, true);
DMLC_DECLARE_TRAITS(is_pod, double, true);

DMLC_DECLARE_TRAITS(is_integral, char, true);
DMLC_DECLARE_TRAITS(is_integral, int8_t, true);
DMLC_DECLARE_TRAITS(is_integral, int16_t, true);
DMLC_DECLARE_TRAITS(is_integral, int32_t, true);
DMLC_DECLARE_TRAITS(is_integral, int64_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint8_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint16_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint32_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint64_t, true);

DMLC_DECLARE_TRAITS(is_floating_point, float, true);
DMLC_DECLARE_TRAITS(is_floating_point, double, true);

#endif

DMLC_DECLARE_TYPE_NAME(float, "float");
DMLC_DECLARE_TYPE_NAME(double, "double");
DMLC_DECLARE_TYPE_NAME(int, "int");
DMLC_DECLARE_TYPE_NAME(uint32_t, "int (non-negative)");
DMLC_DECLARE_TYPE_NAME(uint64_t, "long (non-negative)");
DMLC_DECLARE_TYPE_NAME(std::string, "string");
DMLC_DECLARE_TYPE_NAME(bool, "boolean");

template<typename Then, typename Else>
struct IfThenElseType<true, Then, Else> {
  typedef Then Type;
};

template<typename Then, typename Else>
struct IfThenElseType<false, Then, Else> {
  typedef Else Type;
};
//! \endcond
}  // namespace dmlc
#endif  // DMLC_TYPE_TRAITS_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/type_traits.h =====


#if DMLC_USE_CXX11
#endif

namespace dmlc {
/*!
 * \brief Lightweight JSON Reader to read any STL compositions and structs.
 *  The user need to know the schema of the
 *
 */
class JSONReader {
 public:
  /*!
   * \brief Constructor.
   * \param is the input stream.
   */
  explicit JSONReader(std::istream *is)
      : is_(is),
        line_count_r_(0),
        line_count_n_(0) {}
  /*!
   * \brief Parse next JSON string.
   * \param out_str the output string.
   * \throw dmlc::Error when next token is not string
   */
  inline void ReadString(std::string *out_str);
  /*!
   * \brief Read Number.
   * \param out_value output value;
   * \throw dmlc::Error when next token is not number of ValueType.
   * \tparam ValueType type of the number
   */
  template<typename ValueType>
  inline void ReadNumber(ValueType *out_value);
  /*!
   * \brief Begin parsing an object.
   * \code
   *  std::string key;
   *  // value can be any type that is json serializable.
   *  std::string value;
   *  reader->BeginObject();
   *  while (reader->NextObjectItem(&key)) {
   *    // do somthing to key value
   *    reader->Read(&value);
   *  }
   * \endcode
   */
  inline void BeginObject();
  /*!
   * \brief Begin parsing an array.
   * \code
   *  // value can be any type that is json serializable.
   *  std::string value;
   *  reader->BeginArray();
   *  while (reader->NextObjectArrayItem(&value)) {
   *    // do somthing to value
   *  }
   * \endcode
   */
  inline void BeginArray();
  /*!
   * \brief Try to move to next object item.
   *  If this call is successful, user can proceed to call
   *  reader->Read to read in the value.
   * \param out_key the key to the next object.
   * \return true if the read is successful, false if we are at end of the object.
   */
  inline bool NextObjectItem(std::string *out_key);
  /*!
   * \brief Try to read the next element in the array.
   *  If this call is successful, user can proceed to call
   *  reader->Read to read in the value.
   * \return true if the read is successful, false if we are at end of the array.
   */
  inline bool NextArrayItem();
  /*!
   * \brief Read next ValueType.
   * \param out_value any STL or json readable type to be read
   * \throw dmlc::Error when the read of ValueType is not successful.
   * \tparam ValueType the data type to be read.
   */
  template<typename ValueType>
  inline void Read(ValueType *out_value);

  /*! \return current line count */
  inline std::string line_info() const {
    char temp[64];
    std::ostringstream os;
    os << " Line " << std::max(line_count_r_, line_count_n_);
    is_->getline(temp, 64);
    os << ", around ^`" << temp << "`";
    return os.str();
  }

 private:
  /*! \brief internal reader stream */
  std::istream *is_;
  /*! \brief "\\r" counter */
  size_t line_count_r_;
  /*! \brief "\\n" counter */
  size_t line_count_n_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  std::vector<size_t> scope_counter_;
  /*!
   * \brief Read next nonspace character.
   * \return the next nonspace character.
   */
  inline int NextNonSpace();
  /*!
   * \brief Read just before next nonspace but not read that.
   * \return the next nonspace character.
   */
  inline int PeekNextNonSpace();
};

/*!
 * \brief Lightweight json to write any STL compositions.
 */
class JSONWriter {
 public:
  /*!
   * \brief Constructor.
   * \param os the output stream.
   */
  explicit JSONWriter(std::ostream *os)
      : os_(os) {}
  /*!
   * \brief Write a string that do not contain escape characters.
   * \param s the string to be written.
   */
  inline void WriteNoEscape(const std::string &s);
  /*!
   * \brief Write a string that can contain escape characters.
   * \param s the string to be written.
   */
  inline void WriteString(const std::string &s);
  /*!
   * \brief Write a string that can contain escape characters.
   * \param v the value to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteNumber(const ValueType &v);
  /*!
   * \brief Start beginning of array.
   * \param multi_line whether to start an multi_line array.
   * \code
   *  writer->BeginArray();
   *  for (auto& v : vdata) {
   *    writer->WriteArrayItem(v);
   *  }
   *  writer->EndArray();
   * \endcode
   */
  inline void BeginArray(bool multi_line = true);
  /*! \brief Finish writing an array. */
  inline void EndArray();
  /*!
   * \brief Start beginning of array.
   * \param multi_line whether to start an multi_line array.
   * \code
   *  writer->BeginObject();
   *  for (auto& kv : vmap) {
   *    writer->WriteObjectKeyValue(kv.first, kv.second);
   *  }
   *  writer->EndObject();
   * \endcode
   */
  inline void BeginObject(bool multi_line = true);
  /*! \brief Finish writing object. */
  inline void EndObject();
  /*!
   * \brief Write key value pair in the object.
   * \param key the key of the object.
   * \param value the value of to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteObjectKeyValue(const std::string &key,
                                  const ValueType &value);
  /*!
   * \brief Write value into array.
   * \param value The value of to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteArrayItem(const ValueType &value);
  /*!
   * \brief Write value to json.
   * \param value any STL or json readable that can be written.
   * \tparam ValueType the data type to be write.
   */
  template<typename ValueType>
  inline void Write(const ValueType &value);

 private:
  /*! \brief Output stream */
  std::ostream *os_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  std::vector<size_t> scope_counter_;
  /*! \brief Record whether current is a multiline scope */
  std::vector<bool> scope_multi_line_;
  /*!
   * \brief Write seperating space and newlines
   */
  inline void WriteSeperator();
};

/*!
 * \brief Helper class to read JSON into a class or struct object.
 * \code
 *  struct Param {
 *    std::string name;
 *    int value;
 *    // define load function from JSON
 *    inline void Load(dmlc::JSONReader *reader) {
 *      dmlc::JSONStructReadHelper helper;
 *      helper.DeclareField("name", &name);
 *      helper.DeclareField("value", &value);
 *      helper.ReadAllFields(reader);
 *    }
 *  };
 * \endcode
 */
class JSONObjectReadHelper {
 public:
  /*!
   * \brief Declare field of type T
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareField(const std::string &key, T *addr) {
    DeclareFieldInternal(key, addr, false);
  }
  /*!
   * \brief Declare optional field of type T
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareOptionalField(const std::string &key, T *addr) {
    DeclareFieldInternal(key, addr, true);
  }
  /*!
   * \brief Read in all the declared fields.
   * \param reader the JSONReader to read the json.
   */
  inline void ReadAllFields(JSONReader *reader);

 private:
  /*!
   * \brief Internal function to declare field.
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \param optional if set to true, no error will be reported if the key is not presented.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareFieldInternal(const std::string &key, T *addr, bool optional);
  /*!
   * \brief The internal reader function.
   * \param reader The reader to read.
   * \param addr The memory address to read.
   */
  template<typename T>
  inline static void ReaderFunction(JSONReader *reader, void *addr);
  /*! \brief callback type to reader function */
  typedef void (*ReadFunction)(JSONReader *reader, void *addr);
  /*! \brief internal data entry */
  struct Entry {
    /*! \brief the reader function */
    ReadFunction func;
    /*! \brief the address to read */
    void *addr;
    /*! \brief whether it is optional */
    bool optional;
  };
  /*! \brief the internal map of reader callbacks */
  std::map<std::string, Entry> map_;
};

//! \cond Doxygen_Suppress
namespace json {

/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler;

template<typename ValueType>
struct NumericHandler {
  inline static void Write(JSONWriter *writer, const ValueType &value) {
    writer->WriteNumber<ValueType>(value);
  }
  inline static void Read(JSONReader *reader, ValueType *value) {
    reader->ReadNumber<ValueType>(value);
  }
};

template<typename ContainerType>
struct ArrayHandler {
  inline static void Write(JSONWriter *writer, const ContainerType &array) {
    typedef typename ContainerType::value_type ElemType;
    writer->BeginArray(array.size() > 10 || !dmlc::is_pod<ElemType>::value);
    for (typename ContainerType::const_iterator it = array.begin();
         it != array.end(); ++it) {
      writer->WriteArrayItem(*it);
    }
    writer->EndArray();
  }
  inline static void Read(JSONReader *reader, ContainerType *array) {
    typedef typename ContainerType::value_type ElemType;
    array->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      ElemType value;
      Handler<ElemType>::Read(reader, &value);
      array->insert(array->end(), value);
    }
  }
};

template<typename ContainerType>
struct MapHandler{
  inline static void Write(JSONWriter *writer, const ContainerType &map) {
    writer->BeginObject(map.size() > 1);
    for (typename ContainerType::const_iterator it = map.begin(); it != map.end(); ++it) {
      writer->WriteObjectKeyValue(it->first, it->second);
    }
    writer->EndObject();
  }
  inline static void Read(JSONReader *reader, ContainerType *map) {
    typedef typename ContainerType::mapped_type ElemType;
    map->clear();
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      ElemType value;
      reader->Read(&value);
      (*map)[key] = value;
    }
  }
};

template<typename T>
struct CommonJSONSerializer {
  inline static void Write(JSONWriter *writer, const T &value) {
    value.Save(writer);
  }
  inline static void Read(JSONReader *reader, T *value) {
    value->Load(reader);
  }
};

template<>
struct Handler<std::string> {
  inline static void Write(JSONWriter *writer, const std::string &value) {
    writer->WriteString(value);
  }
  inline static void Read(JSONReader *reader, std::string *str) {
    reader->ReadString(str);
  }
};

template<typename T>
struct Handler<std::vector<T> > : public ArrayHandler<std::vector<T> > {
};

template<typename K, typename V>
struct Handler<std::pair<K, V> > {
  inline static void Write(JSONWriter *writer, const std::pair<K, V> &kv) {
    writer->BeginArray();
    writer->WriteArrayItem(kv.first);
    writer->WriteArrayItem(kv.second);
    writer->EndArray();
  }
  inline static void Read(JSONReader *reader, std::pair<K, V> *kv) {
    reader->BeginArray();
    CHECK(reader->NextArrayItem())
        << "Expect array of length 2";
    Handler<K>::Read(reader, &(kv->first));
    CHECK(reader->NextArrayItem())
        << "Expect array of length 2";
    Handler<V>::Read(reader, &(kv->second));
    CHECK(!reader->NextArrayItem())
        << "Expect array of length 2";
  }
};

template<typename T>
struct Handler<std::list<T> > : public ArrayHandler<std::list<T> > {
};

template<typename V>
struct Handler<std::map<std::string, V> > : public MapHandler<std::map<std::string, V> > {
};

#if DMLC_USE_CXX11
template<typename V>
struct Handler<std::unordered_map<std::string, V> >
    : public MapHandler<std::unordered_map<std::string, V> > {
};
#endif

template<typename T>
struct Handler {
  inline static void Write(JSONWriter *writer, const T &data) {
    typedef typename dmlc::IfThenElseType<dmlc::is_arithmetic<T>::value,
                                          NumericHandler<T>,
                                          CommonJSONSerializer<T> >::Type THandler;
    THandler::Write(writer, data);
  }
  inline static void Read(JSONReader *reader, T *data) {
    typedef typename dmlc::IfThenElseType<dmlc::is_arithmetic<T>::value,
                                          NumericHandler<T>,
                                          CommonJSONSerializer<T> >::Type THandler;
    THandler::Read(reader, data);
  }
};
}  // namespace json

// implementations of JSONReader/Writer
inline int JSONReader::NextNonSpace() {
  int ch;
  do {
    ch = is_->get();
    if (ch == '\n') ++line_count_n_;
    if (ch == '\r') ++line_count_r_;
  } while (isspace(ch));
  return ch;
}

inline int JSONReader::PeekNextNonSpace() {
  int ch;
  while (true) {
    ch = is_->peek();
    if (ch == '\n') ++line_count_n_;
    if (ch == '\r') ++line_count_r_;
    if (!isspace(ch)) break;
    is_->get();
  }
  return ch;
}

inline void JSONReader::ReadString(std::string *out_str) {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '\"')
      << "Error at" << line_info()
      << ", Expect \'\"\' but get \'" << static_cast<char>(ch) << '\'';
  std::ostringstream os;
  while (true) {
    ch = is_->get();
    if (ch == '\"') break;
    if (ch == '\\') {
      os << is_->get();
    } else {
      os << static_cast<char>(ch);
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      LOG(FATAL)
          << "Error at" << line_info()
          << ", Expect \'\"\' but reach end of line ";
    }
  }
  *out_str = os.str();
}

template<typename ValueType>
inline void JSONReader::ReadNumber(ValueType *out_value) {
  *is_ >> *out_value;
  CHECK(!is_->fail())
      << "Error at" << line_info()
      << ", Expect number";
}

inline void JSONReader::BeginObject() {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '{')
      << "Error at" << line_info()
      << ", Expect \'{\' but get \'" << static_cast<char>(ch) << '\'';
  scope_counter_.push_back(0);
}

inline void JSONReader::BeginArray() {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '[')
      << "Error at" << line_info()
      << ", Expect \'{\' but get \'" << static_cast<char>(ch) << '\'';
  scope_counter_.push_back(0);
}

inline bool JSONReader::NextObjectItem(std::string *out_key) {
  bool next = true;
  if (scope_counter_.back() != 0) {
    int ch = NextNonSpace();
    if (ch == EOF) {
      next = false;
    } else if (ch == '}') {
      next = false;
    } else {
      CHECK_EQ(ch, ',')
          << "Error at" << line_info()
          << ", JSON object expect \'}\' or \',\' \'" << static_cast<char>(ch) << '\'';
    }
  } else {
    int ch = PeekNextNonSpace();
    if (ch == '}') {
      is_->get();
      next = false;
    }
  }
  if (!next) {
    scope_counter_.pop_back();
    return false;
  } else {
    scope_counter_.back() += 1;
    ReadString(out_key);
    int ch = NextNonSpace();
    CHECK_EQ(ch, ':')
        << "Error at" << line_info()
        << ", Expect \':\' but get \'" << static_cast<char>(ch) << '\'';
    return true;
  }
}

inline bool JSONReader::NextArrayItem() {
  bool next = true;
  if (scope_counter_.back() != 0) {
    int ch = NextNonSpace();
    if (ch == EOF) {
      next = false;
    } else if (ch == ']') {
      next = false;
    } else {
      CHECK_EQ(ch, ',')
          << "Error at" << line_info()
          << ", JSON array expect \']\' or \',\'. Get \'" << static_cast<char>(ch) << "\' instead";
    }
  } else {
    int ch = PeekNextNonSpace();
    if (ch == ']') {
      is_->get();
      next = false;
    }
  }
  if (!next) {
    scope_counter_.pop_back();
    return false;
  } else {
    scope_counter_.back() += 1;
    return true;
  }
}

template<typename ValueType>
inline void JSONReader::Read(ValueType *out_value) {
  json::Handler<ValueType>::Read(this, out_value);
}

inline void JSONWriter::WriteNoEscape(const std::string &s) {
  *os_ << '\"' << s << '\"';
}

inline void JSONWriter::WriteString(const std::string &s) {
  std::ostream &os = *os_;
  os << '\"';
  for (size_t i = 0; i < s.length(); ++i) {
    char ch = s[i];
    switch (ch) {
      case '\r': os << "\\r"; break;
      case '\n': os << "\\n"; break;
      case '\\': os << "\\\\"; break;
      case '\t': os << "\\t"; break;
      case '\"': os << "\\\""; break;
      default: os << ch;
    }
  }
  os << '\"';
}

template<typename ValueType>
inline void JSONWriter::WriteNumber(const ValueType &v) {
  *os_ << v;
}

inline void JSONWriter::BeginArray(bool multi_line) {
  *os_ << '[';
  scope_multi_line_.push_back(multi_line);
  scope_counter_.push_back(0);
}

inline void JSONWriter::EndArray() {
  CHECK_NE(scope_multi_line_.size(), 0);
  CHECK_NE(scope_counter_.size(), 0);
  bool newline = scope_multi_line_.back();
  size_t nelem = scope_counter_.back();
  scope_multi_line_.pop_back();
  scope_counter_.pop_back();
  if (newline && nelem != 0) WriteSeperator();
  *os_ << ']';
}

inline void JSONWriter::BeginObject(bool multi_line) {
  *os_ << "{";
  scope_multi_line_.push_back(multi_line);
  scope_counter_.push_back(0);
}

inline void JSONWriter::EndObject() {
  CHECK_NE(scope_multi_line_.size(), 0);
  CHECK_NE(scope_counter_.size(), 0);
  bool newline = scope_multi_line_.back();
  size_t nelem = scope_counter_.back();
  scope_multi_line_.pop_back();
  scope_counter_.pop_back();
  if (newline && nelem != 0) WriteSeperator();
  *os_ << '}';
}

template<typename ValueType>
inline void JSONWriter::WriteObjectKeyValue(const std::string &key,
                                            const ValueType &value) {
  std::ostream &os = *os_;
  if (scope_counter_.back() == 0) {
    WriteSeperator();
    os << '\"' << key << "\": ";
  } else {
    os << ", ";
    WriteSeperator();
    os << '\"' << key << "\": ";
  }
  scope_counter_.back() += 1;
  json::Handler<ValueType>::Write(this, value);
}

template<typename ValueType>
inline void JSONWriter::WriteArrayItem(const ValueType &value) {
  std::ostream &os = *os_;
  if (scope_counter_.back() != 0) {
    os << ", ";
  }
  scope_counter_.back() += 1;
  WriteSeperator();
  json::Handler<ValueType>::Write(this, value);
}

template<typename ValueType>
inline void JSONWriter::Write(const ValueType &value) {
  size_t nscope = scope_multi_line_.size();
  json::Handler<ValueType>::Write(this, value);
  CHECK_EQ(nscope, scope_multi_line_.size())
      << "Uneven scope, did you call EndArray/EndObject after each BeginObject/Array?";
}

inline void JSONWriter::WriteSeperator() {
  if (scope_multi_line_.size() == 0 || scope_multi_line_.back()) {
    *os_ << '\n' << std::string(scope_multi_line_.size() * 2, ' ');
  }
}

inline void JSONObjectReadHelper::ReadAllFields(JSONReader *reader) {
  reader->BeginObject();
  std::map<std::string, int> visited;
  std::string key;
  while (reader->NextObjectItem(&key)) {
    if (map_.count(key) != 0) {
      Entry e = map_[key];
      (*e.func)(reader, e.addr);
      visited[key] = 0;
    } else {
      std::ostringstream os;
      os << "JSONReader: Unknown field " << key << ", candidates are: \n";
      for (std::map<std::string, Entry>::iterator
               it = map_.begin(); it != map_.end(); ++it) {
        os << '\"' <<it->first << "\"\n";
      }
      LOG(FATAL) << os.str();
    }
  }
  if (visited.size() != map_.size()) {
    for (std::map<std::string, Entry>::iterator
             it = map_.begin(); it != map_.end(); ++it) {
      if (it->second.optional) continue;
      CHECK_NE(visited.count(it->first), 0)
          << "JSONReader: Missing field \"" << it->first << "\"\n At "
          << reader->line_info();
    }
  }
}

template<typename T>
inline void JSONObjectReadHelper::ReaderFunction(JSONReader *reader, void *addr) {
  json::Handler<T>::Read(reader, static_cast<T*>(addr));
}

template<typename T>
inline void JSONObjectReadHelper::
DeclareFieldInternal(const std::string &key, T *addr, bool optional) {
  CHECK_EQ(map_.count(key), 0)
      << "Adding duplicate field " << key;
  Entry e;
  e.func = ReaderFunction<T>;
  e.addr = static_cast<void*>(addr);
  e.optional = optional;
  map_[key] = e;
}

//! \endcond
}  // namespace dmlc
#endif  // DMLC_JSON_H_

//===== EXPANDED: ../dmlc-core/include/dmlc/json.h =====


namespace dmlc {
// this file is backward compatible with non-c++11
/*! \brief Error throwed by parameter checking */
struct ParamError : public dmlc::Error {
  /*!
   * \brief constructor
   * \param msg error message
   */
  explicit ParamError(const std::string &msg)
      : dmlc::Error(msg) {}
};

/*!
 * \brief Get environment variable with default.
 * \param key the name of environment variable.
 * \param default_value the default value of environment vriable.
 * \return The value received
 */
template<typename ValueType>
inline ValueType GetEnv(const char *key,
                        ValueType default_value);

/*! \brief internal namespace for parameter manangement */
namespace parameter {
// forward declare ParamManager
class ParamManager;
// forward declare FieldAccessEntry
class FieldAccessEntry;
// forward declare FieldEntry
template<typename DType>
class FieldEntry;
// forward declare ParamManagerSingleton
template<typename PType>
struct ParamManagerSingleton;
}  // namespace parameter
/*!
 * \brief Information about a parameter field in string representations.
 */
struct ParamFieldInfo {
  /*! \brief name of the field */
  std::string name;
  /*! \brief type of the field in string format */
  std::string type;
  /*!
   * \brief detailed type information string
   *  This include the default value, enum constran and typename.
   */
  std::string type_info_str;
  /*! \brief detailed description of the type */
  std::string description;
};

/*!
 * \brief Parameter is the base type every parameter struct should inheritate from
 * The following code is a complete example to setup parameters.
 * \code
 *   struct Param : public dmlc::Parameter<Param> {
 *     float learning_rate;
 *     int num_hidden;
 *     std::string name;
 *     // declare parameters in header file
 *     DMLC_DECLARE_PARAMETER(Param) {
 *       DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000);
 *       DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f);
 *       DMLC_DECLARE_FIELD(name).set_default("hello");
 *     }
 *   };
 *   // register it in cc file
 *   DMLC_REGISTER_PARAMETER(Param);
 * \endcode
 *
 *  After that, the Param struct will get all the functions defined in Parameter.
 * \tparam PType the type of parameter struct
 *
 * \sa DMLC_DECLARE_FIELD, DMLC_REGISTER_PARAMETER, DMLC_DECLARE_PARAMETER
 */
template<typename PType>
struct Parameter {
 public:
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This function will initialize the parameter struct, check consistency
   *  and throw error if something wrong happens.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   */
  template<typename Container>
  inline void Init(const Container &kwargs) {
    PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                  kwargs.begin(), kwargs.end(), NULL);
  }
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This is same as Init, but allow unknown arguments.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   * \return vector of pairs of unknown arguments.
   */
  template<typename Container>
  inline std::vector<std::pair<std::string, std::string> >
  InitAllowUnknown(const Container &kwargs) {
    std::vector<std::pair<std::string, std::string> > unknown;
    PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                  kwargs.begin(), kwargs.end(), &unknown);
    return unknown;
  }
  /*!
   * \brief Return a dictionary representation of the parameters
   * \return A dictionary that maps key -> value
   */
  inline std::map<std::string, std::string> __DICT__() const {
    std::vector<std::pair<std::string, std::string> > vec
        = PType::__MANAGER__()->GetDict(this->head());
    return std::map<std::string, std::string>(vec.begin(), vec.end());
  }
  /*!
   * \brief Write the parameters in JSON format.
   * \param writer JSONWriter used for writing.
   */
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->Write(this->__DICT__());
  }
  /*!
   * \brief Load the parameters from JSON.
   * \param reader JSONReader used for loading.
   * \throw ParamError when something go wrong.
   */
  inline void Load(dmlc::JSONReader *reader) {
    std::map<std::string, std::string> kwargs;
    reader->Read(&kwargs);
    this->Init(kwargs);
  }
  /*!
   * \brief Get the fields of the parameters.
   * \return List of ParamFieldInfo of each field.
   */
  inline static std::vector<ParamFieldInfo> __FIELDS__() {
    return PType::__MANAGER__()->GetFieldInfo();
  }
  /*!
   * \brief Print docstring of the parameter
   * \return the printed docstring
   */
  inline static std::string __DOC__() {
    std::ostringstream os;
    PType::__MANAGER__()->PrintDocString(os);
    return os.str();
  }

 protected:
  /*!
   * \brief internal function to allow declare of a parameter memember
   * \param manager the parameter manager
   * \param key the key name of the parameter
   * \param ref the reference to the parameter in the struct.
   */
  template<typename DType>
  inline parameter::FieldEntry<DType>& DECLARE(
      parameter::ParamManagerSingleton<PType> *manager,
      const std::string &key, DType &ref) { // NOLINT(*)
    parameter::FieldEntry<DType> *e =
        new parameter::FieldEntry<DType>();
    e->Init(key, this->head(), ref);
    manager->manager.AddEntry(key, e);
    return *e;
  }

 private:
  /*! \return Get head pointer of child structure */
  inline PType *head() const {
    return static_cast<PType*>(const_cast<Parameter<PType>*>(this));
  }
};

//! \cond Doxygen_Suppress
/*!
 * \brief macro used to declare parameter
 *
 * Example:
 * \code
 *   struct Param : public dmlc::Parameter<Param> {
 *     // declare parameters in header file
 *     DMLC_DECLARE_PARAMETER(Param) {
 *        // details of declarations
 *     }
 *   };
 * \endcode
 *
 * This macro need to be put in a source file so that registeration only happens once.
 * Refer to example code in Parameter for details
 *
 * \param PType the name of parameter struct.
 * \sa Parameter
 */
#define DMLC_DECLARE_PARAMETER(PType)                                   \
  static ::dmlc::parameter::ParamManager *__MANAGER__();                \
  inline void __DECLARE__(::dmlc::parameter::ParamManagerSingleton<PType> *manager) \

/*!
 * \brief macro to declare fields
 * \param FieldName the name of the field.
 */
#define DMLC_DECLARE_FIELD(FieldName)  this->DECLARE(manager, #FieldName, FieldName)

/*!
 * \brief macro to declare alias of a fields
 * \param FieldName the name of the field.
 * \param AliasName the name of the alias, must be declared after the field is declared.
 */
#define DMLC_DECLARE_ALIAS(FieldName, AliasName)  manager->manager.AddAlias(#FieldName, #AliasName)

/*!
 * \brief Macro used to register parameter.
 *
 * This macro need to be put in a source file so that registeration only happens once.
 * Refer to example code in Parameter for details
 * \param PType the type of parameter struct.
 * \sa Parameter
 */
#define DMLC_REGISTER_PARAMETER(PType)                                  \
  ::dmlc::parameter::ParamManager *PType::__MANAGER__() {               \
    static ::dmlc::parameter::ParamManagerSingleton<PType> inst(#PType); \
    return &inst.manager;                                               \
  }                                                                     \
  static ::dmlc::parameter::ParamManager &__make__ ## PType ## ParamManager__ = \
      (*PType::__MANAGER__())                                           \

//! \endcond
/*!
 * \brief internal namespace for parameter manangement
 * There is no need to use it directly in normal case
 */
namespace parameter {
/*!
 * \brief FieldAccessEntry interface to help manage the parameters
 *  Each entry can be used to access one parameter in the Parameter struct.
 *
 *  This is an internal interface used that is used to manage parameters
 */
class FieldAccessEntry {
 public:
  FieldAccessEntry()
      : has_default_(false) {}
  /*! \brief destructor */
  virtual ~FieldAccessEntry() {}
  /*!
   * \brief set the default value.
   * \param head the pointer to the head of the struct
   * \throw error if no default is presented
   */
  virtual void SetDefault(void *head) const = 0;
  /*!
   * \brief set the parameter by string value
   * \param head the pointer to the head of the struct
   * \param value the value to be set
   */
  virtual void Set(void *head, const std::string &value) const = 0;
  // check if value is OK
  virtual void Check(void *head) const {}
  /*!
   * \brief get the string representation of value.
   * \param head the pointer to the head of the struct
   */
  virtual std::string GetStringValue(void *head) const = 0;
  /*!
   * \brief Get field information
   * \return the corresponding field information
   */
  virtual ParamFieldInfo GetFieldInfo() const = 0;

 protected:
  /*! \brief whether this parameter have default value */
  bool has_default_;
  /*! \brief positional index of parameter in struct */
  size_t index_;
  /*! \brief parameter key name */
  std::string key_;
  /*! \brief parameter type */
  std::string type_;
  /*! \brief description of the parameter */
  std::string description_;
  /*!
   * \brief print string representation of default value
   * \parma os the stream to print the docstring to.
   */
  virtual void PrintDefaultValueString(std::ostream &os) const = 0;  // NOLINT(*)
  // allow ParamManager to modify self
  friend class ParamManager;
};

/*!
 * \brief manager class to handle parameter setting for each type
 *  An manager will be created for each parameter types.
 */
class ParamManager {
 public:
  /*! \brief destructor */
  ~ParamManager() {
    for (size_t i = 0; i < entry_.size(); ++i) {
      delete entry_[i];
    }
  }
  /*!
   * \brief find the access entry by parameter key
   * \param key the key of the parameter.
   * \return pointer to FieldAccessEntry, NULL if nothing is found.
   */
  inline FieldAccessEntry *Find(const std::string &key) const {
    std::map<std::string, FieldAccessEntry*>::const_iterator it =
        entry_map_.find(key);
    if (it == entry_map_.end()) return NULL;
    return it->second;
  }
  /*!
   * \brief set parameter by keyword arguments.
   * \param head head to the parameter field.
   * \param begin begin iterator of original kwargs
   * \param end end iterator of original kwargs
   * \param unknown_args optional, used to hold unknown arguments
   *          When it is specified, unknown arguments will be stored into here, instead of raise an error
   * \tparam RandomAccessIterator iterator type
   * \throw ParamError when there is unknown argument and unknown_args == NULL, or required argument is missing.
   */
  template<typename RandomAccessIterator>
  inline void RunInit(void *head,
                      RandomAccessIterator begin,
                      RandomAccessIterator end,
                      std::vector<std::pair<std::string, std::string> > *unknown_args) const {
    std::set<FieldAccessEntry*> selected_args;
    for (RandomAccessIterator it = begin; it != end; ++it) {
      FieldAccessEntry *e = Find(it->first);
      if (e != NULL) {
        e->Set(head, it->second);
        e->Check(head);
        selected_args.insert(e);
      } else {
        if (unknown_args != NULL) {
          unknown_args->push_back(*it);
        } else {
          std::ostringstream os;
          os << "Cannot find argument \'" << it->first << "\', Possible Arguments:\n";
          os << "----------------\n";
          PrintDocString(os);
          throw dmlc::ParamError(os.str());
        }
      }
    }

    for (std::map<std::string, FieldAccessEntry*>::const_iterator it = entry_map_.begin();
         it != entry_map_.end(); ++it) {
      if (selected_args.count(it->second) == 0) {
        it->second->SetDefault(head);
      }
    }
  }
  /*!
   * \brief internal function to add entry to manager,
   *  The manager will take ownership of the entry.
   * \param key the key to the parameters
   * \param e the pointer to the new entry.
   */
  inline void AddEntry(const std::string &key, FieldAccessEntry *e) {
    e->index_ = entry_.size();
    // TODO(bing) better error message
    if (entry_map_.count(key) != 0) {
      LOG(FATAL) << "key " << key << " has already been registered in " << name_;
    }
    entry_.push_back(e);
    entry_map_[key] = e;
  }
  /*!
   * \brief internal function to add entry to manager,
   *  The manager will take ownership of the entry.
   * \param key the key to the parameters
   * \param e the pointer to the new entry.
   */
  inline void AddAlias(const std::string& field, const std::string& alias) {
    if (entry_map_.count(field) == 0) {
      LOG(FATAL) << "key " << field << " has not been registered in " << name_;
    }
    if (entry_map_.count(alias) != 0) {
      LOG(FATAL) << "Alias " << alias << " has already been registered in " << name_;
    }
    entry_map_[alias] = entry_map_[field];
  }
  /*!
   * \brief set the name of parameter manager
   * \param name the name to set
   */
  inline void set_name(const std::string &name) {
    name_ = name;
  }
  /*!
   * \brief get field information of each field.
   * \return field information
   */
  inline std::vector<ParamFieldInfo> GetFieldInfo() const {
    std::vector<ParamFieldInfo> ret(entry_.size());
    for (size_t i = 0; i < entry_.size(); ++i) {
      ret[i] = entry_[i]->GetFieldInfo();
    }
    return ret;
  }
  /*!
   * \brief Print readible docstring to ostream, add newline.
   * \parma os the stream to print the docstring to.
   */
  inline void PrintDocString(std::ostream &os) const {  // NOLINT(*)
    for (size_t i = 0; i < entry_.size(); ++i) {
      ParamFieldInfo info = entry_[i]->GetFieldInfo();
      os << info.name << " : " << info.type_info_str << '\n';
      if (info.description.length() != 0) {
        os << "    " << info.description << '\n';
      }
    }
  }
  /*!
   * \brief Get internal parameters in vector of pairs.
   * \param head the head of the struct.
   * \param skip_default skip the values that equals default value.
   * \return the parameter dictionary.
   */
  inline std::vector<std::pair<std::string, std::string> > GetDict(void * head) const {
    std::vector<std::pair<std::string, std::string> > ret;
    for (std::map<std::string, FieldAccessEntry*>::const_iterator
            it = entry_map_.begin(); it != entry_map_.end(); ++it) {
      ret.push_back(std::make_pair(it->first, it->second->GetStringValue(head)));
    }
    return ret;
  }

 private:
  /*! \brief parameter struct name */
  std::string name_;
  /*! \brief positional list of entries */
  std::vector<FieldAccessEntry*> entry_;
  /*! \brief map of key to entry */
  std::map<std::string, FieldAccessEntry*> entry_map_;
};

//! \cond Doxygen_Suppress

// The following piece of code will be template heavy and less documented
// singleton parameter manager for certain type, used for initialization
template<typename PType>
struct ParamManagerSingleton {
  ParamManager manager;
  explicit ParamManagerSingleton(const std::string &param_name) {
    PType param;
    param.__DECLARE__(this);
    manager.set_name(param_name);
  }
};

// Base class of FieldEntry
// implement set_default
template<typename TEntry, typename DType>
class FieldEntryBase : public FieldAccessEntry {
 public:
  // entry type
  typedef TEntry EntryType;
  // implement set value
  virtual void Set(void *head, const std::string &value) const {
    std::istringstream is(value);
    is >> this->Get(head);
    if (!is.fail()) {
      while (!is.eof()) {
        int ch = is.get();
        if (ch == EOF) {
          is.clear(); break;
        }
        if (!isspace(ch)) {
          is.setstate(std::ios::failbit); break;
        }
      }
    }

    if (is.fail()) {
      std::ostringstream os;
      os << "Invalid Parameter format for " << key_
         << " expect " << type_ << " but value=\'" << value<< '\'';
      throw dmlc::ParamError(os.str());
    }
  }
  virtual std::string GetStringValue(void *head) const {
    std::ostringstream os;
    PrintValue(os, this->Get(head));
    return os.str();
  }
  virtual ParamFieldInfo GetFieldInfo() const {
    ParamFieldInfo info;
    std::ostringstream os;
    info.name = key_;
    info.type = type_;
    os << type_;
    if (has_default_) {
      os << ',' << " optional, default=";
      PrintDefaultValueString(os);
    } else {
      os << ", required";
    }
    info.type_info_str = os.str();
    info.description = description_;
    return info;
  }
  // implement set head to default value
  virtual void SetDefault(void *head) const {
    if (!has_default_) {
      std::ostringstream os;
      os << "Parameter " << key_
         << " is not presented";
      throw dmlc::ParamError(os.str());
    } else {
      this->Get(head) = default_value_;
    }
  }
  // return reference of self as derived type
  inline TEntry &self() {
    return *(static_cast<TEntry*>(this));
  }
  // implement set_default
  inline TEntry &set_default(const DType &default_value) {
    default_value_ = default_value;
    has_default_ = true;
    // return self to allow chaining
    return this->self();
  }
  // implement describe
  inline TEntry &describe(const std::string &description) {
    description_ = description;
    // return self to allow chaining
    return this->self();
  }
  // initialization function
  inline void Init(const std::string &key,
                   void *head, DType &ref) { // NOLINT(*)
    this->key_ = key;
    if (this->type_.length() == 0) {
      this->type_ = dmlc::type_name<DType>();
    }
    this->offset_ = ((char*)&ref) - ((char*)head);  // NOLINT(*)
  }

 protected:
  // print the value
  virtual void PrintValue(std::ostream &os, DType value) const { // NOLINT(*)
    os << value;
  }
  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    PrintValue(os, default_value_);
  }
  // get the internal representation of parameter
  // for example if this entry corresponds field param.learning_rate
  // then Get(&param) will return reference to param.learning_rate
  inline DType &Get(void *head) const {
    return *(DType*)((char*)(head) + offset_);  // NOLINT(*)
  }
  // internal offset of the field
  ptrdiff_t offset_;
  // default value of field
  DType default_value_;
};

// parameter base for numeric types that have range
template<typename TEntry, typename DType>
class FieldEntryNumeric
    : public FieldEntryBase<TEntry, DType> {
 public:
  FieldEntryNumeric()
      : has_begin_(false), has_end_(false) {}
  // implement set_range
  virtual TEntry &set_range(DType begin, DType end) {
    begin_ = begin; end_ = end;
    has_begin_ = true; has_end_ = true;
    return this->self();
  }
  // implement set_range
  virtual TEntry &set_lower_bound(DType begin) {
    begin_ = begin; has_begin_ = true;
    return this->self();
  }
  // consistency check for numeric ranges
  virtual void Check(void *head) const {
    FieldEntryBase<TEntry, DType>::Check(head);
    DType v = this->Get(head);
    if (has_begin_ && has_end_) {
      if (v < begin_ || v > end_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " exceed bound [" << begin_ << ',' << end_ <<']';
        throw dmlc::ParamError(os.str());
      }
    } else if (has_begin_ && v < begin_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " should be greater equal to " << begin_;
        throw dmlc::ParamError(os.str());
    } else if (has_end_ && v > end_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " should be smaller equal to " << end_;
        throw dmlc::ParamError(os.str());
    }
  }

 protected:
  // whether it have begin and end range
  bool has_begin_, has_end_;
  // data bound
  DType begin_, end_;
};

/*!
 * \brief FieldEntry defines parsing and checking behavior of DType.
 * This class can be specialized to implement specific behavior of more settings.
 * \tparam DType the data type of the entry.
 */
template<typename DType>
class FieldEntry :
      public IfThenElseType<dmlc::is_arithmetic<DType>::value,
                            FieldEntryNumeric<FieldEntry<DType>, DType>,
                            FieldEntryBase<FieldEntry<DType>, DType> >::Type {
};

// specialize define for int(enum)
template<>
class FieldEntry<int>
    : public FieldEntryNumeric<FieldEntry<int>, int> {
 public:
  // construct
  FieldEntry<int>() : is_enum_(false) {}
  // parent
  typedef FieldEntryNumeric<FieldEntry<int>, int> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    if (is_enum_) {
      std::map<std::string, int>::const_iterator it = enum_map_.find(value);
      std::ostringstream os;
      if (it == enum_map_.end()) {
        os << "Invalid Input: \'" << value;
        os << "\', valid values are: ";
        PrintEnums(os);
        throw dmlc::ParamError(os.str());
      } else {
        os << it->second;
        Parent::Set(head, os.str());
      }
    } else {
      Parent::Set(head, value);
    }
  }
  virtual ParamFieldInfo GetFieldInfo() const {
    if (is_enum_) {
      ParamFieldInfo info;
      std::ostringstream os;
      info.name = key_;
      info.type = type_;
      PrintEnums(os);
      if (has_default_) {
        os << ',' << "optional, default=";
        PrintDefaultValueString(os);
      } else {
        os << ", required";
      }
      info.type_info_str = os.str();
      info.description = description_;
      return info;
    } else {
      return Parent::GetFieldInfo();
    }
  }
  // add enum
  inline FieldEntry<int> &add_enum(const std::string &key, int value) {
    if ((enum_map_.size() != 0 && enum_map_.count(key) != 0) || \
        enum_back_map_.count(value) != 0) {
      std::ostringstream os;
      os << "Enum " << "(" << key << ": " << value << " exisit!" << ")\n";
      os << "Enums: ";
      for (std::map<std::string, int>::const_iterator it = enum_map_.begin();
           it != enum_map_.end(); ++it) {
        os << "(" << it->first << ": " << it->second << "), ";
      }
      throw dmlc::ParamError(os.str());
    }
    enum_map_[key] = value;
    enum_back_map_[value] = key;
    is_enum_ = true;
    return this->self();
  }

 protected:
  // enum flag
  bool is_enum_;
  // enum map
  std::map<std::string, int> enum_map_;
  // enum map
  std::map<int, std::string> enum_back_map_;
  // override print behavior
  virtual void PrintDefaultValueString(std::ostream &os) const { // NOLINT(*)
    os << '\'';
    PrintValue(os, default_value_);
    os << '\'';
  }
  // override print default
  virtual void PrintValue(std::ostream &os, int value) const {  // NOLINT(*)
    if (is_enum_) {
      CHECK_NE(enum_back_map_.count(value), 0)
          << "Value not found in enum declared";
      os << enum_back_map_.at(value);
    } else {
      os << value;
    }
  }


 private:
  inline void PrintEnums(std::ostream &os) const {  // NOLINT(*)
    os << '{';
    for (std::map<std::string, int>::const_iterator
             it = enum_map_.begin(); it != enum_map_.end(); ++it) {
      if (it != enum_map_.begin()) {
        os << ", ";
      }
      os << "\'" << it->first << '\'';
    }
    os << '}';
  }
};

// specialize define for string
template<>
class FieldEntry<std::string>
    : public FieldEntryBase<FieldEntry<std::string>, std::string> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<std::string>, std::string> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    this->Get(head) = value;
  }
  // override print default
  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    os << '\'' << default_value_ << '\'';
  }
};

// specialize define for bool
template<>
class FieldEntry<bool>
    : public FieldEntryBase<FieldEntry<bool>, bool> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<bool>, bool> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    std::string lower_case; lower_case.resize(value.length());
    std::transform(value.begin(), value.end(), lower_case.begin(), ::tolower);
    bool &ref = this->Get(head);
    if (lower_case == "true") {
      ref = true;
    } else if (lower_case == "false") {
      ref = false;
    } else if (lower_case == "1") {
      ref = true;
    } else if (lower_case == "0") {
      ref = false;
    } else {
      std::ostringstream os;
      os << "Invalid Parameter format for " << key_
         << " expect " << type_ << " but value=\'" << value<< '\'';
      throw dmlc::ParamError(os.str());
    }
  }

 protected:
  // print default string
  virtual void PrintValue(std::ostream &os, bool value) const {  // NOLINT(*)
    if (value) {
      os << "True";
    } else {
      os << "False";
    }
  }
};

}  // namespace parameter
//! \endcond

// implement GetEnv
template<typename ValueType>
inline ValueType GetEnv(const char *key,
                        ValueType default_value) {
  const char *val = getenv(key);
  if (val == NULL) return default_value;
  ValueType ret;
  parameter::FieldEntry<ValueType> e;
  e.Init(key, &ret, ret);
  e.Set(&ret, val);
  return ret;
}
}  // namespace dmlc
#endif  // DMLC_PARAMETER_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/parameter.h =====


namespace dmlc {
/*!
 * \brief Registry class.
 *  Registry can be used to register global singletons.
 *  The most commonly use case are factory functions.
 *
 * \tparam EntryType Type of Registry entries,
 *     EntryType need to name a name field.
 */
template<typename EntryType>
class Registry {
 public:
  /*! \return list of functions in the registry */
  inline static const std::vector<const EntryType*> &List() {
    return Get()->entry_list_;
  }
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const EntryType *Find(const std::string &name) {
    const std::map<std::string, EntryType*> &fmap = Get()->fmap_;
    typename std::map<std::string, EntryType*>::const_iterator p = fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return NULL;
    }
  }
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER__(const std::string& name) {
    CHECK_EQ(fmap_.count(name), 0)
        << name << " already registered";
    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    entry_list_.push_back(e);
    return *e;
  }
  /*!
   * \brief Internal function to either register or get registered entry
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER_OR_GET__(const std::string& name) {
    if (fmap_.count(name) == 0) {
      return __REGISTER__(name);
    } else {
      return *fmap_.at(name);
    }
  }
  /*!
   * \brief get a singleton of the Registry.
   *  This function can be defined by DMLC_ENABLE_REGISTRY.
   * \return get a singleton
   */
  static Registry *Get();

 private:
  /*! \brief list of entry types */
  std::vector<const EntryType*> entry_list_;
  /*! \brief map of name->function */
  std::map<std::string, EntryType*> fmap_;
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {
    for (typename std::map<std::string, EntryType*>::iterator p = fmap_.begin();
         p != fmap_.end(); ++p) {
      delete p->second;
    }
  }
};

/*!
 * \brief Common base class for function registry.
 *
 * \code
 *  // This example demonstrates how to use Registry to create a factory of trees.
 *  struct TreeFactory :
 *      public FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
 *  };
 *
 *  // in a independent cc file
 *  namespace dmlc {
 *  DMLC_REGISTRY_ENABLE(TreeFactory);
 *  }
 *  // register binary tree constructor into the registry.
 *  DMLC_REGISTRY_REGISTER(TreeFactory, TreeFactory, BinaryTree)
 *      .describe("Constructor of BinaryTree")
 *      .set_body([]() { return new BinaryTree(); });
 * \endcode
 *
 * \tparam EntryType The type of subclass that inheritate the base.
 * \tparam FunctionType The function type this registry is registerd.
 */
template<typename EntryType, typename FunctionType>
class FunctionRegEntryBase {
 public:
  /*! \brief name of the entry */
  std::string name;
  /*! \brief description of the entry */
  std::string description;
  /*! \brief additional arguments to the factory function */
  std::vector<ParamFieldInfo> arguments;
  /*! \brief Function body to create ProductType */
  FunctionType body;
  /*!
   * \brief Set the function body.
   * \param body Function body to set.
   * \return reference to self.
   */
  inline EntryType &set_body(FunctionType body) {
    this->body = body;
    return this->self();
  }
  /*!
   * \brief Describe the function.
   * \param description The description of the factory function.
   * \return reference to self.
   */
  inline EntryType &describe(const std::string &description) {
    this->description = description;
    return this->self();
  }
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline EntryType &add_argument(const std::string &name,
                                 const std::string &type,
                                 const std::string &description) {
    ParamFieldInfo info;
    info.name = name;
    info.type = type;
    info.type_info_str = info.type;
    info.description = description;
    arguments.push_back(info);
    return this->self();
  }
  /*!
   * \brief Append list if arguments to the end.
   * \param args Additional list of arguments.
   * \return reference to self.
   */
  inline EntryType &add_arguments(const std::vector<ParamFieldInfo> &args) {
    arguments.insert(arguments.end(), args.begin(), args.end());
    return this->self();
  }

 protected:
  /*!
   * \return reference of self as derived type
   */
  inline EntryType &self() {
    return *(static_cast<EntryType*>(this));
  }
};

/*!
 * \brief Macro to enable the registry of EntryType.
 * This macro must be used under namespace dmlc, and only used once in cc file.
 * \param EntryType Type of registry entry
 */
#define DMLC_REGISTRY_ENABLE(EntryType)                                 \
  template<>                                                            \
  Registry<EntryType > *Registry<EntryType >::Get() {                   \
    static Registry<EntryType > inst;                                   \
    return &inst;                                                       \
  }                                                                     \

/*!
 * \brief Generic macro to register an EntryType
 *  There is a complete example in FactoryRegistryEntryBase.
 *
 * \param EntryType The type of registry entry.
 * \param EntryTypeName The typename of EntryType, must do not contain namespace :: .
 * \param Name The name to be registered.
 * \sa FactoryRegistryEntryBase
 */
#define DMLC_REGISTRY_REGISTER(EntryType, EntryTypeName, Name)          \
  static EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ =      \
      ::dmlc::Registry<EntryType>::Get()->__REGISTER__(#Name)           \

/*!
 * \brief (Optional) Declare a file tag to current file that contains object registrations.
 *
 *  This will declare a dummy function that will be called by register file to
 *  incur a link dependency.
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_LINK_TAG
 */
#define DMLC_REGISTRY_FILE_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __() { return 0; }

/*!
 * \brief (Optional) Force link to all the objects registered in file tag.
 *
 *  This macro must be used in the same file as DMLC_REGISTRY_ENABLE and
 *  in the same namespace as DMLC_REGISTRY_FILE_TAG
 *
 *  DMLC_REGISTRY_FILE_TAG and DMLC_REGISTRY_LINK_TAG are optional macros for registration.
 *  They are used to encforce link of certain file into during static linking.
 *
 *  This is mainly used to solve problem during statically link a library which contains backward registration.
 *  Specifically, this avoids the objects in these file tags to be ignored by compiler.
 *
 *  For dynamic linking, this problem won't occur as everything is loaded by default.
 *
 *  Use of this is optional as it will create an error when a file tag do not exist.
 *  An alternative solution is always ask user to enable --whole-archieve during static link.
 *
 * \begincode
 * // in file objective_registry.cc
 * DMLC_REGISTRY_ENABLE(MyObjective);
 * DMLC_REGISTRY_LINK_TAG(regression_op);
 * DMLC_REGISTRY_LINK_TAG(rank_op);
 *
 * // in file regression_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(regression_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, logistic_reg, logistic_reg);
 * // ...
 *
 * // in file rank_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(rank_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, pairwiserank, pairwiserank);
 *
 * \endcode
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_ENABLE, DMLC_REGISTRY_FILE_TAG
 */
#define DMLC_REGISTRY_LINK_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __();                   \
  static int __reg_file_tag_ ## UniqueTag ## __ = __dmlc_registry_file_tag_ ## UniqueTag ## __();
}  // namespace dmlc
#endif  // DMLC_REGISTRY_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/registry.h =====

//===== EXPANDIND: ../include/mxnet/base.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuation of mxnet as well as basic data structure.
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_

//===== EXPANDIND: ../dmlc-core/include/dmlc/io.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file io.h
 * \brief defines serializable interface of dmlc
 */
#ifndef DMLC_IO_H_
#define DMLC_IO_H_

// include uint64_t only to make io standalone
#ifdef _MSC_VER
/*! \brief uint64 */
typedef unsigned __int64 uint64_t;
#else
#endif

/*! \brief namespace for dmlc */
namespace dmlc {
/*!
 * \brief interface of stream I/O for serialization
 */
class Stream {  // NOLINT(*)
 public:
  /*!
   * \brief reads data from a stream
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief writes data to a stream
   * \param ptr pointer to a memory buffer
   * \param size block size
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~Stream(void) {}
  /*!
   * \brief generic factory function
   *  create an stream, the stream will close the underlying files upon deletion
   *
   * \param uri the uri of the input currently we support
   *            hdfs://, s3://, and file:// by default file:// will be used
   * \param flag can be "w", "r", "a"
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  static Stream *Create(const char *uri,
                        const char* const flag,
                        bool allow_null = false);
  // helper functions to write/read different data structures
  /*!
   * \brief writes a data to stream
   *
   * dmlc::Stream support Write/Read of most STL
   * composites and base types.
   * If the data type is not supported, a compile time error will
   * be issued.
   *
   * \param data data to be written
   * \tparam T the data type to be written
   */
  template<typename T>
  inline void Write(const T &data);
  /*!
   * \brief loads a data from stream.
   *
   * dmlc::Stream support Write/Read of most STL
   * composites and base types.
   * If the data type is not supported, a compile time error will
   * be issued.
   *
   * \param out_data place holder of data to be deserialized
   * \return whether the load was successful
   */
  template<typename T>
  inline bool Read(T *out_data);
};

/*! \brief interface of i/o stream that support seek */
class SeekStream: public Stream {
 public:
  // virtual destructor
  virtual ~SeekStream(void) {}
  /*! \brief seek to certain position of the file */
  virtual void Seek(size_t pos) = 0;
  /*! \brief tell the position of the stream */
  virtual size_t Tell(void) = 0;
  /*!
   * \brief generic factory function
   *  create an SeekStream for read only,
   *  the stream will close the underlying files upon deletion
   *  error will be reported and the system will exit when create failed
   * \param uri the uri of the input currently we support
   *            hdfs://, s3://, and file:// by default file:// will be used
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  static SeekStream *CreateForRead(const char *uri,
                                   bool allow_null = false);
};

/*! \brief interface for serializable objects */
class Serializable {
 public:
  /*! \brief virtual destructor */
  virtual ~Serializable() {}
  /*!
  * \brief load the model from a stream
  * \param fi stream where to load the model from
  */
  virtual void Load(Stream *fi) = 0;
  /*!
  * \brief saves the model to a stream
  * \param fo stream where to save the model to
  */
  virtual void Save(Stream *fo) const = 0;
};

/*!
 * \brief input split creates that allows reading
 *  of records from split of data,
 *  independent part that covers all the dataset
 *
 *  see InputSplit::Create for definition of record
 */
class InputSplit {
 public:
  /*! \brief a blob of memory region */
  struct Blob {
    /*! \brief points to start of the memory region */
    void *dptr;
    /*! \brief size of the memory region */
    size_t size;
  };
  /*!
   * \brief hint the inputsplit how large the chunk size
   *  it should return when implementing NextChunk
   *  this is a hint so may not be enforced,
   *  but InputSplit will try adjust its internal buffer
   *  size to the hinted value
   * \param chunk_size the chunk size
   */
  virtual void HintChunkSize(size_t chunk_size) {}
  /*! \brief reset the position of InputSplit to beginning */
  virtual void BeforeFirst(void) = 0;
  /*!
   * \brief get the next record, the returning value
   *   is valid until next call to NextRecord or NextChunk
   *   caller can modify the memory content of out_rec
   *
   *   For text, out_rec contains a single line
   *   For recordio, out_rec contains one record content(with header striped)
   *
   * \param out_rec used to store the result
   * \return true if we can successfully get next record
   *     false if we reached end of split
   * \sa InputSplit::Create for definition of record
   */
  virtual bool NextRecord(Blob *out_rec) = 0;
  /*!
   * \brief get a chunk of memory that can contain multiple records,
   *  the caller needs to parse the content of the resulting chunk,
   *  for text file, out_chunk can contain data of multiple lines
   *  for recordio, out_chunk can contain multiple records(including headers)
   *
   *  This function ensures there won't be partial record in the chunk
   *  caller can modify the memory content of out_chunk,
   *  the memory is valid until next call to NextRecord or NextChunk
   *
   *  Usually NextRecord is sufficient, NextChunk can be used by some
   *  multi-threaded parsers to parse the input content
   *
   * \param out_chunk used to store the result
   * \return true if we can successfully get next record
   *     false if we reached end of split
   * \sa InputSplit::Create for definition of record
   * \sa RecordIOChunkReader to parse recordio content from out_chunk
   */
  virtual bool NextChunk(Blob *out_chunk) = 0;
  /*! \brief destructor*/
  virtual ~InputSplit(void) {}
  /*!
   * \brief factory function:
   *  create input split given a uri
   * \param uri the uri of the input, can contain hdfs prefix
   * \param part_index the part id of current input
   * \param num_parts total number of splits
   * \param type type of record
   *   List of possible types: "text", "recordio"
   *     - "text":
   *         text file, each line is treated as a record
   *         input split will split on '\\n' or '\\r'
   *     - "recordio":
   *         binary recordio file, see recordio.h
   * \return a new input split
   * \sa InputSplit::Type
   */
  static InputSplit* Create(const char *uri,
                            unsigned part_index,
                            unsigned num_parts,
                            const char *type);
};

/*!
 * \brief a std::ostream class that can can wrap Stream objects,
 *  can use ostream with that output to underlying Stream
 *
 * Usage example:
 * \code
 *
 *   Stream *fs = Stream::Create("hdfs:///test.txt", "w");
 *   dmlc::ostream os(fs);
 *   os << "hello world" << std::endl;
 *   delete fs;
 * \endcode
 */
class ostream : public std::basic_ostream<char> {
 public:
  /*!
   * \brief construct std::ostream type
   * \param stream the Stream output to be used
   * \param buffer_size internal streambuf size
   */
  explicit ostream(Stream *stream,
                   size_t buffer_size = (1 << 10))
      : std::basic_ostream<char>(NULL), buf_(buffer_size) {
    this->set_stream(stream);
  }
  // explictly synchronize the buffer
  virtual ~ostream() {
    buf_.pubsync();
  }
  /*!
   * \brief set internal stream to be stream, reset states
   * \param stream new stream as output
   */
  inline void set_stream(Stream *stream) {
    buf_.set_stream(stream);
    this->rdbuf(&buf_);
  }

  /*! \return how many bytes we written so far */
  inline size_t bytes_written(void) const {
    return buf_.bytes_out();
  }

 private:
  // internal streambuf
  class OutBuf : public std::streambuf {
   public:
    explicit OutBuf(size_t buffer_size)
        : stream_(NULL), buffer_(buffer_size), bytes_out_(0) {
      if (buffer_size == 0) buffer_.resize(2);
    }
    // set stream to the buffer
    inline void set_stream(Stream *stream);

    inline size_t bytes_out() const { return bytes_out_; }
   private:
    /*! \brief internal stream by StreamBuf */
    Stream *stream_;
    /*! \brief internal buffer */
    std::vector<char> buffer_;
    /*! \brief number of bytes written so far */
    size_t bytes_out_;
    // override sync
    inline int_type sync(void);
    // override overflow
    inline int_type overflow(int c);
  };
  /*! \brief buffer of the stream */
  OutBuf buf_;
};

/*!
 * \brief a std::istream class that can can wrap Stream objects,
 *  can use istream with that output to underlying Stream
 *
 * Usage example:
 * \code
 *
 *   Stream *fs = Stream::Create("hdfs:///test.txt", "r");
 *   dmlc::istream is(fs);
 *   is >> mydata;
 *   delete fs;
 * \endcode
 */
class istream : public std::basic_istream<char> {
 public:
  /*!
   * \brief construct std::ostream type
   * \param stream the Stream output to be used
   * \param buffer_size internal buffer size
   */
  explicit istream(Stream *stream,
                   size_t buffer_size = (1 << 10))
      : std::basic_istream<char>(NULL), buf_(buffer_size) {
    this->set_stream(stream);
  }
  virtual ~istream() {}
  /*!
   * \brief set internal stream to be stream, reset states
   * \param stream new stream as output
   */
  inline void set_stream(Stream *stream) {
    buf_.set_stream(stream);
    this->rdbuf(&buf_);
  }
  /*! \return how many bytes we read so far */
  inline size_t bytes_read(void) const {
    return buf_.bytes_read();
  }

 private:
  // internal streambuf
  class InBuf : public std::streambuf {
   public:
    explicit InBuf(size_t buffer_size)
        : stream_(NULL), bytes_read_(0),
          buffer_(buffer_size) {
      if (buffer_size == 0) buffer_.resize(2);
    }
    // set stream to the buffer
    inline void set_stream(Stream *stream);
    // return how many bytes read so far
    inline size_t bytes_read(void) const {
      return bytes_read_;
    }
   private:
    /*! \brief internal stream by StreamBuf */
    Stream *stream_;
    /*! \brief how many bytes we read so far */
    size_t bytes_read_;
    /*! \brief internal buffer */
    std::vector<char> buffer_;
    // override underflow
    inline int_type underflow();
  };
  /*! \brief input buffer */
  InBuf buf_;
};
}  // namespace dmlc

//===== EXPANDIND: ../dmlc-core/include/dmlc/serializer.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file serializer.h
 * \brief serializer template class that helps serialization.
 *  This file do not need to be directly used by most user.
 */
#ifndef DMLC_SERIALIZER_H_
#define DMLC_SERIALIZER_H_



#if DMLC_USE_CXX11
#endif

namespace dmlc {
/*! \brief internal namespace for serializers */
namespace serializer {
/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler;

//! \cond Doxygen_Suppress
/*!
 * \brief Serializer that redirect calls by condition
 * \tparam cond the condition
 * \tparam Then the serializer used for then condition
 * \tparam Else the serializer used for else condition
 * \tparam Return the type of data the serializer handles
 */
template<bool cond, typename Then, typename Else, typename Return>
struct IfThenElse;

template<typename Then, typename Else, typename T>
struct IfThenElse<true, Then, Else, T> {
  inline static void Write(Stream *strm, const T &data) {
    Then::Write(strm, data);
  }
  inline static bool Read(Stream *strm, T *data) {
    return Then::Read(strm, data);
  }
};
template<typename Then, typename Else, typename T>
struct IfThenElse<false, Then, Else, T> {
  inline static void Write(Stream *strm, const T &data) {
    Else::Write(strm, data);
  }
  inline static bool Read(Stream *strm, T *data) {
    return Else::Read(strm, data);
  }
};

/*! \brief Serializer for POD(plain-old-data) data */
template<typename T>
struct PODHandler {
  inline static void Write(Stream *strm, const T &data) {
    strm->Write(&data, sizeof(T));
  }
  inline static bool Read(Stream *strm, T *dptr) {
    return strm->Read((void*)dptr, sizeof(T)) == sizeof(T);  // NOLINT(*)
  }
};

// serializer for class that have save/load function
template<typename T>
struct SaveLoadClassHandler {
  inline static void Write(Stream *strm, const T &data) {
    data.Save(strm);
  }
  inline static bool Read(Stream *strm, T *data) {
    return data->Load(strm);
  }
};

/*!
 * \brief dummy class for undefined serialization.
 *   This is used to generate error message when user tries to
 *   serialize something that is not supported.
 * \tparam T the type to be serialized
 */
template<typename T>
struct UndefinedSerializerFor {
};

/*!
 * \brief Serializer handler for std::vector<T> where T is POD type.
 * \tparam T element type
 */
template<typename T>
struct PODVectorHandler {
  inline static void Write(Stream *strm, const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write(&sz, sizeof(sz));
    if (sz != 0) {
      strm->Write(&vec[0], sizeof(T) * vec.size());
    }
  }
  inline static bool Read(Stream *strm, std::vector<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    if (sz != 0) {
      size_t nbytes = sizeof(T) * size;
      return strm->Read(&(*out_vec)[0], nbytes) == nbytes;
    }
    return true;
  }
};

/*!
 * \brief Serializer handler for std::vector<T> where T can be composed type
 * \tparam T element type
 */
template<typename T>
struct ComposeVectorHandler {
  inline static void Write(Stream *strm, const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write(&sz, sizeof(sz));
    for (size_t i = 0; i < vec.size(); ++i) {
      Handler<T>::Write(strm, vec[i]);
    }
  }
  inline static bool Read(Stream *strm, std::vector<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    for (size_t i = 0; i < size; ++i) {
      if (!Handler<T>::Read(strm, &(*out_vec)[i])) return false;
    }
    return true;
  }
};

/*!
 * \brief Serializer handler for std::basic_string<T> where T is POD type.
 * \tparam T element type
 */
template<typename T>
struct PODStringHandler {
  inline static void Write(Stream *strm, const std::basic_string<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.length());
    strm->Write(&sz, sizeof(sz));
    if (sz != 0) {
      strm->Write(&vec[0], sizeof(T) * vec.length());
    }
  }
  inline static bool Read(Stream *strm, std::basic_string<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    if (sz != 0) {
      size_t nbytes = sizeof(T) * size;
      return strm->Read(&(*out_vec)[0], nbytes) == nbytes;
    }
    return true;
  }
};

/*! \brief Serializer for std::pair */
template<typename TA, typename TB>
struct PairHandler {
  inline static void Write(Stream *strm, const std::pair<TA, TB> &data) {
    Handler<TA>::Write(strm, data.first);
    Handler<TB>::Write(strm, data.second);
  }
  inline static bool Read(Stream *strm, std::pair<TA, TB> *data) {
    return Handler<TA>::Read(strm, &(data->first)) &&
        Handler<TB>::Read(strm, &(data->second));
  }
};

// set type handler that can handle most collection type case
template<typename ContainerType>
struct CollectionHandler {
  inline static void Write(Stream *strm, const ContainerType &data) {
    typedef typename ContainerType::value_type ElemType;
    // dump data to vector
    std::vector<ElemType> vdata(data.begin(), data.end());
    // serialize the vector
    Handler<std::vector<ElemType> >::Write(strm, vdata);
  }
  inline static bool Read(Stream *strm, ContainerType *data) {
    typedef typename ContainerType::value_type ElemType;
    std::vector<ElemType> vdata;
    if (!Handler<std::vector<ElemType> >::Read(strm, &vdata)) return false;
    data->clear();
    data->insert(vdata.begin(), vdata.end());
    return true;
  }
};


// handler that can handle most list type case
// this type insert function takes additional iterator
template<typename ListType>
struct ListHandler {
  inline static void Write(Stream *strm, const ListType &data) {
    typedef typename ListType::value_type ElemType;
    // dump data to vector
    std::vector<ElemType> vdata(data.begin(), data.end());
    // serialize the vector
    Handler<std::vector<ElemType> >::Write(strm, vdata);
  }
  inline static bool Read(Stream *strm, ListType *data) {
    typedef typename ListType::value_type ElemType;
    std::vector<ElemType> vdata;
    if (!Handler<std::vector<ElemType> >::Read(strm, &vdata)) return false;
    data->clear();
    data->insert(data->begin(), vdata.begin(), vdata.end());
    return true;
  }
};

//! \endcond

/*!
 * \brief generic serialization handler for type T
 *
 *  User can define specialization of this class to support
 *  composite serialization of their own class.
 *
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler {
  /*!
   * \brief write data to stream
   * \param strm the stream we write the data.
   * \param data the data obeject to be serialized
   */
  inline static void Write(Stream *strm, const T &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODHandler<T>,
               IfThenElse<dmlc::has_saveload<T>::value,
                          SaveLoadClassHandler<T>,
                          UndefinedSerializerFor<T>, T>,
               T>
        ::Write(strm, data);
  }
  /*!
   * \brief read data to stream
   * \param strm the stream to read the data.
   * \param data the pointer to the data obeject to read
   * \return whether the read is successful
   */
  inline static bool Read(Stream *strm, T *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODHandler<T>,
                      IfThenElse<dmlc::has_saveload<T>::value,
                                 SaveLoadClassHandler<T>,
                                 UndefinedSerializerFor<T>, T>,
                      T>
    ::Read(strm, data);
  }
};

//! \cond Doxygen_Suppress
template<typename T>
struct Handler<std::vector<T> > {
  inline static void Write(Stream *strm, const std::vector<T> &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODVectorHandler<T>,
               ComposeVectorHandler<T>, std::vector<T> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::vector<T> *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODVectorHandler<T>,
                      ComposeVectorHandler<T>,
                      std::vector<T> >
    ::Read(strm, data);
  }
};

template<typename T>
struct Handler<std::basic_string<T> > {
  inline static void Write(Stream *strm, const std::basic_string<T> &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODStringHandler<T>,
               UndefinedSerializerFor<T>,
               std::basic_string<T> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::basic_string<T> *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODStringHandler<T>,
                      UndefinedSerializerFor<T>,
                      std::basic_string<T> >
    ::Read(strm, data);
  }
};

template<typename TA, typename TB>
struct Handler<std::pair<TA, TB> > {
  inline static void Write(Stream *strm, const std::pair<TA, TB> &data) {
    IfThenElse<dmlc::is_pod<TA>::value && dmlc::is_pod<TB>::value,
               PODHandler<std::pair<TA, TB> >,
               PairHandler<TA, TB>,
               std::pair<TA, TB> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::pair<TA, TB> *data) {
    return IfThenElse<dmlc::is_pod<TA>::value && dmlc::is_pod<TB>::value,
                      PODHandler<std::pair<TA, TB> >,
                      PairHandler<TA, TB>,
                      std::pair<TA, TB> >
    ::Read(strm, data);
  }
};

template<typename K, typename V>
struct Handler<std::map<K, V> >
    : public CollectionHandler<std::map<K, V> > {
};

template<typename K, typename V>
struct Handler<std::multimap<K, V> >
    : public CollectionHandler<std::multimap<K, V> > {
};

template<typename T>
struct Handler<std::set<T> >
    : public CollectionHandler<std::set<T> > {
};

template<typename T>
struct Handler<std::multiset<T> >
    : public CollectionHandler<std::multiset<T> > {
};

template<typename T>
struct Handler<std::list<T> >
    : public ListHandler<std::list<T> > {
};

template<typename T>
struct Handler<std::deque<T> >
    : public ListHandler<std::deque<T> > {
};

#if DMLC_USE_CXX11
template<typename K, typename V>
struct Handler<std::unordered_map<K, V> >
    : public CollectionHandler<std::unordered_map<K, V> > {
};

template<typename K, typename V>
struct Handler<std::unordered_multimap<K, V> >
    : public CollectionHandler<std::unordered_multimap<K, V> > {
};

template<typename T>
struct Handler<std::unordered_set<T> >
    : public CollectionHandler<std::unordered_set<T> > {
};

template<typename T>
struct Handler<std::unordered_multiset<T> >
    : public CollectionHandler<std::unordered_multiset<T> > {
};
#endif
//! \endcond
}  // namespace serializer
}  // namespace dmlc
#endif  // DMLC_SERIALIZER_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/serializer.h =====


namespace dmlc {
// implementations of inline functions
template<typename T>
inline void Stream::Write(const T &data) {
  serializer::Handler<T>::Write(this, data);
}
template<typename T>
inline bool Stream::Read(T *out_data) {
  return serializer::Handler<T>::Read(this, out_data);
}

// implementations for ostream
inline void ostream::OutBuf::set_stream(Stream *stream) {
  if (stream_ != NULL) this->pubsync();
  this->stream_ = stream;
  this->setp(&buffer_[0], &buffer_[0] + buffer_.size() - 1);
}
inline int ostream::OutBuf::sync(void) {
  if (stream_ == NULL) return -1;
  std::ptrdiff_t n = pptr() - pbase();
  stream_->Write(pbase(), n);
  this->pbump(-static_cast<int>(n));
  bytes_out_ += n;
  return 0;
}
inline int ostream::OutBuf::overflow(int c) {
  *(this->pptr()) = c;
  std::ptrdiff_t n = pptr() - pbase();
  this->pbump(-static_cast<int>(n));
  if (c == EOF) {
    stream_->Write(pbase(), n);
    bytes_out_ += n;
  } else {
    stream_->Write(pbase(), n + 1);
    bytes_out_ += n + 1;
  }
  return c;
}

// implementations for istream
inline void istream::InBuf::set_stream(Stream *stream) {
  stream_ = stream;
  this->setg(&buffer_[0], &buffer_[0], &buffer_[0]);
}
inline int istream::InBuf::underflow() {
  char *bhead = &buffer_[0];
  if (this->gptr() == this->egptr()) {
    size_t sz = stream_->Read(bhead, buffer_.size());
    this->setg(bhead, bhead, bhead + sz);
    bytes_read_ += sz;
  }
  if (this->gptr() == this->egptr()) {
    return traits_type::eof();
  } else {
    return traits_type::to_int_type(*gptr());
  }
}
}  // namespace dmlc
#endif  // DMLC_IO_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/io.h =====

//===== EXPANDIND: ../mshadow/mshadow/tensor.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor.h
 * \brief header file of tensor data structure and functions
 *  This lib requires explicit memory allocation and de-allocation
 *  all the data structure Tensor<cpu,1>, Tensor<gpu,1> are like handles(pointers),
 *  no memory allocation is happening during calculation
 *
 *  For STL style tensor, see tensor_container.h
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_H_
#define MSHADOW_TENSOR_H_
//===== EXPANDIND: ../mshadow/mshadow/base.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file base.h
 * \brief definitions of base types, operators, macros functions
 *
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_BASE_H_
#define MSHADOW_BASE_H_
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#define NOMINMAX
#endif

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#endif
// macro defintiions
/*!
 * \brief if this macro is define to be 1,
 * mshadow should compile without any of other libs
 */
#ifndef MSHADOW_STAND_ALONE
#define MSHADOW_STAND_ALONE 0
#endif
/*! \brief whether do padding during allocation */
#ifndef MSHADOW_ALLOC_PAD
#define MSHADOW_ALLOC_PAD true
#endif
/*!
 * \brief
 *  x dimension of data must be bigger pad_size * ratio to be alloced padded memory,
 *  otherwise use tide allocation
 *  for example, if pad_ratio=2, GPU memory alignement size is 32,
 *  then we will only allocate padded memory if x dimension > 64
 *  set it to 0 then we will always allocate padded memory
 */
#ifndef MSHADOW_MIN_PAD_RATIO
  #define MSHADOW_MIN_PAD_RATIO 2
#endif

#if MSHADOW_STAND_ALONE
  #define MSHADOW_USE_CBLAS 0
  #define MSHADOW_USE_MKL   0
  #define MSHADOW_USE_CUDA  0
#endif

/*!
 * \brief force user to use GPU stream during computation
 *  error will be shot when default stream NULL is used
 */
#ifndef MSHADOW_FORCE_STREAM
#define MSHADOW_FORCE_STREAM 1
#endif

/*! \brief use CBLAS for CBLAS */
#ifndef MSHADOW_USE_CBLAS
  #define MSHADOW_USE_CBLAS 0
#endif
/*! \brief use MKL for BLAS */
#ifndef MSHADOW_USE_MKL
  #define MSHADOW_USE_MKL   0
#endif

/*!
 * \brief use CUDA support, must ensure that the cuda include path is correct,
 * or directly compile using nvcc
 */
#ifndef MSHADOW_USE_CUDA
  #define MSHADOW_USE_CUDA   0
#endif

/*!
 * \brief use CUDNN support, must ensure that the cudnn include path is correct
 */
#ifndef MSHADOW_USE_CUDNN
  #define MSHADOW_USE_CUDNN 0
#endif

/*!
 * \brief seems CUDAARCH is deprecated in future NVCC
 * set this to 1 if you want to use CUDA version smaller than 2.0
 */
#ifndef MSHADOW_OLD_CUDA
#define MSHADOW_OLD_CUDA 0
#endif

/*!
 * \brief macro to decide existence of c++11 compiler
 */
#ifndef MSHADOW_IN_CXX11
#define MSHADOW_IN_CXX11 (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
                          __cplusplus >= 201103L || defined(_MSC_VER))
#endif

/*! \brief whether use SSE */
#ifndef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 1
#endif
/*! \brief whether use NVML to get dynamic info */
#ifndef MSHADOW_USE_NVML
  #define MSHADOW_USE_NVML 0
#endif
// SSE is conflict with cudacc
#ifdef __CUDACC__
  #undef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 0
#endif

#if MSHADOW_USE_CBLAS
extern "C" {
}
#elif MSHADOW_USE_MKL
#endif

#if MSHADOW_USE_CUDA
#endif

#if MSHADOW_USE_CUDNN == 1
#endif

#if MSHADOW_USE_NVML
#endif

// --------------------------------
// MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code
#ifdef MSHADOW_XINLINE
  #error "MSHADOW_XINLINE must not be defined"
#endif
#ifdef _MSC_VER
#define MSHADOW_FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define MSHADOW_FORCE_INLINE inline __attribute__((always_inline))
#endif
#ifdef __CUDACC__
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE __device__ __host__
#else
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE
#endif
/*! \brief cpu force inline */
#define MSHADOW_CINLINE MSHADOW_FORCE_INLINE

#if defined(__GXX_EXPERIMENTAL_CXX0X) ||\
    defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
  #define MSHADOW_CONSTEXPR constexpr
#else
  #define MSHADOW_CONSTEXPR const
#endif

/*!
 * \brief default data type for tensor string
 *  in code release, change it to default_real_t
 *  during development, change it to empty string so that missing
 *  template arguments can be detected
 */
#ifndef MSHADOW_DEFAULT_DTYPE
#define MSHADOW_DEFAULT_DTYPE = default_real_t
#endif

/*!
 * \brief DMLC marco for logging
 */
#ifndef MSHADOW_USE_GLOG
#define MSHADOW_USE_GLOG DMLC_USE_GLOG
#endif  // MSHADOW_USE_GLOG

/*!
 * \brief Protected cuda call in mshadow
 * \param func Expression to call.
 * It checks for CUDA errors after invocation of the expression.
 */
#define MSHADOW_CUDA_CALL(func)                                    \
  {                                                                \
    cudaError_t e = (func);                                        \
    if (e == cudaErrorCudartUnloading) {                           \
      throw dmlc::Error(cudaGetErrorString(e));                    \
    }                                                              \
    CHECK(e == cudaSuccess)                                        \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Run function and catch error, log unknown error.
 * \param func Expression to call.
 */
#define MSHADOW_CATCH_ERROR(func)                                     \
  {                                                                   \
    try {                                                             \
      (func);                                                         \
    } catch (const dmlc::Error &e) {                                    \
      std::string what = e.what();                                      \
      if (what.find("driver shutting down") == std::string::npos) {     \
        LOG(ERROR) << "Ignore CUDA Error " << what;                     \
      }                                                                 \
    }                                                                   \
  }

//===== EXPANDIND: ../mshadow/mshadow/half.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file half.h
 * \brief definition of half (float16) type.
 *
 * \author Junyuan Xie
 */
#ifndef MSHADOW_HALF_H_
#define MSHADOW_HALF_H_

#if (MSHADOW_USE_CUDA && CUDA_VERSION >= 7050 && defined(__CUDA_ARCH__))
  #define MSHADOW_CUDA_HALF 1
#else
  #define MSHADOW_CUDA_HALF 0
#endif

/*! \brief namespace for mshadow */
namespace mshadow {
/* \brief name space for host/device portable half-precision floats */
namespace half {
#define MSHADOW_HALF_OPERATOR(RTYPE, OP)                                  \
  MSHADOW_XINLINE RTYPE operator OP (half_t a, half_t b) {                \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (half_t a, T b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (T a, half_t b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }

#define MSHADOW_HALF_ASSIGNOP(AOP, OP)                                    \
  template<typename T>                                                    \
  MSHADOW_XINLINE half_t operator AOP (T a) {                             \
    return *this = half_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }

#if MSHADOW_CUDA_HALF
#define MSHADOW_HALF_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(__half2float(cuhalf_));  /* NOLINT(*)*/                      \
  }
#else
#define MSHADOW_HALF_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(half2float(half_));  /* NOLINT(*)*/                          \
  }
#endif  // MSHADOW_CUDA_HALF

class half_t {
 public:
  static MSHADOW_XINLINE half_t Binary(uint16_t value) {
    half_t res;
    res.half_ = value;
    return res;
  }

  MSHADOW_XINLINE half_t() {}

#if MSHADOW_CUDA_HALF
  MSHADOW_XINLINE explicit half_t(const __half& value) {
    cuhalf_ = value;
  }
#endif  // MSHADOW_CUDA_HALF

  MSHADOW_XINLINE explicit half_t(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const int32_t& value) { constructor(value); }

  MSHADOW_HALF_CONVERSIONOP(float)
  MSHADOW_HALF_CONVERSIONOP(double)
  MSHADOW_HALF_CONVERSIONOP(uint8_t)
  MSHADOW_HALF_CONVERSIONOP(int32_t)

  MSHADOW_HALF_ASSIGNOP(+=, +)
  MSHADOW_HALF_ASSIGNOP(-=, -)
  MSHADOW_HALF_ASSIGNOP(*=, *)
  MSHADOW_HALF_ASSIGNOP(/=, /)

  MSHADOW_XINLINE half_t operator+() {
    return *this;
  }

  MSHADOW_XINLINE half_t operator-() {
    return half_t(-float(*this));  // NOLINT(*)
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const shift = 13;
  static int const shiftSign = 16;

  static int32_t const infN = 0x7F800000;  // flt32 infinity
  static int32_t const maxN = 0x477FE000;  // max flt16 normal as a flt32
  static int32_t const minN = 0x38800000;  // min flt16 normal as a flt32
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  union {
    uint16_t half_;
#if MSHADOW_CUDA_HALF
    __half cuhalf_;
#endif  // MSHADOW_CUDA_HALF
  };

  MSHADOW_XINLINE uint16_t float2half(const float value) const {
    Bits v, s;
    v.f = value;
    uint32_t sign = v.si & signN;
    v.si ^= sign;
    sign >>= shiftSign;  // logical shift
    s.si = mulN;
    s.si = s.f * v.f;  // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;  // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    return v.ui | sign;
  }

  MSHADOW_XINLINE float half2float(const uint16_t value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  MSHADOW_XINLINE void constructor(const T& value) {
#if MSHADOW_CUDA_HALF
    cuhalf_ = __float2half(float(value));  // NOLINT(*)
#else
    half_ = float2half(float(value));  // NOLINT(*)
#endif  // MSHADOW_CUDA_HALF
  }
};

/*! \brief overloaded + operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, +)
/*! \brief overloaded - operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, -)
/*! \brief overloaded * operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, *)
/*! \brief overloaded / operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, /)
/*! \brief overloaded > operator for half_t */
MSHADOW_HALF_OPERATOR(bool, >)
/*! \brief overloaded < operator for half_t */
MSHADOW_HALF_OPERATOR(bool, <)
/*! \brief overloaded >= operator for half_t */
MSHADOW_HALF_OPERATOR(bool, >=)
/*! \brief overloaded <= operator for half_t */
MSHADOW_HALF_OPERATOR(bool, <=)

#define MSHADOW_HALF_MIN mshadow::half::half_t::Binary(0x0400);
}  // namespace half
}  // namespace mshadow
#endif  // MSHADOW_HALF_H_

//===== EXPANDED: ../mshadow/mshadow/half.h =====


/*! \brief namespace for mshadow */
namespace mshadow {
/*! \brief buffer size for each random number generator */
const unsigned kRandBufferSize = 1000000;
/*! \brief pi  */
const float kPi = 3.1415926f;
/*! \brief type that will be used for index */
typedef unsigned index_t;
/*! \brief float point type that will be used in default by mshadow */
typedef float default_real_t;

/*! \brief data type flag */
enum TypeFlag {
  kFloat32,
  kFloat64,
  kFloat16,
  kUint8,
  kInt32
};

template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = kFloat32;
};
template<>
struct DataType<double> {
  static const int kFlag = kFloat64;
};
template<>
struct DataType<half::half_t> {
  static const int kFlag = kFloat16;
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = kUint8;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = kInt32;
};

/*! \brief type enum value for default real type */
const int default_type_flag = DataType<default_real_t>::kFlag;

/*! \brief namespace for operators */
namespace op {
// binary operator
/*! \brief mul operator */
struct mul{
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a * b;
  }
};
/*! \brief plus operator */
struct plus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a + b;
  }
};
/*! \brief minus operator */
struct minus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a - b;
  }
};
/*! \brief divide operator */
struct div {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a / b;
  }
};
/*! \brief get rhs */
struct right {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return b;
  }
};
// unary operator/ function: example
// these operators can be defined by user,
// in the same style as binary and unary operator
// to use, simply write F<op::identity>( src )
/*! \brief identity function that maps a real number to it self */
struct identity{
  /*! \brief map a to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a;
  }
};
}  // namespace op
/*! \brief namespace for savers */
namespace sv {
/*! \brief save to saver: = */
struct saveto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a = b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 0.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::right OPType;
};
/*! \brief save to saver: += */
struct plusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a += b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::plus OPType;
};
/*! \brief minus to saver: -= */
struct minusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a -= b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return -1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::minus OPType;
};
/*! \brief multiply to saver: *= */
struct multo {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a *= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::mul OPType;
};
/*! \brief divide to saver: /= */
struct divto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType& a, DType b) { // NOLINT(*)
    a /= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::div OPType;
};
}  // namespace sv
/*! \brief namespace for potential reducer operations */
namespace red {
namespace limits {
/*!
 * \brief minimum value of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType MinValue(void);
/*! \brief minimum value of float */
template<>
MSHADOW_XINLINE float MinValue<float>(void) {
  return -FLT_MAX;
}
/*! \brief minimum value of double */
template<>
MSHADOW_XINLINE double MinValue<double>(void) {
  return -DBL_MAX;
}
/*! \brief minimum value of half */
template<>
MSHADOW_XINLINE half::half_t MinValue<half::half_t>(void) {
  return MSHADOW_HALF_MIN;
}
/*! \brief minimum value of int */
template<>
MSHADOW_XINLINE int MinValue<int>(void) {
  return INT_MIN;
}
}  // namespace limits

/*! \brief sum reducer */
struct sum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    dst += src;
  }
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 0;
  }
};
/*! \brief maximum reducer */
struct maximum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    using namespace std;
    dst = max(dst, src);
  }
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = limits::MinValue<DType>();
  }
};
/*! \brief minimum reducer */
struct minimum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    using namespace std;
    dst = min(dst, src);
  }
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = -limits::MinValue<DType>();
  }
};
}  // namespace red
}  // namespace mshadow
#endif  // MSHADOW_BASE_H_
//===== EXPANDED: ../mshadow/mshadow/base.h =====

//===== EXPANDIND: ../mshadow/mshadow/expression.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file expression.h
 * \brief definitions of abstract expressions and expressions template
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXPRESSION_H_
#define MSHADOW_EXPRESSION_H_

namespace mshadow {
/*!
 * \brief namespace for abstract expressions and expressions template,
 *        have no dependecy on tensor.h,
 *        These data structure takes no charge in computations,
 *        they are only used to define operations and represent expression in a symbolic way
 */
namespace expr {
/*! \brief type of expressions */
namespace type {
// type expression type are defined as bitmask
// subtype relationshop kRValue < kMapper < kPull < kComplex
/*! 
 * \brief this expression directly correspnds to a data class,
 *   can be used to assign data 
 */
const int kRValue = 0;
/*! 
 * \brief expression contains element-wise tensor operations,
 *   map a expression to same shape 
 */
const int kMapper = 1;
/*!
 * \brief expression that can be chained with other expressiones
 *    Usually it have function Eval(i,j) defined, which pulls the result (i, j) from input
 *    expression and output the result at certain position.
 */
const int kChainer = 3;
/*! \brief othercase: e.g dot product */
const int kComplex = 7;
}  // namespace type
/*!
 * \brief expression engine that actually interprets these expressions
 *   this is a function template that needed to be implemented for specific expressions
 * \tparam Saver the save method
 * \tparam RValue the type of RValue to be saved
 * \sa namespace sv
 */
template<typename Saver, typename RValue, typename DType>
struct ExpEngine;
/*! \brief defines how expression exp can be evaluated and stored into dst */
// template<typename EType>
// inline static void Eval(RValue *dst, const EType &exp);
/*!
 * \brief base class for expression
 * \tparam SubType inheritated class must put their type into this parameter
 * \tparam DType the data type of each element in the expression
 * \tparam exp_type expression type, see namespace type
 */
template<typename SubType, typename DType, int exp_type>
struct Exp {
 public:
  /*! \return  subtype instance of current class */
  inline const SubType& self(void) const {
    return *static_cast<const SubType*>(this);
  }
  /*! \return reference of subtype instance of current class */
  inline SubType* ptrself(void) {
    return static_cast<SubType*>(this);
  }
};
/*!
 * \brief scalar expression 
 * \tparam DType the data type of the scalar
 */
template<typename DType>
struct ScalarExp: public Exp<ScalarExp<DType>, DType, type::kMapper> {
  /*! \brief scalar value */
  DType scalar_;
  /*! \brief implicit constructor, MUST NOT BE explicit */
  ScalarExp(DType scalar) : scalar_(scalar) {}  // NOLINT(*)
};
/*! \brief create an scalar expression */
template<typename DType>
inline ScalarExp<DType> scalar(DType s) {
  return ScalarExp<DType>(s);
}
/*!
 * \brief typecast expression, cast the type of elements
 * \tparam DstDType the target type we want to cast into
 * \tparam SrcDType the target type we want to cast from
 * \tparam EType the type of the source expression
 * \tparam etype the type of expression after cast
 */
template<typename DstDType, typename SrcDType, typename EType, int etype>
struct TypecastExp:
      public Exp<TypecastExp<DstDType, SrcDType, EType, etype>,
                 DstDType, etype> {
  /*! \brief expression to be typecasted */
  const EType &exp;
  /*! \brief constructor */
  explicit TypecastExp(const EType &e) : exp(e) {}
};
/*! \brief create an scalar expression */
template<typename DstDType, typename SrcDType,
         typename EType, int etype>
inline TypecastExp<DstDType, SrcDType, EType, (etype|type::kMapper)>
tcast(const Exp<EType, SrcDType, etype> &exp) {
  return TypecastExp<DstDType, SrcDType, EType,
                     (etype|type::kMapper)>(exp.self());
}
/*! \brief represent a transpose expression of a container */
template<typename EType, typename DType>
struct TransposeExp: public Exp<TransposeExp<EType, DType>,
                                DType, type::kChainer> {
  /*! \brief expression to be transposed */
  const EType &exp;
  /*! \brief constructor */
  explicit TransposeExp(const EType &e) : exp(e) {}
  /*! \brief transpose expression */
  inline const EType &T(void) const {
    return exp;
  }
};
/*!
 * \brief base class of all rvalues
 * \tparam Container the actually class of data container, e.g. Tensor1D
 * \tparam DataType the element data type of each element in the container
 */
template<typename Container, typename DType>
class RValueExp: public Exp<Container, DType, type::kRValue> {
 public:
  /*!
   *\brief transpose of a matrix
   *\return transpose of current expression
   */
  inline const TransposeExp<Container, DType> T(void) const {
    return TransposeExp<Container, DType>(this->self());
  }
  /*! \brief operator overload */
  inline Container &operator+=(DType s) {
    ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator-=(DType s) {
    ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator*=(DType s) {
    ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &operator/=(DType s) {
    ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief operator overload */
  inline Container &__assign(DType s) {
    ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
    return *(this->ptrself());
  }
  /*! \brief  we can not define container = container */
  template<typename E, int etype>
  inline Container &__assign(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief operator overload, assign */
  inline Container &__assign(const Exp<Container, DType, type::kRValue> &exp);
  /*! \brief implementation of operator+= */
  template<typename E, int etype>
  inline Container &operator+=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator-= */
  template<typename E, int etype>
  inline Container &operator-=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator*= */
  template<typename E, int etype>
  inline Container &operator*=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
  /*! \brief implementation of operator/= */
  template<typename E, int etype>
  inline Container &operator/=(const Exp<E, DType, etype> &exp) {
    ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), exp.self());
    return *(this->ptrself());
  }
};
/*!
 * \brief matrix multiplication expression dot(lhs[.T], rhs[.T])
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam ltrans whether lhs is transposed
 * \tparam rtrans whether rhs is transposed
 * \tparam DType the data type of the scalar
 */
template<typename TA, typename TB, bool ltrans, bool rtrans, typename DType>
struct DotExp: public Exp<DotExp<TA, TB, ltrans, rtrans, DType>,
                          DType, type::kComplex> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief scale over result */
  DType scale_;
  /*! \brief constructor */
  explicit DotExp(const TA &lhs, const TB &rhs, DType scale)
      : lhs_(lhs), rhs_(rhs), scale_(scale) {}
};
// definition of dot expression
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, false, DType>
dot(const RValueExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, false, false, DType>(lhs.self(), rhs.self(), DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, true, false, DType>
dot(const TransposeExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, true, false, DType>(lhs.exp, rhs.self(), DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, true, DType>
dot(const RValueExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
  return DotExp<TA, TB, false, true, DType>(lhs.self(), rhs.exp, DType(1.0f));
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, true, true, DType>
dot(const TransposeExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
  return DotExp<TA, TB, true, true, DType>(lhs.exp, rhs.exp, DType(1.0f));
}
//---------------
// BinaryMapExp
// --------------
/*!
 * \brief binary map expression lhs [op] rhs
 * \tparam OP operator
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, TA, TB, DType, etype>,
                                DType, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit BinaryMapExp(const TA &lhs, const TB &rhs)
      :lhs_(lhs), rhs_(rhs) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, DType, (ta|tb|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return BinaryMapExp<OP, TA, TB, DType,
                      (ta|tb|type::kMapper)>(lhs.self(), rhs.self());
}
/*!
 * \brief short hand for MakeExp, usage F<op>(lhs, rhs). create a binary operation expression 
 * \param lhs left operand
 * \param rhs right operand
 * \return the result expression
 * \tparam binary operator 
 * \tparam TA lhs expression
 * \tparam ta lhs expression type
 * \tparam TB rhs expression
 * \tparam tb rhs expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, DType, (ta|tb|type::kMapper)>
F(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
// operator rules
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::plus, TA, TB, DType, (ta|tb|type::kMapper)>
operator+(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::minus, TA, TB, DType, (ta|tb|type::kMapper)>
operator-(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::mul, TA, TB, DType, (ta|tb|type::kMapper)>
operator*(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<op::div, TA, TB, DType, (ta|tb|type::kMapper)>
operator/(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
//---------------
// UnaryMapExp
// --------------
/*!
 * \brief unary map expression op(src)
 * \tparam OP operator
 * \tparam TA type of src
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename DType, int etype>
struct UnaryMapExp: public Exp<UnaryMapExp<OP, TA, DType, etype>,
                               DType, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit UnaryMapExp(const TA &src) : src_(src) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename DType, int ta>
inline UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &src) {
  return UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>(src.self());
}
/*! 
 * \brief short hand for MakeExp, usage F<op>(src), create a unary operation expression 
 * \param src source expression
 * \return the result expression
 * \tparam operator 
 * \tparam TA source expression
 * \tparam ta source expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, typename DType, int ta>
inline UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>
F(const Exp<TA, DType, ta> &src) {
  return MakeExp<OP>(src);
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPRESSION_H_
//===== EXPANDED: ../mshadow/mshadow/expression.h =====


namespace mshadow {
/*! \brief device name CPU */
struct cpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = true;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 0;
};
/*! \brief device name CPU */
struct gpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = false;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 1;
};
template<int ndim>
struct Shape;

/*!
 * \brief allow string printing of the shape
 * \param os the output stream
 * \param shape the shape
 * \return the ostream
 */
template<int ndim>
inline std::ostream &operator<<(std::ostream &os, const Shape<ndim> &shape); // NOLINT(*)

/*!
 * \brief shape of a tensor
 *       IMPORTANT NOTE: this shape is different from numpy.shape
 *       shape[0] gives the lowest dimension, shape[dimension-1] gives the highest dimension
 *       shape[k] corresponds to k-th dimension of tensor
 * \tparam dimension dimension of tensor
 */
template<int dimension>
struct Shape {
  /*! \brief dimension of current shape */
  static const int kDimension = dimension;
  /*! \brief dimension of current shape minus one */
  static const int kSubdim = dimension - 1;
  /*! \brief storing the dimension information */
  index_t shape_[kDimension];
  /*! \brief default constructor, do nothing */
  MSHADOW_XINLINE Shape(void) {}
  /*! \brief constuctor */
  MSHADOW_XINLINE Shape(const Shape<kDimension> &s) {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      this->shape_[i] = s[i];
    }
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE index_t &operator[](index_t idx) {
    return shape_[idx];
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE const index_t &operator[](index_t idx) const {
    return shape_[idx];
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   */
  MSHADOW_XINLINE bool operator==(const Shape<kDimension> &s) const {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      if (s.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equal
   * \param s the shape to compare against
   */
  MSHADOW_XINLINE bool operator!=(const Shape<kDimension> &s) const {
    return !(*this == s);
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  MSHADOW_XINLINE Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    s.shape_[1] = this->shape_[kDimension - 1];
    index_t ymax = 1;
    #pragma unroll
    for (int i = 0; i < kDimension - 1; ++i) {
      ymax *= this->shape_[i];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*! \return number of valid elements */
  MSHADOW_XINLINE size_t Size(void) const {
    size_t size = this->shape_[0];
    #pragma unroll
    for (int i = 1; i < kDimension; ++i) {
      size *= this->shape_[i];
    }
    return size;
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  MSHADOW_XINLINE index_t ProdShape(int dimstart, int dimend) const {
    index_t num = 1;
    #pragma unroll
    for (int i = dimstart; i < dimend; ++i) {
      num *= this->shape_[i];
    }
    return num;
  }
  /*!
   * \brief get subshape that takes off largest dimension
v   * \return subshape
   */
  MSHADOW_XINLINE Shape<kSubdim> SubShape(void) const {
    Shape<kSubdim> s;
    // for cuda
    #pragma unroll
    for (int i = 0; i < kSubdim; ++i) {
      s.shape_[i] = this->shape_[i + 1];
    }
    return s;
  }
  /*!
   * \brief slice the shape from start to end
   * \tparam dimstart start dimension
   * \tparam dimend end dimension
   * \return the sliced shape
   */
  template<int dimstart, int dimend>
  MSHADOW_XINLINE Shape<dimend - dimstart> Slice(void) const {
    Shape<dimend - dimstart> s;
    #pragma unroll
    for (int i = dimstart; i < dimend; ++i) {
      s[i - dimstart] = this->shape_[i];
    }
    return s;
  }
  //! \cond Doxygen_Suppress
  template<int dim>
  friend std::ostream &operator<<(std::ostream &os, const Shape<dim> &shape); // NOLINT(*)
  //! \endcond
};  // Shape
//------------------------------------------------
// useful construction functions to generate shape
//-------------------------------------------------
/*!
 * \brief construct a one dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<1> Shape1(index_t s0) {
  Shape<1> s; s[0] = s0;
  return s;
}
/*!
 * \brief construct a two dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
  Shape<2> s; s[0] = s0; s[1] = s1;
  return s;
}
/*!
 * \brief construct a three dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \param s2 size of dimension 2
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<3> Shape3(index_t s0, index_t s1, index_t s2) {
  Shape<3> s;
  s[0] = s0; s[1] = s1; s[2] = s2;
  return s;
}
/*!
 * \brief construct a four dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \param s2 size of dimension 2
 * \param s3 size of dimension 3
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<4> Shape4(index_t s0, index_t s1,
                                index_t s2, index_t s3) {
  Shape<4> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
  return s;
}
/*!
* \brief construct a five dimension shape, stride will equal s0
* \param s0 size of dimension 0
* \param s1 size of dimension 1
* \param s2 size of dimension 2
* \param s3 size of dimension 3
* \param s4 size of dimension 4
* \return the shape construction
*/
MSHADOW_XINLINE Shape<5> Shape5(index_t s0, index_t s1, index_t s2,
                                index_t s3, index_t s4) {
  Shape<5> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; s[4] = s4;
  return s;
}
/*!
 * \brief computaion stream structure, used for asynchronize computation
 */
template<typename Device>
struct Stream {
  // this is only a dummy implementation for CPU
  // for GPU, the actual implementation will be specialized in tensor_gpu-inl.h
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {}
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    return true;
  }
  /*! \brief create a blas handle */
  inline void CreateBlasHandle() {}
};
/*!
 * \brief Tensor RValue, this is the super type of all kinds of possible tensors
 * \tparam Container the tensor type
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Container, typename Device, int dimension, typename DType>
struct TRValue: public expr::RValueExp<Container, DType> {
};
// more compact template
/*!
 * \brief general tensor
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Device, int dimension,
         typename DType MSHADOW_DEFAULT_DTYPE>
struct Tensor: public TRValue<Tensor<Device, dimension, DType>,
                              Device, dimension, DType> {
 public:
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief whether current type lies in cpu */
  static const bool kDevCPU = Device::kDevCPU;
  /*! \brief dimension of subtype */
  static const int  kSubdim = dimension - 1;
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief pointer to the data */
  DType *dptr_;
  /*! \brief shape of the tensor */
  Shape<dimension> shape_;
  /*!
   * \brief storing the stride information in x dimension
   *    this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
   */
  index_t stride_;
  /*!
   * \brief stream where the computation lies
   * stream is a device dependency concept where each computation
   */
  Stream<Device> *stream_;
  //--------------------------------
  // functions
  //--------------------------------
  /*! \brief default constructor */
  MSHADOW_XINLINE Tensor(void) : stream_(NULL) {}
  /*! \brief constructor from shape  */
  MSHADOW_XINLINE Tensor(const Shape<dimension> &shape)
      : shape_(shape), stream_(NULL) {}
  /*! \brief constructor from data pointer and shape, without stride */
  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape)
      : dptr_(dptr), shape_(shape), stride_(shape[kSubdim]), stream_(NULL) {}
  /*! \brief constructor from data pointer and shape, without stride */
  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape,
                         Stream<Device> *stream)
    : dptr_(dptr), shape_(shape), stride_(shape[kSubdim]), stream_(stream) {}
  /*! \brief constructor from data pointer and shape  */
  MSHADOW_XINLINE Tensor(DType *dptr,
                         const Shape<dimension> &shape,
                         index_t stride, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}
  /*!
   * \brief set the stream to do computation of current tensor
   * \param stream the computation stream
   */
  inline void set_stream(Stream<Device> *stream) {
    this->stream_ = stream;
  }
  /*!
   * \return memory cost of the tensor, including the aligned x dimension
   * \tparam startdim the starting dimension
   */
  template<int startdim>
  MSHADOW_XINLINE size_t MemSize(void) const {
    size_t memsz = this->stride_;
    #pragma unroll
    for (int i = startdim; i < kSubdim; ++i) {
      memsz *= this->shape_[i];
    }
    return memsz;
  }
  /*!
   * \return whether the tensor's memory is continuous
   * x dimension same as stride
   */
  MSHADOW_XINLINE bool CheckContiguous(void) const {
    return this->shape_[dimension - 1] == stride_;
  }
  /*!
   * \return memory cost of the tensor, including the aligned x dimension
   */
  MSHADOW_XINLINE size_t MSize(void) const {
    return this->MemSize<0>();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param idx the dimension count from the highest dimensin
   * \return the size
   */
  MSHADOW_XINLINE index_t size(index_t idx) const {
    return shape_[idx];
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \return tensor after flatten
   */
  MSHADOW_XINLINE Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }
  /*!
   * \brief get a element of dimension - 1
   * \param idx index
   * \return the result tensor
   */
  MSHADOW_XINLINE Tensor<Device, kSubdim, DType> operator[](index_t idx) const {
    return Tensor<Device, kSubdim, DType>(dptr_ + this->MemSize<1>() * idx,
                                          shape_.SubShape(), stride_, stream_);
  }
  /*!
   * \brief slice the tensor in highest dimension [begin,end)
   * \param begin begin position of slice
   * \param end end position of slice
   * \return tensor after slice
   */
  MSHADOW_XINLINE Tensor<Device, dimension, DType>
  Slice(index_t begin, index_t end) const {
    Shape<dimension> s = this->shape_;
    s[0] = end - begin;
    return Tensor<Device, dimension, DType>(dptr_ + this->MemSize<1>() * begin,
                                            s, stride_, stream_);
  }
  /*!\brief implement the assignment of same type */
  inline Tensor<Device, dimension, DType> &
  operator=(const Tensor<Device, dimension, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }
  /*!\brief functions to fit expression template */
  template<typename E, int etype>
  inline Tensor<Device, dimension, DType> &
  operator=(const expr::Exp<E, DType, etype> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  inline Tensor<Device, dimension, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};
/*
 *  respecialized class Tensor1D, thei is due to different implementation in operator[]
 */
template<typename Device, typename DType>
struct Tensor<Device, 1, DType>:
      public TRValue<Tensor<Device, 1, DType>, Device, 1, DType> {
 public:
  DType *dptr_;
  Shape<1> shape_;
  index_t stride_;
  Stream<Device> *stream_;
  // constructor
  MSHADOW_XINLINE Tensor(void) : stream_(NULL) {}
  MSHADOW_XINLINE Tensor(const Shape<1> &shape)
      : shape_(shape), stream_(NULL) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape)
      : dptr_(dptr), shape_(shape), stride_(shape[0]), stream_(NULL) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape,
                         index_t stride, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}
  inline void set_stream(Stream<Device> *stream) {
    this->stream_ = stream;
  }
  MSHADOW_XINLINE Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }
  MSHADOW_XINLINE Tensor<Device, 1, DType> Slice(index_t begin, index_t end) const {
    Shape<1> s;
    s[0] = end  - begin;
    return Tensor<Device, 1, DType>(dptr_ + begin, s, s[0], stream_);
  }
  MSHADOW_XINLINE bool CheckContiguous(void) const {
    return true;
  }
  MSHADOW_XINLINE size_t MSize(void) const {
    return shape_[0];
  }
  MSHADOW_XINLINE index_t size(index_t i) const {
    return shape_[0];
  }
  MSHADOW_XINLINE DType &operator[](index_t idx) {
    return dptr_[idx];
  }
  MSHADOW_XINLINE const DType &operator[](index_t idx) const {
    return dptr_[idx];
  }
  /*!\brief implement the assignment of same type */
  inline Tensor<Device, 1, DType> &
  operator=(const Tensor<Device, 1, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }
  template<typename E, int etype>
  inline Tensor<Device, 1, DType> &
  operator=(const expr::Exp<E, DType, etype> &exp) {
    return this->__assign(exp);
  }
  inline Tensor<Device, 1, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};
//------------------------
// Function Declarations
//-----------------------
/*!
 * \brief initialize tensor engine, used to call intialization functions of dependent libs
 *        this function should be called before all GPU tensor operations,
 *        for using tensors in CPU, this call is actually not needed
 * \param device_id GPU device id to be choosed
 * \tparam Device the device type
 */
template<typename Device>
inline void InitTensorEngine(int device_id = 0);
/*!
 * \brief Shutdown tensor engine on current device
 *     this function should be called after all GPU tensor operations,
 *     for using tensors in CPU, this call is actually not needed
 * \tparam Device the device type
 */
template<typename Device>
inline void ShutdownTensorEngine(void);
/*!
 * \brief set the device of current thread to work on
 * \param devid the device id
 * \tparam Device the device type
 */
template<typename Device>
inline void SetDevice(int devid);
/*!
 * \brief create a new stream from system
 * \param create_blas_handle whether create blas handle in stream
 * \param create_dnn_handle whether create cudnn handle in stream
 * \return a pointer to the created stream
 * \tparam Device the device type
 */
template<typename Device>
inline Stream<Device> *NewStream(bool create_blas_handle,
                                 bool create_dnn_handle);
/*! \brief default behavior: create cublas handle */
template<typename Device>
inline Stream<Device> *NewStream() {
  return NewStream<Device>(true, false);
}
/*!
 * \brief delete the computing stream
 * \param stream the stream parameter to be deleted
 */
template<typename Device>
inline void DeleteStream(Stream<Device> *stream);
/*!
 * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
 *        this function is responsible to set the stride_ in each obj.shape
 * \param obj the tensor object, with shape specified
 * \param pad whether padding dimension 0, to make last dimension aligned,
 *            padding may help improve efficiency of matrix multiplications
 *            if true, will allocate space with stride_ that may not equals shape[0]
 *            if false, will allocate continuous space
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void AllocSpace(Tensor<cpu, dim, DType> *obj,
                       bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
 *        this function is responsible to set the stride_ in each obj.shape
 * \param obj the tensor object, with shape specified
 * \param pad whether padding dimension 0, to make last dimension aligned,
 *            padding may help improve efficiency of matrix multiplications
 *            if true, will allocate space with stride_ that may not equals shape[0]
 *            if false, will allocate continuous space
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void AllocSpace(Tensor<gpu, dim, DType> *obj,
                       bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
 * \param obj the tensor object
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> *obj);
/*!
 * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
 * \param obj the tensor object
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> *obj);
/*!
 * \brief CPU/GPU: short cut to allocate and initialize a Tensor
 * \param shape: shape of tensor
 * \param initv: initialization value
 * \param pad : padding option
 * \param stream : stream of tensor
 * \tparam Device device of tensor
 * \tparam DType type of element in tensor
 * \tparam dim dimention of tensor
 * \return a new allocated tensor
 * \sa AllocSpace
 */
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType> NewTensor(const Shape<dim> &shape,
                                            DType initv,
                                            bool pad = MSHADOW_ALLOC_PAD,
                                            Stream<Device> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<cpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief CPU/GPU: normalize softmax: dst[i][j] = exp(energy[i][j]) /(sum_j exp(energy[i][j]))
 * \param dst destination
 * \param energy input energy
 */
template<typename DType>
inline void Softmax(Tensor<cpu, 2, DType> dst, const Tensor<cpu, 2, DType> &energy);
/*!
 * \brief CPU/GPU: normalize softmax: dst[i][j] = exp(energy[i][j]) /(sum_j exp(energy[i][j]))
 * \param dst destination
 * \param energy input energy
 */
template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> dst, const Tensor<gpu, 2, DType> &energy);

/*!
 * \brief CPU/GPU: softmax gradient
 * \param dst destination
 * \param src source output
 * \param label label info
 */
template<typename DType>
inline void SoftmaxGrad(Tensor<cpu, 2, DType> dst,
                        const Tensor<cpu, 2, DType> &src,
                        const Tensor<cpu, 1, DType> &label);
/*!
 * \brief CPU/GPU: softmax gradient
 * \param dst destination
 * \param src source output
 * \param label label info
 */
template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label);
// function declarations to support expression, no need to understand them
// these functions do not need to be directly used
/*!
 * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
 * \tparam Saver specify storage method
 * \tparam R specifies the storage type of the tensor
 * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
 */
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, cpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp);
/*!
 * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
 * \tparam Saver specify storage method
 * \tparam R specifies the storage type of the tensor
 * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
 */
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, gpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, cpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, typename R,
         typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, gpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, cpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, gpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale = 1);

/*!
 * \brief CPU/GPU: 1 dimension vector dot
 * \param dst Length 1 vector, used to hold the result.
 * \param lhs Left operand vector
 * \param rhs right operand vector
 */
template<typename Device, typename DType>
inline void VectorDot(Tensor<Device, 1, DType> dst,
                      const Tensor<Device, 1, DType> &lhs,
                      const Tensor<Device, 1, DType> &rhs);
}  // namespace mshadow
// include headers
//===== EXPANDIND: ../mshadow/mshadow/stream_gpu-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file stream_gpu-inl.h
 * \brief implementation of GPU code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_STREAM_GPU_INL_H_
#define MSHADOW_STREAM_GPU_INL_H_
//===== EXPANDIND: ../mshadow/mshadow/logging.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of dmlc
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef MSHADOW_LOGGING_H_
#define MSHADOW_LOGGING_H_
#ifndef DMLC_LOGGING_H_
#define DMLC_LOGGING_H_


namespace dmlc {
/*! \brief taken from DMLC directly */

/*!
 * \brief exception class that will be thrown by
 *  default logger if DMLC_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept(a)
#endif

#if DMLC_USE_CXX11
#define DMLC_THROW_EXCEPTION noexcept(false)
#else
#define DMLC_THROW_EXCEPTION
#endif

#if DMLC_USE_GLOG

namespace dmlc {
/*! \brief taken from DMLC directly */
inline void InitLogging(const char* argv0) {
  google::InitGoogleLogging(argv0);
}
}  // namespace dmlc

#else
// use a light version of glog

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#endif

namespace dmlc {
inline void InitLogging(const char* argv0) {
  // DO NOTHING
}

// Always-on checking
#define CHECK(x)                                           \
  if (!(x))                                                \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check "  \
      "failed: " #x << ' '
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x) \
  ((x) == NULL ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)
// Debug-only checking.
#ifdef NDEBUG
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

#define LOG_INFO dmlc::LogMessage(__FILE__, __LINE__)
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LOG_INFO
#define LOG_FATAL dmlc::LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

// Poor man version of VLOG
#define VLOG(x) LOG_INFO.stream()

#define LOG(severity) LOG_##severity.stream()
#define LG LOG_INFO.stream()
#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)

#ifdef NDEBUG
#define LOG_DFATAL LOG_ERROR
#define DFATAL ERROR
#define DLOG(severity) true ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#else
#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)
#endif

// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm now;
    localtime_r(&time_value, &now);
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", now.tm_hour,
             now.tm_min, now.tm_sec);
#endif
    return buffer_;
  }
 private:
  char buffer_[9];
};

class LogMessage {
 public:
  LogMessage(const char* file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() { log_stream_ << "\n"; }
  std::ostream& stream() { return log_stream_; }

 protected:
  std::ostream& log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};

#if DMLC_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n";
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() DMLC_THROW_EXCEPTION {
    // throwing out of destructor is evil
    // hopefully we can do it here
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

}  // namespace dmlc

#endif
#endif  // DMLC_LOGGING_H_
#endif  // MSHADOW_LOGGING_H_

//===== EXPANDED: ../mshadow/mshadow/logging.h =====


namespace mshadow {
#if MSHADOW_USE_CUDA == 1
// Stream alocation
// actual implementation of GPU stream in CUDA
template<>
struct Stream<gpu> {
  /*! \brief handle state */
  enum HandleState {
    NoHandle = 0,
    OwnHandle = 1,
  };
  /*! \brief cudaStream */
  cudaStream_t stream_;
  /*! \brief cublas handle */
  cublasHandle_t blas_handle_;
  /*! \brief cudnn handle */
  #if MSHADOW_USE_CUDNN == 1
  cudnnHandle_t dnn_handle_;
  #endif
  /*! \brief cublas handle ownership */
  HandleState blas_handle_ownership_;
  /*! \brief cudnn handle ownership */
  HandleState dnn_handle_ownership_;

  Stream(void) : stream_(0),
                 blas_handle_ownership_(NoHandle),
                 dnn_handle_ownership_(NoHandle) {}
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(stream_));
  }
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    LOG(FATAL) << cudaGetErrorString(err);
    return false;
  }
  /*!
   * \brief returns actual cudaStream_t given an input GPU stream pointer
   * \param stream pointer to GPU stream
   */
  inline static cudaStream_t GetStream(Stream<gpu> *stream) {
    if (stream == NULL) {
#if MSHADOW_FORCE_STREAM
      LOG(FATAL) << "Default GPU stream was used when MSHADOW_FORCE_STREAM was on";
#endif
      return 0;
    } else {
      return stream->stream_;
    }
  }
  /*!
   * \brief return actual cublasHandle
   * \param pointer to GPU stream
   */
  inline static cublasHandle_t GetBlasHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->blas_handle_ownership_, NoHandle)
        << "No handle exist in source stream";
      return stream->blas_handle_;
    }
  }
  /*! \brief Destory cublas handle if own it */
  inline void DestoryBlasHandle() {
    if (blas_handle_ownership_ == OwnHandle) {
      cublasStatus_t err = cublasDestroy(blas_handle_);
      blas_handle_ownership_ = NoHandle;
      CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Destory cublas handle failed";
    }
  }
  /*! \brief Destory original blas handle and create a new one */
  inline void CreateBlasHandle() {
    this->DestoryBlasHandle();
    cublasStatus_t err = cublasCreate(&blas_handle_);
    blas_handle_ownership_ = OwnHandle;
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Create cublas handle failed";
  }
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
  inline static cudnnHandle_t GetDnnHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->dnn_handle_ownership_, NoHandle) << "No handle exist in source stream";
      return stream->dnn_handle_;
    }
  }
#endif
  inline void DestroyDnnHandle() {
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    if (dnn_handle_ownership_ == OwnHandle) {
      cudnnStatus_t err = cudnnDestroy(dnn_handle_);
      CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    }
#endif
  }
  inline void CreateDnnHandle() {
// #if MSHADOW_USE_CUDNN == 1 && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    this->DestroyDnnHandle();
    cudnnStatus_t err = cudnnCreate(&dnn_handle_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    err = cudnnSetStream(dnn_handle_, stream_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    this->dnn_handle_ownership_ = OwnHandle;
#endif
  }
};
template<>
inline Stream<gpu> *NewStream<gpu>(bool create_blas_handle,
                                   bool create_dnn_handle) {
  Stream<gpu> *st = new Stream<gpu>();
  MSHADOW_CUDA_CALL(cudaStreamCreate(&st->stream_));
  if (create_blas_handle) {
    st->CreateBlasHandle();
  }
  if (create_dnn_handle) {
    st->CreateDnnHandle();
  }
  return st;
}
template<>
inline void DeleteStream<gpu>(Stream<gpu> *stream) {
  MSHADOW_CUDA_CALL(cudaStreamDestroy(stream->stream_));
  stream->DestoryBlasHandle();
  stream->DestroyDnnHandle();
  delete stream;
}
#endif
}  // namespace mshadow
#endif  // MSHADOW_STREAM_GPU_INL_H_
//===== EXPANDED: ../mshadow/mshadow/stream_gpu-inl.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension.h =====

/*!
 * Copyright by Contributors
 * \file extension.h
 * \brief some extension of expressions,
 *  used to support something beyond elementwise op
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXTENSION_H_
#define MSHADOW_EXTENSION_H_
//===== EXPANDIND: ../mshadow/mshadow/expr_engine-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file expr_engine-inl.h
 * \brief definitions of how expressions should be evaluated
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXPR_ENGINE_INL_H_
#define MSHADOW_EXPR_ENGINE_INL_H_

namespace mshadow {
namespace expr {
/*!
 * \brief a general class that allows extension that makes tensors of some shape
 * \tparam SubType type of subclass
 * \tparam SrcExp source expression of the MakeTensorExp, the source of operation
 * \tparam dim dimension of the expression
 * \tparam DType the type of elements
 */
template<typename SubType, typename SrcExp, int dim, typename DType>
struct MakeTensorExp
    : public Exp<MakeTensorExp<SubType, SrcExp, dim, DType>,
                 DType, type::kChainer> {
  /*! \brief the shape of this expression */
  Shape<dim> shape_;
  /*! \brief true self of subtype */
  inline const SubType& real_self(void) const{
    return *static_cast<const SubType*>(this);
  }
};
//----------------------------------------------------------------------
// This part of code gives plan that can be used to carry out execution
//---------------------------------------------------------------------
// Declarations of plans
template<typename ExpType, typename DType>
class Plan {
 public:
  /*!
   * \brief evaluate the expression at index [y][x]
   *  to be implemented by SubType, for RValue, the return type will be DType &
   */
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const;
};
// tensor plan
template <typename Device, int dim, typename DType>
class Plan<Tensor<Device, dim, DType>, DType> {
 public:
  explicit Plan(const Tensor<Device, dim, DType> &t)
      : dptr_(t.dptr_), stride_(t.stride_) {}
  // for RValue, the return type should be reference
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return dptr_[y * stride_ + x];
  }
  // const evaluation
  MSHADOW_XINLINE const DType &Eval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

 private:
  DType  *dptr_;
  index_t stride_;
};
// special evaluation case for 1d tensor, no stride
template <typename Device, typename DType>
class Plan<Tensor<Device, 1, DType>, DType> {
 public:
  explicit Plan(const Tensor<Device, 1, DType> &t) : dptr_(t.dptr_) {}
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return dptr_[x];
  }
  MSHADOW_XINLINE const DType &Eval(index_t y, index_t x) const {
    return dptr_[x];
  }

 private:
  DType  *dptr_;
};
// scalar
template<typename DType>
class Plan<ScalarExp<DType>, DType> {
 public:
  explicit Plan(DType scalar) : scalar_(scalar) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return scalar_;
  }

 private:
  DType scalar_;
};
// unary expression
template<typename DstDType, typename SrcDType,
         typename EType, int etype>
class Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType> {
 public:
  explicit Plan(const Plan<EType, SrcDType> &src) : src_(src) {}
  MSHADOW_XINLINE DstDType Eval(index_t y, index_t x) const {
    return DstDType(src_.Eval(y, x));  // NOLINT(*)
  }

 private:
  Plan<EType, SrcDType> src_;
};
// binary expression
template<typename OP, typename TA, typename TB, int etype, typename DType>
class Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType> {
 public:
  explicit Plan(const Plan<TA, DType> &lhs, const Plan<TB, DType> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(lhs_.Eval(y, x), rhs_.Eval(y, x));
  }

 private:
  Plan<TA, DType> lhs_;
  Plan<TB, DType> rhs_;
};
// unary expression
template<typename OP, typename TA, int etype, typename DType>
class Plan<UnaryMapExp<OP, TA, DType, etype>, DType> {
 public:
  explicit Plan(const Plan<TA, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

 private:
  Plan<TA, DType> src_;
};
// remaps map tensor expression to subtype's plan
template<typename SubType, typename SrcExp, int dim, typename DType>
struct Plan<MakeTensorExp<SubType, SrcExp, dim, DType>, DType> {
 public:
  Plan(const Plan<SubType, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(y, x);
  }

 private:
  Plan<SubType, DType> src_;
};
// tranpsoe
template<typename EType, typename DType>
class Plan<TransposeExp<EType, DType>, DType> {
 public:
  explicit Plan(const Plan<EType, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(x, y);
  }

 private:
  Plan<EType, DType> src_;
};
//----------------------------------------------------------------------
// Mappings from expression to plans
//---------------------------------------------------------------------
template<typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template<typename DType>
inline Plan<ScalarExp<DType>, DType> MakePlan(const ScalarExp<DType> &e) {
  return Plan<ScalarExp<DType>, DType>(e.scalar_);
}

template<typename DstDType, typename SrcDType, typename EType, int etype>
inline Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>
MakePlan(const TypecastExp<DstDType, SrcDType, EType, etype> &e) {
  return Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>(MakePlan(e.exp));
}

template<typename T, typename DType>
inline Plan<T, DType> MakePlan(const RValueExp<T, DType> &e) {
  return Plan<T, DType>(e.self());
}

template<typename T, typename DType>
inline Plan<TransposeExp<T, DType>, DType>
MakePlan(const TransposeExp<T, DType> &e) {
  return Plan<TransposeExp<T, DType>, DType>(MakePlan(e.exp));
}

template<typename T, typename SrcExp, int dim, typename DType>
inline Plan<T, DType>
MakePlan(const MakeTensorExp<T, SrcExp, dim, DType> &e) {
  return Plan<T, DType>(e.real_self());
}

template<typename OP, typename TA, typename DType, int etype>
inline Plan<UnaryMapExp<OP, TA, DType, etype>, DType>
MakePlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return Plan<UnaryMapExp<OP, TA, DType, etype>, DType>(MakePlan(e.src_));
}

template<typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return Plan<BinaryMapExp<OP, TA, TB, DType, etype>,
              DType>(MakePlan(e.lhs_), MakePlan(e.rhs_));
}
//----------------------------------------------------------------
// Static Type inference and Type Checking
//----------------------------------------------------------------
/*!
 * \brief static type inference template,
 *        used to get the dimension of each expression,
 *        if ExpInfo<E>::kDim == -1, this means here are mismatch in expression
 *        if (ExpInfo<E>::kDevMask & cpu::kDevMask) != 0, this means this expression can be assigned to cpu
 * \tparam E expression
 */
template<typename E>
struct ExpInfo {
  static const int kDim = -1;
  static const int kDevMask = 0;
};
template<typename DType>
struct ExpInfo< ScalarExp<DType> > {
  static const int kDim = 0;
  static const int kDevMask = 0xffff;
};
template<typename E, typename DType>
struct ExpInfo<TransposeExp<E, DType> > {
  static const int kDim = ExpInfo<E>::kDim;
  static const int kDevMask = ExpInfo<E>::kDevMask;
};
template<typename DstDType, typename SrcDType, typename EType, int etype>
struct ExpInfo<TypecastExp<DstDType, SrcDType, EType, etype> > {
  static const int kDim = ExpInfo<EType>::kDim;
  static const int kDevMask = ExpInfo<EType>::kDevMask;
};
template<typename Device, int dim, typename DType>
struct ExpInfo<Tensor<Device, dim, DType> > {
  static const int kDim = dim;
  static const int kDevMask = Device::kDevMask;
};
template<typename T, typename SrcExp, int dim, typename DType>
struct ExpInfo<MakeTensorExp<T, SrcExp, dim, DType> > {
  static const int kDimSrc = ExpInfo<SrcExp>::kDim;
  static const int kDim = kDimSrc >= 0 ? dim : -1;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
template<typename OP, typename TA, typename DType, int etype>
struct ExpInfo<UnaryMapExp<OP, TA, DType, etype> > {
  static const int kDim = ExpInfo<TA>::kDim;
  static const int kDevMask = ExpInfo<TA>::kDevMask;
};
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct ExpInfo<BinaryMapExp<OP, TA, TB, DType, etype> > {
  static const int kDimLhs = ExpInfo<TA>::kDim;
  static const int kDimRhs = ExpInfo<TB>::kDim;
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ?\
      (kDimLhs == 0 ?\
       kDimRhs :\
       ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<TA>::kDevMask & ExpInfo<TB>::kDevMask;
};
/*! \brief template to do type check */
template<typename Device, int dim, typename DType, typename E>
struct TypeCheck {
  /*! \brief dimension of expression*/
  static const int kExpDim = ExpInfo<E>::kDim;
  /*! \brief whether the expression device type matches */
  static const bool kDevPass = (ExpInfo<E>::kDevMask & Device::kDevMask) != 0;
  /*! \brief whether the expression can be mapped to expression of dim */
  static const bool kMapPass = (kExpDim == 0 || kExpDim == dim) && kDevPass;
  /*! \brief whether the expression can be reduced to expression of dim */
  static const bool kRedPass = (kExpDim > dim) && kDevPass;
};
/*! \brief used to help static type check*/
template<bool kPass>
struct TypeCheckPass;
// Todo : add static assert using C++11
template<>
struct TypeCheckPass<false> {};
template<>
struct TypeCheckPass<true> {
  inline static void Error_All_Tensor_in_Exp_Must_Have_Same_Type(void) {}
  inline static void Error_TypeCheck_Not_Pass_For_Reduce_Exp(void) {}
  inline static void Error_Expression_Does_Not_Meet_Dimension_Req(void) {}
};

//----------------------------------------------------------------
// Runtime Stream Getting
//----------------------------------------------------------------
template<typename Device, typename E>
struct StreamInfo {
  inline static Stream<Device> *Get(const E &t);
};
template<int dim, typename Device, typename DType>
struct StreamInfo<Device, Tensor<Device, dim, DType> > {
  inline static Stream<Device> *Get(const Tensor<Device, dim, DType> &t) {
    return t.stream_;
  }
};
//----------------------------------------------------------------
// Runtime Shape Checking
//----------------------------------------------------------------
/*!
 * \brief runtime shape checking template
 *    get the shape of an expression, report error if shape mismatch
 * \tparam dim the dimension of the shape
 * \tparam E expression
 */
template<int dim, typename E>
struct ShapeCheck {
  inline static Shape<dim> Check(const E &t);
};
template<int dim, typename DType>
struct ShapeCheck<dim, ScalarExp<DType> > {
  inline static Shape<dim> Check(const ScalarExp<DType> &exp) {
    // use lowest dimension to mark scalar exp
    Shape<dim> shape;
    for (int i = 0; i < dim; ++i) {
      shape[i] = 0;
    }
    return shape;
  }
};
template<int dim, typename DstDType, typename SrcDType, typename EType, int etype>
struct ShapeCheck<dim, TypecastExp<DstDType, SrcDType, EType, etype> > {
  inline static Shape<dim>
  Check(const TypecastExp<DstDType, SrcDType, EType, etype> &exp) {
    return ShapeCheck<dim, EType>::Check(exp.exp);
  }
};
template<int dim, typename E, typename DType>
struct ShapeCheck<dim, TransposeExp<E, DType> > {
  inline static Shape<dim> Check(const TransposeExp<E, DType> &e) {
    // swap the lowest two dimensions
    Shape<dim> s = ShapeCheck<dim, E>::Check(e.exp);
    std::swap(s[0], s[1]);
    return s;
  }
};
template<int dim, typename Device, typename DType>
struct ShapeCheck<dim, Tensor<Device, dim, DType> > {
  inline static Shape<dim> Check(const Tensor<Device, dim, DType> &t) {
    return t.shape_;
  }
};
template<int dim, typename SrcExp, typename T, typename DType>
struct ShapeCheck<dim, MakeTensorExp<T, SrcExp, dim, DType> > {
  inline static Shape<dim>
  Check(const MakeTensorExp<T, SrcExp, dim, DType> &t) {
    return t.shape_;
  }
};
template<int dim, typename OP, typename TA, typename DType, int etype>
struct ShapeCheck<dim, UnaryMapExp<OP, TA, DType, etype> > {
  inline static Shape<dim> Check(const UnaryMapExp<OP, TA, DType, etype> &t) {
    Shape<dim> s = ShapeCheck<dim, TA>::Check(t.src_);
    return s;
  }
};
template<int dim, typename OP, typename TA, typename TB,
         typename DType, int etype>
struct ShapeCheck<dim, BinaryMapExp<OP, TA, TB, DType, etype> > {
  inline static Shape<dim>
  Check(const BinaryMapExp<OP, TA, TB, DType, etype> &t) {
    Shape<dim> shape1 = ShapeCheck<dim, TA>::Check(t.lhs_);
    Shape<dim> shape2 = ShapeCheck<dim, TB>::Check(t.rhs_);
    if (shape1[0] == 0) return shape2;
    if (shape2[0] == 0) return shape1;
    CHECK_EQ(shape1, shape2) << "BinaryMapExp: Shapes of operands are not the same";
    return shape1;
  }
};
}  // namespace expr
}  // namespace mshadow
// include definition of dot engine
//===== EXPANDIND: ../mshadow/mshadow/dot_engine-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file dot_engine-inl.h
 * \brief definitions of how Matrix Multiplications can be evaluated
 * \author Tianqi Chen
 */
#ifndef MSHADOW_DOT_ENGINE_INL_H_
#define MSHADOW_DOT_ENGINE_INL_H_

//===== EXPANDIND: ../mshadow/mshadow/extension/implicit_gemm.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file implicit_gemm.h
 * \brief support for implicit GEMM operation
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_IMPLICIT_GEMM_H_
#define MSHADOW_EXTENSION_IMPLICIT_GEMM_H_

//===== EXPANDIND: ../mshadow/mshadow/packet-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file packet-inl.h
 * \brief Generic packet vectorization code
 */
#ifndef MSHADOW_PACKET_INL_H_
#define MSHADOW_PACKET_INL_H_

#ifdef __APPLE__
#else
#endif


namespace mshadow {
/*! \brief namespace of packet math*/
namespace packet {

enum PacketArch {
  kPlain,
  kSSE2,
};

#if MSHADOW_USE_SSE
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kSSE2
#else
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kPlain
#endif

// whether packet operator is enabled.
/*!
 * \brief Generic packet type
 * \tparam DType The data type of the packet.
 * \tparam Arch the Arch of the packet.
 */
template<typename DType, PacketArch Arch = MSHADOW_DEFAULT_PACKET>
struct Packet;

template<PacketArch Arch>
struct AlignBytes {
  static const index_t value = 4;
};

}  // namespace packet
}  // namespace mshadow

namespace mshadow {
namespace packet {
/*!
 * \brief analog to cudaMallocPitch, allocate a aligned space with num_line * lspace cells
 * \param out_pitch output parameter, the actuall space allocated for each line
 * \param lspace number of cells required for each line
 * \param num_line number of lines to be allocated
 */
inline void* AlignedMallocPitch(size_t *out_pitch,
                                size_t lspace,
                                size_t num_line) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t mask = (1 << bits) - 1;

  size_t pitch = ((lspace + mask) >> bits) << bits;
  *out_pitch = pitch;
#ifdef _MSC_VER
  void *res = _aligned_malloc(pitch * num_line, 1 << bits);
#else
  void *res;
  int ret = posix_memalign(&res, 1 << bits, pitch * num_line);
  CHECK_EQ(ret, 0) << "AlignedMallocPitch failed";
#endif
  if (res == NULL) {
    LOG(FATAL) << "AlignedMallocPitch failed";
  }
  return res;
}

/*!
 * \brief free aligned space
 * \param ptr pointer to space to be freed
 */
inline void AlignedFree(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/*! \brief check if a pointer is aligned */
template<PacketArch Arch>
inline bool CheckAlign(size_t pitch) {
  const index_t bits = AlignBytes<Arch>::value;
  return !(pitch & ((1 << bits) - 1));
}

/*! \brief check if a pointer is aligned */
template<PacketArch Arch>
inline bool CheckAlign(void *ptr) {
  return CheckAlign<Arch>(reinterpret_cast<size_t>(ptr));
}

/*!
 * \brief get upper bound of aligned index of size
 * \param size size of the array
 * \param fsize size of float
 */
template<typename DType, PacketArch Arch>
inline index_t UpperAlign(index_t size) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t mask = (1 << bits) - 1;
  const index_t fsize = sizeof(DType);
  return (((size * fsize + mask) >> bits) << bits) / fsize;
}

/*!
 * \brief get lower bound of aligned index of size
 * \param size size of the array
 * \param fsize size of float
 */
template<typename DType, PacketArch Arch>
inline index_t LowerAlign(index_t size) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t fsize = sizeof(DType);
  return (((size * fsize) >> bits) << bits) / fsize;
}

/*!
 * \brief generic Packet operator
 * \tparam OP The operator
 * \tparam DType The data type
 * \tparam Arch The architecture.
 */
template<typename OP, typename DType, PacketArch Arch>
struct PacketOp {
  static const bool kEnabled = false;
};
// specialization of operators
template<typename DType, PacketArch Arch>
struct PacketOp<op::plus, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                   const Packet<DType, Arch>& rhs) {
    return lhs + rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::minus, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs - rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::mul, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs * rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::div, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs / rhs;
  }
};

template<typename DType, PacketArch Arch>
struct PacketOp<op::identity, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& src) {
    return src;
  }
};


// savers to do storage
template<typename SV, typename TFloat, PacketArch Arch>
struct Saver{
  MSHADOW_CINLINE static void Save(TFloat *dst, const Packet<TFloat, Arch>& src) {
    Packet<TFloat, Arch> lhs = Packet<TFloat, Arch>::Load(dst);
    Packet<TFloat, Arch> ans = PacketOp<typename SV::OPType, TFloat, Arch>::Map(lhs, src);
    ans.Store(dst);
  }
};
template<typename TFloat, PacketArch Arch>
struct Saver<sv::saveto, TFloat, Arch> {
  MSHADOW_CINLINE static void Save(TFloat *dst, const Packet<TFloat, Arch>& src) {
    src.Store(dst);
  }
};
}  // namespace packet
}  // namespace mshadow

//===== EXPANDIND: ../mshadow/mshadow/packet/plain-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file plain-inl.h
 * \brief support of plain packet that use the plain datatype.
 */
#ifndef MSHADOW_PACKET_PLAIN_INL_H_
#define MSHADOW_PACKET_PLAIN_INL_H_


namespace mshadow {
namespace packet {
template<typename DType>
struct Packet<DType, kPlain> {
 public:
  /*! \brief number of float in vector */
  static const index_t kSize = 1;
  /*! \brief The internal data */
  DType data_;
  // enable default copy constructor
  Packet(void) {}
  // constructor from the intrinsic type
  explicit Packet(DType data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<DType, kPlain> Fill(DType s) {
    return Packet<DType, kPlain>(s);
  }
  // load from address
  MSHADOW_CINLINE static Packet<DType, kPlain> Load(const DType* src) {
    return Packet<DType, kPlain>(*src);
  }
  // load from address
  MSHADOW_CINLINE static Packet<DType, kPlain> LoadUnAligned(const DType* src) {
    return Packet<DType, kPlain>(*src);
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<DType, kPlain>& operator=(DType s) {
    data_ = s;
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(DType* dst) const {
    *dst = data_;
  }
  // get the sum of all contents
  MSHADOW_CINLINE DType Sum() const {
    return data_;
  }
};

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator+(const Packet<DType, kPlain>& lhs,
                                                const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ + rhs.data_);
}

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator-(const Packet<DType, kPlain>& lhs,
                                                const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ - rhs.data_);
}
template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator*(const Packet<DType, kPlain>& lhs,
                                                    const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ * rhs.data_);
}

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator/(const Packet<DType, kPlain>& lhs,
                                                    const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ / rhs.data_);
}
}  // namespace packet
}  // namespace mshadow
#endif  // MSHADOW_PACKET_PLAIN_INL_H_
//===== EXPANDED: ../mshadow/mshadow/packet/plain-inl.h =====

#if MSHADOW_USE_SSE && !defined(__CUDACC__)
//===== EXPANDIND: ../mshadow/mshadow/packet/sse-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file sse-inl.h
 * \brief support of sse2 packet optimization of some operations
 * \author Tianqi Chen
 */
#ifndef MSHADOW_PACKET_SSE_INL_H_
#define MSHADOW_PACKET_SSE_INL_H_


namespace mshadow {
namespace packet {
template<>
struct Packet<float, kSSE2> {
 public:
  /*! \brief number of float in vector */
  static const index_t kSize = 4;
  /*! \brief The internal data */
  __m128 data_;
  // enable default copy constructor
  Packet(void) {}
  // constructor from the intrinsic type
  explicit Packet(__m128 data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<float, kSSE2> Fill(float s) {
    return Packet<float, kSSE2>(_mm_set1_ps(s));
  }
  // load from address
  MSHADOW_CINLINE static Packet<float, kSSE2> Load(const float* src) {
    return Packet<float, kSSE2>(_mm_load_ps(src));
  }
  // load from address
  MSHADOW_CINLINE static Packet<float, kSSE2> LoadUnAligned(const float* src) {
    return Packet<float, kSSE2>(_mm_loadu_ps(src));
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<float, kSSE2>& operator=(float s) {
    data_ = _mm_set1_ps(s);
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(float* dst) const {
    _mm_store_ps(dst, data_);
  }
  // get the sum of all contents
  MSHADOW_CINLINE float Sum() const {
    __m128 ans  = _mm_add_ps(data_, _mm_movehl_ps(data_, data_));
    __m128 rst  = _mm_add_ss(ans, _mm_shuffle_ps(ans, ans, 1));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return rst.m128_f32[0];
#else
    float rr = _mm_cvtss_f32(rst);
    return rr;
#endif
  }
};


/*! \brief vector real type for float */
template<>
struct Packet<double, kSSE2> {
  /*! \brief number of float in vector */
  static const index_t kSize = 2;
  // internal data
  __m128d data_;
  // constructor
  Packet(void) {}
  explicit Packet(__m128d data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<double, kSSE2> Fill(double s) {
    return Packet<double, kSSE2>(_mm_set1_pd(s));
  }
  // load from address
  MSHADOW_CINLINE static Packet<double, kSSE2> Load(const double* src) {
    return Packet<double, kSSE2>(_mm_load_pd(src));
  }
  MSHADOW_CINLINE static Packet<double, kSSE2> LoadUnAligned(const double* src) {
    return Packet<double, kSSE2>(_mm_loadu_pd(src));
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<double, kSSE2>& operator=(double s) {
    data_ = _mm_set1_pd(s);
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(double* dst) const {
    _mm_store_pd(dst, data_);
  }
  // get sum of all content
  inline double Sum(void) const {
    __m128d tmp =  _mm_add_sd(data_, _mm_unpackhi_pd(data_, data_));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return tmp.m128d_f64[0];
#else
    double ans = _mm_cvtsd_f64(tmp);
    return ans;
#endif
  }
};

MSHADOW_CINLINE Packet<float, kSSE2> operator+(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_add_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator+(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_add_pd(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<float, kSSE2> operator-(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_sub_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator-(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_sub_pd(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<float, kSSE2> operator*(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_mul_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator*(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_mul_pd(lhs.data_, rhs.data_));
}


MSHADOW_CINLINE Packet<float, kSSE2> operator/(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_div_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator/(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_div_pd(lhs.data_, rhs.data_));
}

}  // namespace packet
}  // namespace mshadow
#endif  // MSHADOW_PACKET_SSE_INL_H_
//===== EXPANDED: ../mshadow/mshadow/packet/sse-inl.h =====

#endif

namespace mshadow {
namespace expr {

typedef packet::PacketArch PacketArch;

// same as plan, but use packet
template<typename ExpType, typename DType, PacketArch Arch>
class PacketPlan {
 public:
  /*!
   * \brief evaluate the expression at index [y][x],
   * x will be aligned to Packet<DType, Arch>::kSize
   */
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const;
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const;
};

template <typename Device, int dim, typename DType, PacketArch Arch>
class PacketPlan<Tensor<Device, dim, DType>, DType, Arch> {
 public:
  explicit PacketPlan(const Tensor<Device, dim, DType> &t)
      :dptr_(t.dptr_), stride_(t.stride_) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::Packet<DType, Arch>::Load(&dptr_[y * stride_ + x]);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

 private:
  const DType  *dptr_;
  index_t stride_;
};

template<typename DType, PacketArch Arch>
class PacketPlan<ScalarExp<DType>, DType, Arch> {
 public:
  explicit PacketPlan(DType scalar) : scalar_(scalar) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::Packet<DType, Arch>::Fill(scalar_);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return scalar_;
  }

 private:
  DType scalar_;
};

template<typename OP, typename TA, typename TB, int etype, typename DType, PacketArch Arch>
class PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch> {
 public:
  PacketPlan(const PacketPlan<TA, DType, Arch> &lhs, const PacketPlan<TB, DType, Arch> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::PacketOp<OP, DType, Arch>::Map(lhs_.EvalPacket(y, x), rhs_.EvalPacket(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(lhs_.Eval(y, x), rhs_.Eval(y, x));
  }

 private:
  PacketPlan<TA, DType, Arch> lhs_;
  PacketPlan<TB, DType, Arch> rhs_;
};

template<typename OP, typename TA, int etype, typename DType, PacketArch Arch>
class PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch> {
 public:
  PacketPlan(const PacketPlan<TA, DType, Arch> &src) : src_(src) {}
  MSHADOW_CINLINE packet::Packet<DType> EvalPacket(index_t y, index_t x) const {
    return packet::PacketOp<OP, DType, Arch>::Map(src_.EvalPacket(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

 private:
  PacketPlan<TA, DType, Arch> src_;
};

template<PacketArch Arch, typename OP, typename TA, typename TB, typename DType, int etype>
inline PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch>
MakePacketPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template<PacketArch Arch, typename DType>
inline PacketPlan<ScalarExp<DType>, DType, Arch> MakePacketPlan(const ScalarExp<DType> &e) {
  return PacketPlan<ScalarExp<DType>, DType, Arch>(e.scalar_);
}
template<PacketArch Arch, typename T, typename DType>
inline PacketPlan<T, DType, Arch> MakePacketPlan(const RValueExp<T, DType> &e) {
  return PacketPlan<T, DType, Arch>(e.self());
}
template<PacketArch Arch, typename T, int dim, typename DType>
inline PacketPlan<T, DType, Arch>
MakePacketPlan(const MakeTensorExp<T, cpu, dim, DType> &e) {
  return PacketPlan<T, DType, Arch>(e.real_self());
}
template<PacketArch Arch, typename OP, typename TA, typename DType, int etype>
inline PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch>
MakePacketPlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch>(MakePacketPlan<Arch>(e.src_));
}
template<PacketArch Arch, typename OP, typename TA, typename TB, typename DType, int etype>
inline PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch>
MakePacketPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>,
                    DType, Arch>(MakePacketPlan<Arch>(e.lhs_), MakePacketPlan<Arch>(e.rhs_));
}

/*!
 * \brief static check packet enable
 *
 * \tparam Device the type of Device
 * \tparam dim dimension of the tensor
 * \tparam E expression
 */
template<typename E, PacketArch Arch>
struct PacketCheck{
  static const bool kPass = false;
};
template<PacketArch Arch>
struct PacketCheck<float, Arch> {
  static const bool kPass = true;
};
template<PacketArch Arch>
struct PacketCheck<double, Arch> {
  static const bool kPass = true;
};
template<typename DType, PacketArch Arch>
struct PacketCheck<ScalarExp<DType>, Arch> {
  static const bool kPass = PacketCheck<DType, Arch>::kPass;
};
template<int dim, typename DType, PacketArch Arch>
struct PacketCheck<Tensor<cpu, dim, DType>, Arch> {
  static const bool kPass = PacketCheck<DType, Arch>::kPass;
};
template<typename OP, typename TA, typename DType, int etype, PacketArch Arch>
struct PacketCheck<UnaryMapExp<OP, TA, DType, etype>, Arch> {
  static const bool kPass = PacketCheck<TA, Arch>::kPass &&
      packet::PacketOp<OP, DType, Arch>::kEnabled;
};
template<typename OP, typename TA, typename TB, typename DType, int etype, PacketArch Arch>
struct PacketCheck< BinaryMapExp<OP, TA, TB, DType, etype>, Arch> {
  static const bool kPass = packet::PacketOp<OP, DType, Arch>::kEnabled &&
      PacketCheck<TA, Arch>::kPass && PacketCheck<TB, Arch>::kPass;
};
//----------------------------------------------------
// Check if data is aligned and allow packet operation
//----------------------------------------------------
template<int dim, typename E, PacketArch Arch>
struct PacketAlignCheck {
  inline static bool Check(const E &exp) {
    return false;
  }
};
template<int dim, typename DType, PacketArch Arch>
struct PacketAlignCheck<dim, ScalarExp<DType>, Arch> {
  inline static bool Check(const ScalarExp<DType> &exp) {
    return true;
  }
};
template<int dim, typename DType, PacketArch Arch>
struct PacketAlignCheck<dim, Tensor<cpu, dim, DType>, Arch> {
  inline static bool Check(const Tensor<cpu, dim, DType> &t) {
    return packet::CheckAlign<Arch>(t.dptr_) &&
        packet::CheckAlign<Arch>(t.stride_ * sizeof(DType));
  }
};
template<int dim, typename OP, typename TA, typename DType, int etype, PacketArch Arch>
struct PacketAlignCheck<dim, UnaryMapExp<OP, TA, DType, etype>, Arch> {
  inline static bool Check(const UnaryMapExp<OP, TA, DType, etype> &t) {
    return PacketAlignCheck<dim, TA, Arch>::Check(t.src_);
  }
};
template<int dim, typename OP, typename TA, typename TB,
         typename DType, int etype, PacketArch Arch>
struct PacketAlignCheck<dim, BinaryMapExp<OP, TA, TB, DType, etype>, Arch> {
  inline static bool Check(const BinaryMapExp<OP, TA, TB, DType, etype> &t) {
    return PacketAlignCheck<dim, TA, Arch>::Check(t.lhs_) &&
        PacketAlignCheck<dim, TB, Arch>::Check(t.rhs_);
  }
};

/*!
 * \brief use PacketPlan to compute result
 */
template<typename SV, typename E, int dim, typename DType, PacketArch Arch>
inline void MapPacketPlan(Tensor<cpu, dim, DType> _dst,
                          const expr::PacketPlan<E, DType, Arch>& plan) {
  Tensor<cpu, 2, DType> dst = _dst.FlatTo2D();
  const index_t xlen = packet::LowerAlign<DType, Arch>(dst.size(1));
  for (index_t y = 0; y < dst.size(0); ++y) {
    for (index_t x = 0; x < xlen; x += packet::Packet<DType, Arch>::kSize) {
      packet::Saver<SV, DType, Arch>::Save(&dst[y][x], plan.EvalPacket(y, x));
    }
    for (index_t x = xlen; x < dst.size(1); ++x) {
      SV::Save(dst[y][x], plan.Eval(y, x));
    }
  }
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_PACKET_INL_H_
//===== EXPANDED: ../mshadow/mshadow/packet-inl.h =====


namespace mshadow {
namespace expr {
/*!
 * \brief Matrix multiplication.
 * \tparam LhsExp type of lhs expression
 * \tparam LhsExp type of rhs expression
 * \tparam DType the type of elements
 */
template<typename LhsExp, typename RhsExp, typename DType>
struct ImplicitGEMMExp:
      public Exp<ImplicitGEMMExp<LhsExp, RhsExp, DType>,
                 DType, type::kChainer> {
  /*! \brief lhs operand */
  const LhsExp &lhs_;
  /*! \brief rhs operand */
  const RhsExp &rhs_;
  /*! \brief internal production size*/
  index_t prod_size_;
  /*! \brief the shape of this expression */
  Shape<2> shape_;
  /*! \brief constructor */
  ImplicitGEMMExp(const LhsExp &lhs, const RhsExp &rhs)
      : lhs_(lhs), rhs_(rhs) {
    Shape<2> slhs = ShapeCheck<2, LhsExp>::Check(lhs_);
    Shape<2> srhs = ShapeCheck<2, RhsExp>::Check(rhs_);
    this->shape_ = mshadow::Shape2(slhs[0], srhs[1]);
    prod_size_ = slhs[1];
  }
};


template<typename LhsExp, typename RhsExp, typename DType, int e1, int e2>
inline ImplicitGEMMExp<LhsExp, RhsExp, DType>
implicit_dot(const Exp<LhsExp, DType, e1> &lhs,
             const Exp<RhsExp, DType, e2> &rhs) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == 2 && ExpInfo<RhsExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ImplicitGEMMExp<LhsExp, RhsExp, DType>(lhs.self(), rhs.self());
}

//----------------------
// Execution plan
//----------------------
template<typename LhsExp, typename RhsExp, typename DType>
struct Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType> {
 public:
  explicit Plan(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &e)
      : lhs_(MakePlan(e.lhs_)),
        rhs_(MakePlan(e.rhs_)),
        prod_size_(e.prod_size_),
        prod_size_lower_align_(packet::LowerAlign<DType, MSHADOW_DEFAULT_PACKET>(e.prod_size_)) {
  }

  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    typedef packet::Packet<DType> Packet;
    Packet sum = Packet::Fill(0);

    DType lhs_temp[Packet::kSize], rhs_temp[Packet::kSize];

    for (index_t i = 0; i < prod_size_lower_align_; i += packet::Packet<DType>::kSize) {
      // unroll
      for (index_t j = 0; j < Packet::kSize; ++j) {
        lhs_temp[j] = lhs_.Eval(y, i + j);
      }
      for (index_t j = 0; j < Packet::kSize; ++j) {
        rhs_temp[j] = rhs_.Eval(i + j, x);
      }
      sum = sum + Packet::LoadUnAligned(lhs_temp) * Packet::LoadUnAligned(rhs_temp);
    }
    DType ret_result = sum.Sum();

    for (index_t i =  prod_size_lower_align_; i < prod_size_; ++i) {
      ret_result += lhs_.Eval(y, i) * rhs_.Eval(i, x);
    }
    return ret_result;
  }

 private:
  expr::Plan<LhsExp, DType> lhs_;
  expr::Plan<RhsExp, DType> rhs_;
  const index_t prod_size_;
  const index_t prod_size_lower_align_;
};

template<typename LhsExp, typename RhsExp, typename DType>
inline Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType>
MakePlan(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &exp) {
  return Plan<ImplicitGEMMExp<LhsExp, RhsExp, DType>, DType>(exp);
}


template<int dim, typename LhsExp, typename RhsExp, typename DType>
struct ShapeCheck<dim, ImplicitGEMMExp<LhsExp, RhsExp, DType> > {
  inline static Shape<dim>
  Check(const ImplicitGEMMExp<LhsExp, RhsExp, DType> &t) {
    CHECK(dim == 2)
        << "ImplicitGEMMExp only support 2 dimension";
    Shape<dim> shape1 = ShapeCheck<dim, LhsExp>::Check(t.lhs_);
    Shape<dim> shape2 = ShapeCheck<dim, RhsExp>::Check(t.rhs_);
    CHECK_EQ(shape1[1], shape2[0])
      << "implicit_dot The matrix shape do  not match";
    return t.shape_;
  }
};

template<typename LhsExp, typename RhsExp, typename DType>
struct ExpInfo<ImplicitGEMMExp<LhsExp, RhsExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask & ExpInfo<RhsExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_IMPLICIT_GEMM_H_

//===== EXPANDED: ../mshadow/mshadow/extension/implicit_gemm.h =====


namespace mshadow {
namespace expr {
//---------------------------------------------------------------------
// Matrix Multiplications, depends on BLAS Engine
//---------------------------------------------------------------------
template<typename SV, typename Device, int ddim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct DotEngine {
  inline static void Eval(Tensor<Device, ddim, DType> *p_dst,
                          const Tensor<Device, ldim, DType> &lhs,
                          const Tensor<Device, rdim, DType> &rhs,
                          DType scale);
};
// handles the dot
template<typename Device, typename DType = default_real_t>
struct BLASEngine {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<Device> *stream) {
  }
  inline static void gemm(Stream<Device> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, DType alpha,
                          const DType *A, int lda, const DType *B, int ldb,
                          DType beta, DType *C, int ldc) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void gemv(Stream<Device> *stream,
                          bool trans, int m, int n,
                          DType alpha, const DType *A, int lda,
                          const DType *X, int incX,
                          DType beta, DType *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<Device> *stream,
                         int n,
                         const DType* X, int incX,
                         const DType* Y, int incY,
                         DType* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#if MSHADOW_STAND_ALONE
template<>
struct BLASEngine<cpu, float> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, float> lhs((float*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, float> rhs((float*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, float> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, double> lhs((double*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, double> rhs((double*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, double> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          double alpha, const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#elif (MSHADOW_USE_MKL || MSHADOW_USE_CBLAS)  // NOLINT(*)
template<>
struct BLASEngine<cpu, float> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    cblas_sgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    *ret = cblas_sdot(n, X, incX, Y, incY);
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cblas_dgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    *ret = cblas_ddot(n, X, incX, Y, incY);
  }
};
#endif  // MSHADOW_USE_CBLAS || MSHADOW_USE_MKL || MSHADOW_STAND_ALONE
// CuBLAS redirect code
#if MSHADOW_USE_CUDA
// All CuBLAS goes to here, use legacy API: not threadsafe
template<>
struct BLASEngine<gpu, float> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda,
                          const float *B, int ldb, float beta,
                          float *C, int ldc) {
    cublasStatus_t err = cublasSgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas Sgemm fail";
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, float alpha,
                          const float *A, int lda,
                          const float *X, int incX, float beta,
                          float *Y, int incY) {
    cublasStatus_t err = cublasSgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sgemv fail";
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cublasStatus_t err = cublasSger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sger fail";
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasSdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};

template<>
struct BLASEngine<gpu, double> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda,
                          const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cublasStatus_t err = cublasDgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemm fail";
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cublasStatus_t err = cublasDgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemv fail";
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cublasStatus_t err = cublasDger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dger fail";
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasDdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};
#endif  // MSHADOW_USE_CUDA
// helper function to decide which shape we are in
inline static Shape<2> GetShape(const Shape<2> &shape, bool transpose) {
  return transpose ? Shape2(shape[1], shape[0]) : shape;
}
// dst = dot(lhs[.T], rhs[.T])
template<typename SV, typename xpu,
         bool transpose_left, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 2, 2, 2, transpose_left, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 2, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sleft = GetShape(lhs.shape_, transpose_left);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    CHECK(dst.size(0) == sleft[0] && dst.size(1) == sright[1] && sleft[1] == sright[0])
      << "dot-gemm: matrix shape mismatch";
    // use column major argument to compatible with most BLAS
    BLASEngine<xpu, DType>::gemm
        (dst.stream_,
         transpose_right , transpose_left,
         transpose_right ? rhs.size(0) : rhs.size(1),
         transpose_left  ? lhs.size(1) : lhs.size(0),
         transpose_right ? rhs.size(1) : rhs.size(0),
         DType(scale * SV::AlphaBLAS()),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, lhs.stride_,
         DType(SV::BetaBLAS()),
         dst.dptr_, dst.stride_);
  }
};
template<typename SV, typename xpu, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 1, 1, 2, false, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 1, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 1, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sright = GetShape(rhs.shape, transpose_right);
    CHECK(dst.size(0) == sright[1] && lhs.size(0) == sright[0])
      << "dot-gemv: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << sright << "\n";
    BLASEngine<xpu, DType>::gemv
        (dst.stream_,
         transpose_right,
         rhs.size(1), rhs.size(0), scale * SV::AlphaBLAS(),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, 1, SV::BetaBLAS(),
         dst.dptr_, 1);
  }
};
template<typename SV, typename xpu, typename DType>
struct DotEngine<SV, xpu, 2, 1, 1, true, false, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 1, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    CHECK_EQ(dst.size(0), lhs.size(0) && dst.size(1) == rhs.size(0))
      << "dot-ger: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << rhs.shape_;
    if (SV::BetaBLAS() == 0.0f) {
      BLASEngine<xpu, DType>::ger
          (dst.stream_, rhs.size(0), lhs.size(0), scale * SV::AlphaBLAS(),
           rhs.dptr_, 1, lhs.dptr_, 1, dst.dptr_, dst.stride_);
    } else {
      DotEngine<SV, xpu, 2, 2, 2, true, false,
                DType>::Eval(dst, lhs.FlatTo2D(), rhs.FlatTo2D(), scale);
    }
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_DOT_ENGINE_INL_H_
//===== EXPANDED: ../mshadow/mshadow/dot_engine-inl.h =====


namespace mshadow {
namespace expr {
/*! \brief some engine that evaluate complex expression */
template<typename SV, typename RV, typename E, typename DType>
struct ExpComplexEngine {
  inline static void Eval(RV *dst, const E &exp);
};
/*! \brief the engine that dispatches simple operations*/
template<typename SV, typename RV, typename DType>
struct ExpEngine {
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kMapper> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kChainer> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kRValue> &exp) {
    MapExp<SV>(dst, exp);
  }
  template<typename E>
  inline static void Eval(RV *dst,
                          const Exp<E, DType, type::kComplex> &exp) {
    ExpComplexEngine<SV, RV, E, DType>::Eval(dst->ptrself(), exp.self());
  }
};
template<typename SV, typename Device, int dim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct ExpComplexEngine<SV,
                        Tensor<Device, dim, DType>,
                        DotExp<Tensor<Device, ldim, DType>,
                               Tensor<Device, rdim, DType>,
                               ltrans, rtrans, DType>,
                        DType> {
  inline static void Eval(Tensor<Device, dim, DType> *dst,
                          const DotExp<Tensor<Device, ldim, DType>,
                                       Tensor<Device, rdim, DType>,
                                       ltrans, rtrans, DType> &exp) {
    DotEngine<SV, Device, dim, ldim, rdim,
              ltrans, rtrans, DType>::Eval(dst, exp.lhs_, exp.rhs_, exp.scale_);
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPR_ENGINE_INL_H_
//===== EXPANDED: ../mshadow/mshadow/expr_engine-inl.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/broadcast.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file broadcast.h
 * \brief support for broadcast and repmat
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_BROADCAST_H_
#define MSHADOW_EXTENSION_BROADCAST_H_
namespace mshadow {
namespace expr {
/*!
 * \brief broadcast Tensor1D into a higher dimension Tensor
 * input: Tensor<Device,1>: ishape[0]
 * output: Tensor<Device,dimdst> : oshape[dimcast] = ishape[0]
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst  target tensor dimension
 * \tparam dimcast_m_dst  dimcast - dimdst
 */
template<typename SrcExp, typename DType, int dimdst, int dimdst_m_cast>
struct Broadcast1DExp:
      public MakeTensorExp<Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast>,
                           SrcExp, dimdst, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief constructor */
  Broadcast1DExp(const SrcExp &src, Shape<dimdst> shape)
      : src_(src) {
    this->shape_ = shape;
  }
};
/*!
 * \brief a expression that replicate a 1 dimension tensor in dimension dimcast
 * \param src Tensor<Device,1>: shape[0]
 * \param shape shape of output
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam dimcast target dimension where the 1D tensor will be broadcasted
 * \tparam SrcExp type of input expression
 * \tparam DType the type of elements
 * \tparam dimdst dimension of destination tensor
 * \tparam dimcast_lowest the dimension we want to cast the data into
 */
template<int dimcast, typename SrcExp, typename DType,
         int etype, int dimdst>
inline Broadcast1DExp<SrcExp, DType, dimdst, dimdst - dimcast>
broadcast(const expr::Exp<SrcExp, DType, etype> &src, Shape<dimdst> shape) {
  TypeCheckPass<dimcast < dimdst && ExpInfo<SrcExp>::kDim == 1>
                ::Error_Expression_Does_Not_Meet_Dimension_Req();
  typedef ShapeCheck<1, SrcExp> ShapeCheckDim1SrcExp;
  CHECK_EQ(ShapeCheckDim1SrcExp::Check(src.self())[0], shape[dimcast])
    << "broadcast, shape mismatch";
  return Broadcast1DExp<SrcExp, DType, dimdst,
                        dimdst - dimcast>(src.self(), shape);
}
// short cut functions
/*!
 * \brief a expression that replicate a 1 dimension tensor for nrow times
 * \param src Tensor<Device,1>: shape[0]
 * \param nrow number of rows to replicate
 * \return a expresion with type Tensor<Device,2> size(1), size(0) = nrow
 * \tparam Device which device it lies
 */
template<typename SrcExp, typename DType, int etype>
inline Broadcast1DExp<SrcExp, DType, 2, 1>
repmat(const expr::Exp<SrcExp, DType, etype> &src, index_t nrow) {
  return broadcast<1>
      (src, Shape2(nrow, ShapeCheck<1, SrcExp>::Check(src.self())[0]));
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimdst, int dimdst_m_cast>
struct Plan<Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast>, DType> {
 public:
  static const int dimcast = dimdst - dimdst_m_cast;
  explicit Plan(const Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast> &e)
      : src_(MakePlan(e.src_)),
        ystride_(e.shape_.ProdShape(dimcast + 1, dimdst - 1)),
        length_(e.shape_[dimcast]) {
    TypeCheckPass<dimcast != dimdst - 1>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, (y / ystride_) % length_);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  const index_t  ystride_, length_;
};

/*! \brief execution plan of Broadcast1DExp */
template<typename SrcExp, typename DType, int dimdst>
struct Plan<Broadcast1DExp<SrcExp, DType, dimdst, 1>, DType>{
 public:
  explicit Plan(const Broadcast1DExp<SrcExp, DType, dimdst, 1> &e)
      : src_(MakePlan(e.src_)) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, x);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_H_
//===== EXPANDED: ../mshadow/mshadow/extension/broadcast.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/unpack_patch2col.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file unpack_patch2col.h
 * \brief support for unpack
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
#define MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
namespace mshadow {
namespace expr {
/*!
 * \brief unpack local (overlap) patches of image to column of mat,
 *  can be used to implement convolution, this expression allow unpack of a batch
 *  this is a version support unpacking multiple images
 *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
 * \tparam SrcExp source expression
 * \tparam dstdim destination dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UnpackPatchToColXExp:
      public MakeTensorExp<UnpackPatchToColXExp<SrcExp, DType, srcdim>,
                           SrcExp, 2, DType>{
  /*! \brief source operand */
  const SrcExp &img_;
  /*! \brief patch height */
  index_t psize_y_;
  /*! \brief patch width */
  index_t psize_x_;
  /*! \brief patch stride */
  index_t pstride_y_;
  index_t pstride_x_;
  /*! \brief patch dilate */
  index_t pdilate_y_;
  index_t pdilate_x_;
  /*! \brief number of input channel */
  index_t i_channel_;
  /*! \brief height of img */
  index_t i_height_;
  /*! \brief width of img */
  index_t i_width_;
  /*! \brief constructor */
  UnpackPatchToColXExp(const SrcExp &img,
                       index_t psize_y,
                       index_t psize_x,
                       index_t pstride_y,
                       index_t pstride_x,
                       index_t pdilate_y,
                       index_t pdilate_x)
      : img_(img), psize_y_(psize_y), psize_x_(psize_x),
      pstride_y_(pstride_y), pstride_x_(pstride_x),
      pdilate_y_(pdilate_y), pdilate_x_(pdilate_x){
    Shape<srcdim> imshape = ShapeCheck<srcdim, SrcExp>::Check(img_);
    CHECK(imshape[srcdim - 1] >= psize_x && imshape[srcdim - 2] >= psize_y)
      << "UnpackPatchToCol:image shape smaller than patch size";
    this->i_channel_ = imshape[srcdim - 3];
    this->i_height_  = imshape[srcdim - 2];
    this->i_width_   = imshape[srcdim - 1];
    // calculate number of batches
    const index_t num = imshape.ProdShape(0, srcdim - 3);
    const index_t o_height = (i_height_ -
        (pdilate_y == 1 ? psize_y : psize_y * pdilate_y - 1)) / pstride_y + 1;
    const index_t o_width  = (i_width_  -
        (pdilate_x == 1 ? psize_x : psize_x * pdilate_x - 1)) / pstride_x + 1;
    this->shape_[1] = o_height * o_width * num;
    this->shape_[0] = psize_y * psize_x * i_channel_;
  }
};

/*!
 * \brief  unpack local (overlap) patches of image to column of mat, can be used to implement convolution
 *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
 *
 *  weight; shape[0]: out_channel, shape[1]: ichannel * psize_y * psize_x
 *  output; shape[0]: out_channel, shape[1]: out_height * out_width * num_of_images
 *  out_height = (in_height - psize_y) / pstride + 1, this means we pad inperfect patch with 0
 *  out_width  = (in_width - psize_x) / pstride + 1
 *
 * \return mat target matrix; shape[0]: in_channel*psize_y*psize_x  shape[1]: out_height*out_width * num_of_images
 * \param img source image; shape[-3]: in_channels, shape[-2]: in_height, shape[-1]: in_width, can be 3D or 4D tensor(multiple images)
 * \param psize_y height of each patch
 * \param psize_x width of each patch
 * \param pstride stride of each patch
 * \param pdilate dilate of each patch
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpack_patch2col(const Exp<SrcExp, DType, etype> &img,
                 index_t psize_y, index_t psize_x, index_t pstride, index_t pdilate) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (img.self(), psize_y, psize_x, pstride, pstride, pdilate, pdilate);
}

/*!
 *if you want to specify stride_x and stride_y
 */
template<typename SrcExp, typename DType, int etype>
inline UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpack_patch2col(const Exp<SrcExp, DType, etype> &img,
                 index_t psize_y, index_t psize_x, index_t pstride_y_, index_t pstride_x_,
                 index_t pdilate_y_, index_t pdilate_x_) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (img.self(), psize_y, psize_x, pstride_y_, pstride_x_, pdilate_y_, pdilate_x_);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<UnpackPatchToColXExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UnpackPatchToColXExp<SrcExp, DType, srcdim> &e)
      :src_(MakePlan(e.img_)),
       psize_y_(e.psize_y_), psize_x_(e.psize_x_),
       pstride_y_(e.pstride_y_), pstride_x_(e.pstride_x_),
       i_channel_(e.i_channel_), pdilate_y_(e.pdilate_y_), pdilate_x_(e.pdilate_x_),
       i_height_(e.i_height_), i_width_(e.i_width_),
       o_height_((i_height_  - (pdilate_y_ == 1 ?
           psize_y_ : psize_y_ * pdilate_y_ - 1)) / pstride_y_ + 1),
       o_width_((i_width_   - (pdilate_x_ == 1 ?
           psize_x_ : psize_x_ * pdilate_x_ - 1)) / pstride_x_ + 1) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x_offset = i % psize_x_ * pdilate_x_;
    const index_t idivp    = i / psize_x_;
    const index_t y_offset = idivp % psize_y_ * pdilate_y_;
    const index_t c = idivp / psize_y_;
    const index_t x = (j % o_width_) * pstride_x_ + x_offset;
    const index_t jdivw = j / o_width_;
    const index_t y = (jdivw % o_height_) * pstride_y_ + y_offset;
    const index_t n = jdivw / o_height_;
    if (x < i_width_ && y < i_height_) {
      return src_.Eval((n * i_channel_  + c) * i_height_ + y, x);
    } else {
      return 0.0f;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t psize_y_, psize_x_, pstride_y_, pstride_x_, i_channel_;
  const index_t pdilate_y_, pdilate_x_;
  const index_t i_height_, i_width_, o_height_, o_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
//===== EXPANDED: ../mshadow/mshadow/extension/unpack_patch2col.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/pack_col2patch.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file pack_col2patch.h
 * \brief support for pack
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_PACK_COL2PATCH_H_
#define MSHADOW_EXTENSION_PACK_COL2PATCH_H_
namespace mshadow {
namespace expr {
/*!
 * \brief reverse operation of UnpackPatchToCol,
 *    used to backprop gradient back
 *    this is a version supporting multiple images
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam dstdim destination dimension
 */
template<typename SrcExp, typename DType, int dstdim>
struct PackColToPatchXExp:
      public MakeTensorExp<PackColToPatchXExp<SrcExp, DType, dstdim>,
                           SrcExp, dstdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief patch height */
  index_t psize_y_;
  /*! \brief patch height */
  index_t psize_x_;
  /*! \brief patch stride */
  index_t pstride_y_;
  index_t pstride_x_;
  /*! \brief patch dilate */
  index_t pdilate_y_;
  index_t pdilate_x_;
  /*! \brief constructor */
  PackColToPatchXExp(const SrcExp &src, Shape<dstdim> imshape,
                     index_t psize_y, index_t psize_x,
                     index_t pstride_y, index_t pstride_x,
                     index_t pdilate_y, index_t pdilate_x)
      :src_(src), psize_y_(psize_y), psize_x_(psize_x),
       pstride_y_(pstride_y), pstride_x_(pstride_x),
       pdilate_y_(pdilate_y), pdilate_x_(pdilate_x){
    this->shape_ = imshape;
    const index_t o_height = (imshape[dstdim - 2] -
        (pdilate_y == 1 ? psize_y : psize_y * pdilate_y - 1)) / pstride_y + 1;
    const index_t o_width  = (imshape[dstdim - 1] -
        (pdilate_x == 1 ? psize_x : psize_x * pdilate_x - 1)) / pstride_x + 1;
    Shape<2> sshape = ShapeCheck<2, SrcExp>::Check(src_);
    CHECK_EQ(sshape[1], o_height * o_width * imshape.ProdShape(0, dstdim - 3))
      << "PackColToPatchExp: src.size(1) mismatch";
    CHECK_EQ(sshape[0], psize_y * psize_x * imshape[dstdim - 3])
      << "PackColToPatchExp: src.size(0) mismatch";
  }
};
/*!
 * \brief reverse operation of pack_col2patch, can be used to implement deconvolution
 * \return packed img expression
 * \param mat source matrix
 * \param imshape shape of target img
 * \param psize_y height of each patch
 * \param psize_x height of each patch
 * \param pstride stride of each patch
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam dstdim destination dimension
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int dstdim, int etype>
inline PackColToPatchXExp<SrcExp, DType, dstdim>
pack_col2patch(const expr::Exp<SrcExp, DType, etype> &src,
               Shape<dstdim> imshape, index_t psize_y,
               index_t psize_x, index_t pstride, index_t pdilate) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  CHECK(imshape[dstdim - 1] >= psize_x && imshape[dstdim - 2] >= psize_y)
    << "PackColToPatch:image shape smaller than patch size";
  return PackColToPatchXExp<SrcExp, DType, dstdim>(src.self(), imshape,
                                                   psize_y, psize_x, pstride, pstride,
                                                   pdilate, pdilate);
}
/*!
 *if you want to specify kstride_y and kstride_x
 */
template<typename SrcExp, typename DType, int dstdim, int etype>
inline PackColToPatchXExp<SrcExp, DType, dstdim>
pack_col2patch(const expr::Exp<SrcExp, DType, etype> &src,
               Shape<dstdim> imshape, index_t psize_y,
               index_t psize_x, index_t pstride_y, index_t pstride_x,
               index_t pdilate_y, index_t pdilate_x) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  CHECK(imshape[dstdim - 1] >= psize_x && imshape[dstdim - 2] >= psize_y)
    << "PackColToPatch:image shape smaller than patch size";
  return PackColToPatchXExp<SrcExp, DType, dstdim>(src.self(), imshape,
                                                   psize_y, psize_x, pstride_y, pstride_x,
                                                   pdilate_y, pdilate_x);
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dstdim>
struct Plan<PackColToPatchXExp<SrcExp, DType, dstdim>, DType> {
 public:
  explicit Plan(const PackColToPatchXExp<SrcExp, DType, dstdim> &e)
      :src_(MakePlan(e.src_)), psize_y_(e.psize_y_),
       psize_x_(e.psize_x_), pstride_y_(e.pstride_y_), pstride_x_(e.pstride_x_),
       i_channel_(e.shape_[dstdim - 3]), pdilate_y_(e.pdilate_y_), pdilate_x_(e.pdilate_x_),
       i_height_(e.shape_[dstdim - 2]),
       o_height_((e.shape_[dstdim - 2]  - (pdilate_y_ == 1 ?
           psize_y_ : psize_y_ * pdilate_y_ - 1)) / pstride_y_ + 1),
       o_width_((e.shape_[dstdim - 1]  - (pdilate_x_ == 1 ?
           psize_x_ : psize_x_ * pdilate_x_ - 1)) / pstride_x_ + 1) {
    // note: i/o convention are same as unpack
  }
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t y = i % i_height_;
    const index_t idivh = i / i_height_;
    const index_t c = idivh % i_channel_;
    const index_t n = idivh / i_channel_;
    const index_t x = j;

    const index_t psize_y_dilate = (pdilate_y_ == 1 ? psize_y_ : psize_y_ * pdilate_y_ - 1);
    const index_t psize_x_dilate = (pdilate_x_ == 1 ? psize_x_ : psize_x_ * pdilate_x_ - 1);

    const index_t py_min =
        y < psize_y_dilate ? y % pdilate_y_ : (y-psize_y_dilate + pstride_y_) / pstride_y_;
    const index_t px_min =
        x < psize_x_dilate ? x % pdilate_x_ : (x-psize_x_dilate + pstride_x_) / pstride_x_;
    const index_t py_max = min((y + pstride_y_) / pstride_y_, o_height_);
    const index_t px_max = min((x + pstride_x_) / pstride_x_, o_width_);
    DType res = static_cast<DType>(0);
    for (index_t py = py_min; py < py_max; py += pdilate_y_) {
      for (index_t px = px_min; px < px_max; px += pdilate_x_) {
        res += src_.Eval(((c * psize_y_ + (y - py*pstride_y_) / pdilate_y_) * psize_x_ +
                         (x - px * pstride_x_) / pdilate_x_),
                         (n * o_height_ + py) * o_width_ + px);
      }
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t psize_y_, psize_x_, pstride_y_, pstride_x_, i_channel_;
  const index_t pdilate_y_, pdilate_x_;
  const index_t i_height_, o_height_, o_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_PACK_COL2PATCH_H_
//===== EXPANDED: ../mshadow/mshadow/extension/pack_col2patch.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/reshape.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file reshape.h
 * \brief support for reshape
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_RESHAPE_H_
#define MSHADOW_EXTENSION_RESHAPE_H_
namespace mshadow {
namespace expr {
/*!
 * \brief reshape the content to another shape
 * input: Tensor<Device,dimsrc>: ishape
 * output: Tensor<Device,dimdst> ishape.Size() == oshape.Size()
 * \tparam SrcExp source expression
 * \tparam dimdst target dimension
 * \tparam dimsrc source dimension
 */
template<typename SrcExp, typename DType, int dimdst, int dimsrc>
struct ReshapeExp:
      public MakeTensorExp<ReshapeExp<SrcExp, DType, dimdst, dimsrc>,
                           SrcExp, dimdst, DType> {
  /*! \brief source expression */
  const SrcExp &src_;
  /*! \brief smallest dimension of input */
  index_t ishapex_;
  /*! \brief constructor */
  ReshapeExp(const SrcExp &src, Shape<dimdst> shape)
      : src_(src) {
    Shape<dimsrc> ishape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    CHECK_EQ(ishape.Size(), shape.Size()) << "reshape size must match";
    ishapex_ = ishape[dimsrc - 1];
    this->shape_ = shape;
  }
};
/*!
 * \brief a expression that reshapes a tensor to another shape
 * \param src Tensor<Device,dimsrc>:
 * \param oshape target shape
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam SrcExp source expression
 * \tparam etype source expression type
 * \tparam dimdst target dimension
 */
template<typename SrcExp, typename DType, int etype, int dimdst>
inline ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim>
reshape(const Exp<SrcExp, DType, etype> &src, Shape<dimdst> oshape) {
  return ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim>
      (src.self(), oshape);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimdst, int dimsrc>
struct Plan<ReshapeExp<SrcExp, DType, dimdst, dimsrc>, DType> {
 public:
  explicit Plan(const ReshapeExp<SrcExp, DType, dimdst, dimsrc> &e)
      : src_(MakePlan(e.src_)),
        oshapex_(e.shape_[dimdst - 1]), ishapex_(e.ishapex_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    const index_t idx = y * oshapex_ + x;
    return src_.Eval(idx / ishapex_, idx % ishapex_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t oshapex_, ishapex_;
};
// special work plan for 1 dimensional data
template<typename SrcExp, typename DType, int dimdst>
struct Plan<ReshapeExp<SrcExp, DType, dimdst, 1>, DType> {
 public:
  explicit Plan(const ReshapeExp<SrcExp, DType, dimdst, 1> &e)
      : src_(MakePlan(e.src_)), oshapex_(e.shape_[dimdst - 1]) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, y * oshapex_ + x);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t oshapex_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_RESHAPE_H_
//===== EXPANDED: ../mshadow/mshadow/extension/reshape.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/swapaxis.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file swapaxis.h
 * \brief support for swapaxis
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_SWAPAXIS_H_
#define MSHADOW_EXTENSION_SWAPAXIS_H_
namespace mshadow {
namespace expr {
/*!
 * \brief swap two axis of a tensor
 * input: Tensor<Device,dim>: ishape
 * output: Tensor<Device,dimdst> oshape[a1],oshape[a2] = ishape[a2],oshape[a1]
 *
 * \tparam SrcExp type of source expression
 * \tparam DType the type of elements 
 * \tparam dimsrc source dimension, assert a1 > a2
 * \tparam m_a1 one dimension to be swapped, encoded by dimsrc - a1 
 * \tparam a2 second dimension to be swapped, encoded by a2
 */
template<typename SrcExp, typename DType, int dimsrc, int m_a1, int a2>
struct SwapAxisExp:
      public MakeTensorExp<SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2>,
                           SrcExp, dimsrc, DType> {
  // decode the a1, a2
  static const int a1 = dimsrc - m_a1;
  /*! \brief source expression */
  const SrcExp &src_;
  /*! \brief constructor */
  explicit SwapAxisExp(const SrcExp &src) : src_(src) {
    this->shape_ = ShapeCheck<dimsrc, SrcExp>::Check(src);
    std::swap(this->shape_[a1], this->shape_[a2]);
  }
};
/*!
 * \brief a expression that reshapes a tensor to another shape
 * \param src Tensor<Device,dimsrc>:
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam a1 higher dimension to be swapped, assert a1 > a2
 * \tparam a2 lower dimension to be swapped
 * \tparam SrcExp source expression
 * \tparam DType the type of elements 
 * \tparam etype source expression type
 */
template<int a1, int a2, typename SrcExp, typename DType, int etype>
inline SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
                   ExpInfo<SrcExp>::kDim - a1, a2>
swapaxis(const Exp<SrcExp, DType, etype> &src) {
  typedef ExpInfo<SrcExp> Info;
  TypeCheckPass<Info::kDim >= a1 + 1 && Info::kDim >= a2 + 1 &&
                a2 < a1>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
                     ExpInfo<SrcExp>::kDim - a1, a2>(src.self());
}
template<typename SrcExp, typename DType, int dimsrc, int m_a1, int a2>
struct Plan<SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2>, DType> {
 public:
  // decode the a1
  static const int a1 = dimsrc - m_a1;
  explicit Plan(const SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2> &e)
      : src_(MakePlan(e.src_)),
        shapey_(e.shape_.ProdShape(a1 + 1, dimsrc - 1)),
        shapez_(e.shape_[a1]),
        shapec_(e.shape_.ProdShape(a2 + 1, a1)),
        shapen_(e.shape_[a2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % shapey_;
    i /= shapey_;
    const index_t z = i % shapez_;
    i /= shapez_;
    const index_t c = i % shapec_;
    i /= shapec_;
    const index_t n = i % shapen_;
    // swap z and n
    return src_.Eval(((((i / shapen_) * shapez_ + z) * shapec_ +
                          c) * shapen_ + n) * shapey_ + y, j);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t shapey_, shapez_, shapec_, shapen_;
};
template<typename SrcExp, typename DType, int dimsrc, int a2>
struct Plan<SwapAxisExp<SrcExp, DType, dimsrc, 1, a2>, DType> {
 public:
  explicit Plan(const SwapAxisExp<SrcExp, DType, dimsrc, 1, a2> &e)
      : src_(MakePlan(e.src_)),
        shapex_(e.shape_[dimsrc - 1]),
        shapey_(e.shape_.ProdShape(a2 + 1, dimsrc - 1)),
        shapez_(e.shape_[a2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t x) const {
    // swap x and z
    const index_t y = i % shapey_;
    i /= shapey_;
    const index_t z = i % shapez_;
    const index_t n = i / shapez_;
    return src_.Eval((n * shapex_ + x) * shapey_ + y , z);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t shapex_, shapey_, shapez_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SWAPAXIS_H_
//===== EXPANDED: ../mshadow/mshadow/extension/swapaxis.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/reduceto1d.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file reduceto1d.h
 * \brief support for sum_rows and sumall_except_dim
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_REDUCETO1D_H_
#define MSHADOW_EXTENSION_REDUCETO1D_H_
namespace mshadow {
namespace expr {
/*!
 * \brief reduction to 1 dimension tensor
 * input: Tensor<Device,k>: ishape
 * output: Tensor<Device,1> shape[0] = ishape[dimkeep];
 *
 * \tparam SrcExp type of expression to be reduced
 * \tparam DType the data type of the scalar
 * \tparam Reducer which reducer to use
 * \tparam m_dimkeep which dimension to be kept, encoded with dimsrc - dimkeep
 */
template<typename SrcExp, typename DType, typename Reducer, int m_dimkeep>
struct ReduceTo1DExp:
      public Exp<ReduceTo1DExp<SrcExp, DType, Reducer, m_dimkeep>,
                 DType, type::kComplex> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief source operand, scale of the  */
  DType scale_;
  /*! \brief construct a repmat expression from src and nrow */
  ReduceTo1DExp(const SrcExp& src, DType scale) : src_(src), scale_(scale) {}
};
/*!
 * \brief a sum over all dimensions, except dimkeep
 * \param exp input expression that must be a matrix Tensor<?,2>
 * \return a expresion with type Tensor<Device,1>
 * \tparam dimkeep the dimension that will be kept
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<int dimkeep,  typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, red::sum,
                     ExpInfo<SrcExp>::kDim - dimkeep>
sumall_except_dim(const Exp<SrcExp, DType, etype> &exp) {
  return ReduceTo1DExp<SrcExp, DType, red::sum,
                       ExpInfo<SrcExp>::kDim - dimkeep>(exp.self(), 1);
}
/*!
 * \brief reduce over all dimensions, except dimkeep
 * \param exp input expression that must be a matrix Tensor<?,2>
 * \return a expresion with type Tensor<Device,1>
 * \tparam dimkeep the dimension that will be kept
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<int dimkeep, typename Reducer, typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, Reducer,
                     ExpInfo<SrcExp>::kDim - dimkeep>
reduce_except_dim(const Exp<SrcExp, DType, etype> &exp) {
  return ReduceTo1DExp<SrcExp, DType, Reducer,
                       ExpInfo<SrcExp>::kDim - dimkeep>(exp.self(), 1);
}
/*!
 * \brief a expression that sum over rows of a matrix
 * \param exp input expression that must be a matrix Tensor<?, 2>
 * \return a expresion with type Tensor<Device, 1>
 * \tparam SrcExp expression
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, red::sum, 1>
sum_rows(const Exp<SrcExp, DType, etype> &exp) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim ==2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return sumall_except_dim<1>(exp);
}
template<typename SV, typename Device, typename DType,
         typename SrcExp, typename Reducer, int m_dimkeep>
struct ExpComplexEngine<SV,
                        Tensor<Device, 1, DType>,
                        ReduceTo1DExp<SrcExp, DType, Reducer, m_dimkeep>,
                        DType> {
  static const int dimkeep = ExpInfo<SrcExp>::kDim - m_dimkeep;
  inline static void Eval(Tensor<Device, 1, DType> *dst,
                          const ReduceTo1DExp<SrcExp, DType,
                                              Reducer, m_dimkeep> &exp) {
    TypeCheckPass<m_dimkeep != 1>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
    MapReduceKeepHighDim<SV, Reducer, dimkeep>(dst, exp.src_, exp.scale_);
  }
};
template<typename SV, typename Device, typename DType,
         typename SrcExp, typename Reducer>
struct ExpComplexEngine<SV,
                        Tensor<Device, 1, DType>,
                        ReduceTo1DExp<SrcExp, DType, Reducer, 1>, DType> {
  inline static void Eval(Tensor<Device, 1, DType> *dst,
                          const ReduceTo1DExp<SrcExp, DType, Reducer, 1> &exp) {
    MapReduceKeepLowest<SV, Reducer>(dst, exp.src_, exp.scale_);
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCETO1D_H_
//===== EXPANDED: ../mshadow/mshadow/extension/reduceto1d.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/spatial_pool.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file spatial_pool.h
 * \brief support for spatial pooling
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_POOL_H_
#define MSHADOW_EXTENSION_SPATIAL_POOL_H_
namespace mshadow {
namespace expr {
/*!
 * \brief pooling expression, do reduction over local patches of a image
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct PoolingExp:
      public MakeTensorExp<PoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  /*! \brief source height shape[1] */
  index_t src_height_;
  /*! \brief source width shape[0] */
  index_t src_width_;
  /*! \brief constructor */
  PoolingExp(const SrcExp &src,
             index_t ksize_y, index_t ksize_x, index_t kstride)
      : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK(sshape[srcdim - 1] >= ksize_x && sshape[srcdim - 2] >= ksize_y)
      << "PoolingExp: kernel must be smaller than image";
    this->src_height_ = sshape[srcdim - 2];
    this->src_width_  = sshape[srcdim - 1];
    this->shape_ = sshape;
    this->shape_[srcdim - 2] = (src_height_ - ksize_y) / kstride + 1;
    this->shape_[srcdim - 1] = (src_width_  - ksize_x) / kstride + 1;
  }
  /*! \brief constructor, specify shape */
  PoolingExp(const SrcExp &src, Shape<2> pshape,
             index_t ksize_y, index_t ksize_x, index_t kstride)
      : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK(sshape[srcdim - 1] >= ksize_x && sshape[srcdim - 2] >= ksize_y)
      << "PoolingExp: kernel must be smaller than image";
    this->src_height_ = sshape[srcdim - 2];
    this->src_width_  = sshape[srcdim - 1];
    this->shape_ = sshape;
    this->shape_[srcdim - 2] = pshape[0];
    this->shape_[srcdim - 1] = pshape[1];
  }
};
/*!
 * \brief pooling subregion results together
 * \param src source image, shape: (batch, channel, height, width)
 * \param ksize_y kernel size in height
 * \param ksize_x kernel size in width
 * \param kstride stride for each kernel
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline PoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
pool(const Exp<SrcExp, DType, etype> &src,
     index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), ksize_y, ksize_x, kstride);
}
/*!
 * \brief same as pool, except the output shape is specified by pshape
 * \param src source image
 * \param pshape ouput shape
 * \param ksize_y kernel size in y
 * \param ksize_x kernel size in x
 * \param kstride stride for each kernel
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp,
         typename DType, int etype>
inline PoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
pool(const Exp<SrcExp, DType, etype> &src, Shape<2> pshape,
     index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), pshape, ksize_y, ksize_x, kstride);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<PoolingExp< Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const PoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        new_height_(e.shape_[srcdim - 2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t py = i % new_height_;
    const index_t y_start = py * kstride_;
    const index_t y_end = min(y_start + ksize_y_, src_height_);
    const index_t px = j;
    const index_t x_start = px * kstride_;
    const index_t x_end = min(x_start + ksize_x_, src_width_);
    const index_t c = i / new_height_;

    DType res; Reducer::SetInitValue(res);
    for (index_t y = y_start; y < y_end; ++y) {
      for (index_t x = x_start; x < x_end; ++x) {
        Reducer::Reduce(res, src_.Eval(c * src_height_ + y, x));
      }
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t ksize_y_, ksize_x_, kstride_;
  const index_t src_height_, src_width_;
  const index_t new_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_POOL_H_
//===== EXPANDED: ../mshadow/mshadow/extension/spatial_pool.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/spatial_unpool.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file spatial_unpool.h
 * \brief support for unpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
#define MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
namespace mshadow {
namespace expr {
/*!
 * \brief unpooling expr reverse operation of pooling, used to pass gradient back
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct UnPoolingExp:
      public MakeTensorExp<UnPoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of pooled part, to be propgate down */
  const SrcExp &grad_pooled_;
  /*! \brief shape of pooled expression */
  index_t pshape_y_;
  /*! \brief shape of pooled expression */
  index_t pshape_x_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  /*! \brief constructor */
  UnPoolingExp(const SrcExp &data_src,
               const SrcExp &data_pooled,
               const SrcExp &grad_pooled,
               index_t ksize_y, index_t ksize_x, index_t kstride)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_pooled_(grad_pooled),
        ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    typedef ShapeCheck<srcdim, SrcExp> ShapeCheckSrcDimSrcExp;
    CHECK_EQ(pshape, ShapeCheckSrcDimSrcExp::Check(data_pooled))
      << "UnPoolingExp: pooled shape mismatch";
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0;  k < srcdim - 2; ++k) {
      CHECK_EQ(pshape[k], sshape[k]) << "UnPoolingExp: pool and src shape mismatch";
    }
    pshape_x_ = pshape[srcdim - 1];
    pshape_y_ = pshape[srcdim - 2];
    this->shape_ = sshape;
  }
};
/*!
 * \brief unpooling gradient for 4D, backprop gradient value back, revserse operation of pooling,
 *   same as unpooling, but allows unequal size of kernel
 * \param data_src  source input, corresponds to src in pooling
 * \param data_pooled result of pooled data, corresponds to result of pooling
 * \param grad_pooled gradient data of pooled part, to be propgate down
 * \param ksize_y kernel height
 * \param ksize_x kernel width
 * \param kstride stride for each kernel
 * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline UnPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
       index_t ksize_y, index_t ksize_x, index_t kstride) {
  return UnPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (data_src.self(), data_pooled.self(), grad_pooled.self(),
       ksize_y, ksize_x, kstride);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<UnPoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UnPoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : data_src_(MakePlan(e.data_src_)), data_pooled_(MakePlan(e.data_pooled_)),
        grad_pooled_(MakePlan(e.grad_pooled_)), sshape_y_(e.shape_[srcdim - 2]),
        pshape_y_(e.pshape_y_),  pshape_x_(e.pshape_x_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t x = j;
    const index_t y = i % sshape_y_;
    const index_t c = i / sshape_y_;
    const DType vsrc = data_src_.Eval(i, j);
    const index_t py_min =
        y < ksize_y_ ? 0 : (y - ksize_y_ + kstride_) / kstride_;
    const index_t px_min =
        x < ksize_x_ ? 0 : (x - ksize_x_ + kstride_) / kstride_;
    const index_t py_max = min((y + kstride_) / kstride_, pshape_y_);
    const index_t px_max = min((x + kstride_) / kstride_, pshape_x_);

    DType val = static_cast<DType>(0);
    for (index_t py = py_min; py < py_max; ++py) {
      for (index_t px = px_min; px < px_max; ++px) {
        val += Reducer::PartialGrad(vsrc,
                                    data_pooled_.Eval(c * pshape_y_ + py, px)) *
                                    grad_pooled_.Eval(c * pshape_y_ + py, px);
      }
    }

    return val;
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
  const index_t sshape_y_, pshape_y_, pshape_x_;
  const index_t ksize_y_, ksize_x_;
  const index_t kstride_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
//===== EXPANDED: ../mshadow/mshadow/extension/spatial_unpool.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/channel_pool.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file channel_pool.h
 * \brief support for chpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHANNEL_POOL_H_
#define MSHADOW_EXTENSION_CHANNEL_POOL_H_
namespace mshadow {
namespace expr {
/*!
 * \brief channel pooling expression, do reduction over (local nearby) channels,
 *        used to implement local response normalization
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct ChannelPoolingExp:
      public MakeTensorExp<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief neighbor size */
  index_t nsize_;
  /*! \brief stride of pooling */
  index_t stride_;
  /*! \brief pad of pooling of each side */
  index_t pad_;
  index_t src_channel_;
  /*! \brief constructor */
  ChannelPoolingExp(const SrcExp &src, index_t nsize, index_t stride, index_t pad)
      : src_(src), nsize_(nsize), stride_(stride), pad_(pad) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->src_channel_ = this->shape_[srcdim - 3];
    CHECK_GE(this->shape_[srcdim - 3], nsize_)
      << "chpool: local size must be smaller than nchannels";
    this->shape_[srcdim - 3] = (this->src_channel_ - nsize + pad * 2 + 1) / stride;
  }
};
/*!
 * \brief  channel pooling, do reduction over (local nearby) channels,
 *         used to implement local response normalization
 * \param src source data
 * \param nsize neighbor size
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
chpool(const Exp<SrcExp, DType, etype> &src, index_t nsize) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  CHECK_EQ(nsize % 2, 1) << "chpool: if no pad is specified, local size must be odd";
  return ChannelPoolingExp<Reducer, SrcExp,
                           DType, ExpInfo<SrcExp>::kDim>(src.self(), nsize, 1, nsize / 2);
}

template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
chpool(const Exp<SrcExp, DType, etype> &src, index_t nsize, index_t stride, index_t pad) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChannelPoolingExp<Reducer, SrcExp,
                           DType, ExpInfo<SrcExp>::kDim>(src.self(), nsize, stride, pad);
}

//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ChannelPoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), channel_(e.shape_[srcdim - 3]),
        height_(e.shape_[srcdim - 2]), width_(e.shape_[srcdim - 1]),
        hnsize_(e.nsize_), stride_(e.stride_), pad_(e.pad_),
        src_channel_(e.src_channel_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % channel_;
    const index_t n = i / channel_;
    const index_t x = j;
    const index_t cstart = c * stride_ < pad_ ? 0  : c * stride_ - pad_;
    const index_t cend   = min(cstart + hnsize_, channel_);
    DType res; Reducer::SetInitValue(res);
    for (index_t cc = cstart; cc < cend; ++cc) {
      Reducer::Reduce(res, src_.Eval((n * src_channel_ + cc) * height_ + y, x));
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t channel_, height_, width_, hnsize_, stride_, pad_, src_channel_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHANNEL_POOL_H_

//===== EXPANDED: ../mshadow/mshadow/extension/channel_pool.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/channel_unpool.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file channel_pool.h
 * \brief support for chpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_
#define MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_
namespace mshadow {
namespace expr {
/*!
 * \brief channel pooling expression, do reduction over (local nearby) channels,
 *        used to implement local response normalization
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct ChannelUnpoolingExp:
      public MakeTensorExp<ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of pooled part, to be propgate down */
  const SrcExp &grad_pooled_;
  /*! \brief channel of pooled expression */
  index_t pchannel_;
  /*! \brief kernel size in height */
  index_t nsize_;
  /*! \brief kernel size in width */
  index_t kstride_;
  /*! \brief pad */
  index_t pad_;
  /*! \brief constructor */
  ChannelUnpoolingExp(const SrcExp &data_src,
               const SrcExp &data_pooled,
               const SrcExp &grad_pooled,
               index_t nsize, index_t kstride, index_t pad)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_pooled_(grad_pooled),
        nsize_(nsize), kstride_(kstride), pad_(pad) {
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    typedef ShapeCheck<srcdim, SrcExp> ShapeCheckSrcDimSrcExp;
    CHECK_EQ(pshape, ShapeCheckSrcDimSrcExp::Check(data_pooled))
      << "ChannelUnPoolingExp: data and grad shape mismatch";
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0; k < srcdim; ++k) {
      if (k == 1) {
        continue;
      }
      CHECK_EQ(pshape[k], sshape[k])
        << "ChannelUnPoolingExp: pooled tensor and src tensor shape mismatch"
        << pshape[k]
        << " vs "
        << sshape[k];
    }
    pchannel_ = pshape[1];
    this->shape_ = sshape;
  }
};
/*!
 * \brief  channel unpooling, do unroll over (local nearby) channels
 * \param src source data
 * \param nsize neighbor size
 * \param stride stride of the pooling
 * \param pad number of padding at each side
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
ch_unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
      index_t nsize, index_t stride, index_t pad) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
        (data_src.self(), data_pooled.self(), grad_pooled.self(), nsize, stride, pad);
}

template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
ch_unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled, index_t nsize) {
  return ch_unpool(data_src, data_pooled, grad_pooled, nsize, 1, nsize / 2);
}


//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : data_src_(e.data_src_), data_pooled_(e.data_pooled_),
        grad_pooled_(e.grad_pooled_), channel_(e.shape_[srcdim - 3]),
        height_(e.shape_[srcdim - 2]), pchannel_(e.pchannel_),
        hnsize_(e.nsize_), stride_(e.kstride_), pad_(e.pad_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const DType vsrc = data_src_.Eval(i, j);
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % channel_;
    const index_t n = i / channel_;
    const index_t x = j;
    const index_t cstart = c < hnsize_ - pad_ ? 0
                        : (c - (hnsize_ - pad_) + stride_) / stride_;
    const index_t cend = min((c + pad_ + stride_) / stride_, channel_);
    DType val = static_cast<DType>(0);
    for (index_t cc = cstart; cc < cend; ++cc) {
      val += Reducer::PartialGrad(vsrc,
                                  data_pooled_.Eval((n * pchannel_ + cc) * height_ + y, x)) *
                                  grad_pooled_.Eval((n * pchannel_ + cc) * height_ + y, x);
    }
    return val;
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
  const index_t channel_, height_, pchannel_, hnsize_, stride_, pad_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_

//===== EXPANDED: ../mshadow/mshadow/extension/channel_unpool.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/pad.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file pad.h
 * \brief support for pad
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_PAD_H_
#define MSHADOW_EXTENSION_PAD_H_
namespace mshadow {
namespace expr {
/*!
 * \brief padding expression, pad a image with zeros
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct PaddingExp:
      public MakeTensorExp<PaddingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief pad size in y */
  index_t pad_y_;
  /*! \brief pad size in x */
  index_t pad_x_;
  /*! \brief source tensor height */
  index_t src_height_;
  /*! \brief source tensor width */
  index_t src_width_;
  /*! \brief constructor */
  PaddingExp(const SrcExp &src, index_t pad_y, index_t pad_x)
      : src_(src), pad_y_(pad_y), pad_x_(pad_x) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    src_height_ = this->shape_[srcdim - 2];
    src_width_  = this->shape_[srcdim - 1];
    this->shape_[srcdim - 2] += pad_y * 2;  // height
    this->shape_[srcdim - 1] += pad_x * 2;  // width
  }
};
/*!
 * \brief padding expression, pad a image with zeros on boundaries, padding affects shape[0], and shape[1]
 * \param src original image batches
 * \param pad padding size
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
pad(const Exp<SrcExp, DType, etype> &src, index_t pad) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), pad, pad);
}
/*!
 * \brief padding expression, pad a image with zeros on boundaries, padding affects shape[0], and shape[1]
 * \param src original image batches
 * \param pad_y padding size in y
 * \param pad_x padding size in x
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
pad(const Exp<SrcExp, DType, etype> &src, index_t pad_y, index_t pad_x) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), pad_y, pad_x);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<PaddingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const PaddingExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        pad_y_(e.pad_y_), pad_x_(e.pad_x_),
        new_height_(e.shape_[srcdim - 2]),
        src_height_(e.src_height_), src_width_(e.src_width_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    if (y < pad_y_ || x < pad_x_) return static_cast<DType>(0);
    const index_t h = y - pad_y_;
    const index_t w = x - pad_x_;
    if (h < src_height_ && w < src_width_) {
      return src_.Eval(c * src_height_ + h, w);
    } else {
      return static_cast<DType>(0);
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t pad_y_;
  const index_t pad_x_;
  const index_t new_height_;
  const index_t src_height_;
  const index_t src_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_PAD_H_
//===== EXPANDED: ../mshadow/mshadow/extension/pad.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/crop.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file crop.h
 * \brief support for crop
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CROP_H_
#define MSHADOW_EXTENSION_CROP_H_
namespace mshadow {
namespace expr {
/*!
 * \brief crop expression, cut off the boundary region, reverse operation of padding
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct CroppingExp:
      public MakeTensorExp<CroppingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief pad height */
  index_t pad_height_;
  /*! \brief pad height */
  index_t pad_width_;
  /*! \brief src height */
  index_t src_height_;
  /*! \brief constructor */
  explicit CroppingExp(const SrcExp &src, Shape<2> cshape)
      : src_(src) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK_GE(this->shape_[srcdim - 2], cshape[0]) << "CroppingExp: height requirement not met";
    CHECK_GE(this->shape_[srcdim - 1], cshape[1]) << "CroppingExp: width requirement not met";
    pad_height_ = (this->shape_[srcdim - 2] - cshape[0]) / 2;
    pad_width_ = (this->shape_[srcdim - 1] - cshape[1]) / 2;
    src_height_ = this->shape_[srcdim - 2];
    this->shape_[srcdim - 2] = cshape[0];  // height
    this->shape_[srcdim - 1] = cshape[1];  // width
  }
  /*! \brief constructor */
  explicit CroppingExp(const SrcExp &src, Shape<2> cshape,
                       index_t start_height, index_t start_width)
      : src_(src), pad_height_(start_height), pad_width_(start_width) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK_GE(this->shape_[srcdim - 2], cshape[0] + start_height)
      << "CroppingExp: height requirement not met";
    CHECK_GE(this->shape_[srcdim - 1], cshape[1] + start_width)
      << "CroppingExp: width requirement not met";
    src_height_ = this->shape_[srcdim - 2];
    this->shape_[srcdim - 2] = cshape[0];  // height
    this->shape_[srcdim - 1] = cshape[1];  // width
  }
};  // struct CroppingExp
/*!
 * \brief revserse operationg of padding, cut off boundaries,
 *   crop output from center of input
 * \param src original image batches
 * \param oshape output shape to be cropped
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
crop(const Exp<SrcExp, DType, etype> &src, Shape<2> oshape) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), oshape);
}
/*!
 * \brief same as crop, but can specify starting position to do cropping
 * \param src original image batches
 * \param oshape output shape to be cropped
 * \param start_height start height position to do cropping
 * \param start_width  start width position to do cropping
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
crop(const Exp<SrcExp, DType, etype> &src, Shape<2> oshape,
     index_t start_height, index_t start_width) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), oshape, start_height, start_width);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<CroppingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const CroppingExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        pad_height_(e.pad_height_), pad_width_(e.pad_width_),
        new_height_(e.shape_[srcdim - 2]), src_height_(e.src_height_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = y + pad_height_;
    const index_t w = x + pad_width_;
    return src_.Eval(c * src_height_ + h, w);
  }
 private:
  Plan<SrcExp, DType> src_;
  const index_t pad_height_, pad_width_;
  const index_t new_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CROP_H_
//===== EXPANDED: ../mshadow/mshadow/extension/crop.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/mirror.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file mirror.h
 * \brief support for mirror
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_MIRROR_H_
#define MSHADOW_EXTENSION_MIRROR_H_
namespace mshadow {
namespace expr {
/*!
 * \brief mirror expression, mirror a image in width
 * \tparam SrcExp source expression to be mirrored
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct MirroringExp:
      public MakeTensorExp<MirroringExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief constructor */
  explicit MirroringExp(const SrcExp &src) : src_(src) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
  }
};
/*!
 * \brief mirroring expression, mirror images in width
 * \param src original image batches
 * \return expression corresponding to mirrored result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline MirroringExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
mirror(const Exp<SrcExp, DType, etype> &src) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MirroringExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self());
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<MirroringExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const MirroringExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), width_(e.shape_[srcdim - 1]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return src_.Eval(i, width_ - j - 1);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_MIRROR_H_
//===== EXPANDED: ../mshadow/mshadow/extension/mirror.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/concat.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file concat.h
 * \brief support for concatenation
 */
#ifndef MSHADOW_EXTENSION_CONCAT_H_
#define MSHADOW_EXTENSION_CONCAT_H_


namespace mshadow {
namespace expr {
/*!
 * \brief concat expression, concat two tensor's channel
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 * \tparam dimsrc_m_cat dimsrc - dimcat
 */
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ConcatExp : public TRValue<ConcatExp<LhsExp, RhsExp,
                                            Device, DType,
                                            srcdim, dimsrc_m_cat>,
                                  Device, srcdim, DType> {
  static const int dimcat = srcdim - dimsrc_m_cat;
  const LhsExp &src1_;
  const RhsExp &src2_;
  index_t dcat_src1_;
  index_t dcat_src2_;
  Shape<4> shape_;
  ConcatExp(const LhsExp &src1, const RhsExp &src2) : src1_(src1), src2_(src2) {
    Shape<srcdim> sshape1 = ShapeCheck<srcdim, LhsExp>::Check(src1_);
    Shape<srcdim> sshape2 = ShapeCheck<srcdim, RhsExp>::Check(src2_);
    #pragma unroll
    for (int i = 0; i < srcdim; ++i) {
      if (i != dimcat) {
        CHECK_EQ(sshape1[i], sshape2[i]) << "ConcatExp: shape mismatch";
      }
    }
    this->shape_ = sshape1;
    this->shape_[dimcat] = sshape1[dimcat] + sshape2[dimcat];
    this->dcat_src1_ = sshape1[dimcat];
    this->dcat_src2_ = sshape2[dimcat];
  }
  template<typename E, int etype>
  inline void
  operator=(const expr::Exp<E, DType, etype> &exp) {
    this->__assign(exp);
  }
  inline void
  operator=(const DType &exp) {
    this->__assign(exp);
  }
};  // struct ConcatExp
/*!
 * \brief concat two 4D tensor
 * \param src1 source tensor1
 * \param src2 source tensor2
 * \return concated 4D tensor
 * \tparam cdim the dimension to concatnate on
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<int cdim, typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
inline ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, srcdim - cdim>
concat(const TRValue<LhsExp, Device, srcdim, DType> &src1,
       const TRValue<RhsExp, Device, srcdim, DType> &src2) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == ExpInfo<RhsExp>::kDim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  TypeCheckPass<cdim < srcdim && ExpInfo<LhsExp>::kDim == srcdim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, srcdim - cdim>
      (src1.self(), src2.self());
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ShapeCheck<srcdim, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  inline static Shape<srcdim> Check(const ConcatExp<LhsExp, RhsExp,
                                    Device, DType, srcdim, dimsrc_m_cat> &t) {
    return t.shape_;
  }
};
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct StreamInfo<Device, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  inline static Stream<Device> *
  Get(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> &t) {
    Stream<Device> *lhs = StreamInfo<Device, LhsExp>::Get(t.src1_);
    Stream<Device> *rhs = StreamInfo<Device, RhsExp>::Get(t.src2_);
    if (lhs != rhs) return NULL;
    return lhs;
  }
};
// static typecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ExpInfo<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  static const int kDimLhs = ExpInfo<LhsExp>::kDim;
  static const int kDimRhs = ExpInfo<RhsExp>::kDim;
  // copy from binarymap
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ?\
      (kDimLhs == 0 ?\
       kDimRhs :\
       ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask & ExpInfo<RhsExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct Plan<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat>, DType> {
 public:
  static const int dimcat = srcdim - dimsrc_m_cat;
  explicit Plan(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> &e)
      : src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
        height_(e.shape_.ProdShape(dimcat + 1, srcdim - 1)),
        ch_src1_(e.dcat_src1_), ch_src2_(e.dcat_src2_), ch_(e.shape_[dimcat]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) {
      return src1_.Eval((b * ch_src1_ + c) * height_ + y, x);
    } else {
      return src2_.Eval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
    }
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) {
      return src1_.REval((b * ch_src1_ + c) * height_ + y, x);
    } else {
      return src2_.REval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
    }
  }

 private:
  Plan<LhsExp, DType> src1_;
  Plan<RhsExp, DType> src2_;
  const index_t height_, ch_src1_, ch_src2_, ch_;
};  // struct Plan

// specialize for concat in x
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim>
struct Plan<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, 1>, DType> {
 public:
  explicit Plan(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, 1> &e)
      : src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
        width_src1_(e.dcat_src1_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    if (x < width_src1_) {
      return src1_.Eval(y, x);
    } else {
      return src2_.Eval(y, x - width_src1_);
    }
  }
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    if (x < width_src1_) {
      return src1_.REval(y, x);
    } else {
      return src2_.REval(y, x - width_src1_);
    }
  }

 private:
  Plan<LhsExp, DType> src1_;
  Plan<RhsExp, DType> src2_;
  const index_t width_src1_;
};
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_CONCAT_H_
//===== EXPANDED: ../mshadow/mshadow/extension/concat.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/choose.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file choose.h
 * \brief support for implicit array selection operation
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHOOSE_H_
#define MSHADOW_EXTENSION_CHOOSE_H_


namespace mshadow {
namespace expr {
/*!
 * \brief Make a choice of index in the lowest changing dimension.
 * \tparam SrcExp type of lhs expression
 * \tparam IndexExp type of index expression
 * \tparam DType the type of elements
 */
template<typename SrcExp, typename IndexExp, typename DType>
struct MatChooseRowElementExp:
      public Exp<MatChooseRowElementExp<SrcExp, IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief constructor */
  MatChooseRowElementExp(const SrcExp &src, const IndexExp &index)
      : src_(src), index_(index) {}
};

template<typename SrcExp, typename IndexExp,
         typename DType, typename IDType, int e1, int e2>
inline MatChooseRowElementExp<SrcExp, IndexExp, DType>
mat_choose_row_element(const Exp<SrcExp, DType, e1> &src,
                       const Exp<IndexExp, IDType, e2> &index) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2 && ExpInfo<IndexExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MatChooseRowElementExp<SrcExp, IndexExp, DType>(src.self(), index.self());
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename IndexExp, typename DType>
struct Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType> {
 public:
  explicit Plan(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &e)
      : src_(MakePlan(e.src_)),
        index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, x));
    return src_.Eval(x, idx);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<IndexExp, DType> index_;
};

template<typename SrcExp, typename IndexExp, typename DType>
inline Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType>
MakePlan(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &exp) {
  return Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType>(exp);
}

template<int dim, typename SrcExp, typename IndexExp, typename DType>
struct ShapeCheck<dim, MatChooseRowElementExp<SrcExp, IndexExp, DType> > {
  inline static Shape<dim>
  Check(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &t) {
    CHECK(dim == 1)
        << "MatChooseRowElementExp only support 1 dimension output";
    Shape<2> shape1 = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> shape2 = ShapeCheck<dim, IndexExp>::Check(t.index_);
    CHECK_EQ(shape1[0], shape2[0])
        << "mat_choose_row_element index length and number of rows in matrix";
    return shape2;
  }
};

template<typename SrcExp, typename IndexExp, typename DType>
struct ExpInfo<MatChooseRowElementExp<SrcExp, IndexExp, DType> > {
  static const int kDim = 1;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask & ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHOOSE_H_
//===== EXPANDED: ../mshadow/mshadow/extension/choose.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/fill.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file fill.h
 * \brief support for implicit array filling operation
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_FILL_H_
#define MSHADOW_EXTENSION_FILL_H_



namespace mshadow {
namespace expr {
/*!
 * \brief Set value of a specific element in each line of the data matrix.
 * \tparam SrcExp type of src expression
 * \tparam ValExp type of val expression
 * \tparam IndexExp type of index expression
 * \tparam DType the type of ret expression
 */
template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct MatFillRowElementExp:
      public Exp<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief src operand */
  const SrcExp &src_;
  const ValExp &val_;
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief constructor */
  MatFillRowElementExp(const SrcExp &src, const ValExp &val, const IndexExp &index)
      : src_(src), val_(val), index_(index) {}
};

template<typename SrcExp, typename ValExp, typename IndexExp,
        typename SDType, typename VDType, typename IDType, int e1, int e2, int e3>
inline MatFillRowElementExp<SrcExp, ValExp, IndexExp, SDType>
mat_fill_row_element(const Exp<SrcExp, SDType, e1> &src,
                     const Exp<ValExp, VDType, e2> &val,
                     const Exp<IndexExp, IDType, e3> &index) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2 && ExpInfo<ValExp>::kDim == 1
                && ExpInfo<IndexExp>::kDim == 1>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MatFillRowElementExp<SrcExp, ValExp, IndexExp, SDType>(src.self(),
                                                                val.self(), index.self());
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct Plan<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>, DType> {
 public:
  explicit Plan(const MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> &e)
      : src_(MakePlan(e.src_)),
        val_(MakePlan(e.val_)),
        index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    if (idx == x) {
      return static_cast<DType>(val_.Eval(0, y));
    } else {
      return static_cast<DType>(src_.Eval(y, x));
    }
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<ValExp, DType> val_;
  expr::Plan<IndexExp, DType> index_;
};

template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
inline Plan<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>, DType>
MakePlan(const MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> &exp) {
  return Plan<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>, DType>(exp);
}

template<int dim, typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct ShapeCheck<dim, MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> > {
  inline static Shape<dim>
  Check(const MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> &t) {
    CHECK(dim == 2)
        << "MatFillRowElementExp only support 2 dimension output";
    Shape<2> shape_src = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<1> shape_val = ShapeCheck<1, ValExp>::Check(t.val_);
    Shape<1> shape_index = ShapeCheck<1, IndexExp>::Check(t.index_);
    CHECK((shape_src[0] == shape_index[0]) && (shape_index[0] == shape_val[0]))
        << "mat_fill_row_element index length, val length and number of rows in matrix";
    return shape_src;
  }
};

template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct ExpInfo<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask =
          ExpInfo<SrcExp>::kDevMask & ExpInfo<ValExp>::kDevMask & ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_FILL_H_
//===== EXPANDED: ../mshadow/mshadow/extension/fill.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/one_hot.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file one_hot.h
 * \brief Create one-hot indicator array based on the index.
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_ONE_HOT_H_
#define MSHADOW_EXTENSION_ONE_HOT_H_



namespace mshadow {
namespace expr {
/*!
 * \brief Create a one-hot indicator array.
 * \tparam IndexExp type of index expression
 * \tparam DType the type of elements
 */
template<typename IndexExp, typename DType>
struct OneHotEncodeExp:
      public Exp<OneHotEncodeExp<IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief number of choices we can have. */
  index_t num_choices_;
  /*! \brief constructor */
  OneHotEncodeExp(const IndexExp &index, index_t num_choices)
      : index_(index), num_choices_(num_choices) {}
};

template<typename IndexExp,
         typename IDType, int e1>
inline OneHotEncodeExp<IndexExp, default_real_t>
one_hot_encode(const Exp<IndexExp, IDType, e1> &index, index_t num_choices) {
  TypeCheckPass<ExpInfo<IndexExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return OneHotEncodeExp<IndexExp, default_real_t>(index.self(), num_choices);
}

//----------------------
// Execution plan
//----------------------
template<typename IndexExp, typename DType>
struct Plan<OneHotEncodeExp<IndexExp, DType>, DType> {
 public:
  explicit Plan(const OneHotEncodeExp<IndexExp, DType> &e)
      : index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    return static_cast<DType>(x == idx);
  }

 private:
  expr::Plan<IndexExp, DType> index_;
};

template<typename IndexExp, typename DType>
inline Plan<OneHotEncodeExp<IndexExp, DType>, DType>
MakePlan(const OneHotEncodeExp<IndexExp, DType> &exp) {
  return Plan<OneHotEncodeExp<IndexExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename DType>
struct ShapeCheck<dim, OneHotEncodeExp<IndexExp, DType> > {
  inline static Shape<dim>
  Check(const OneHotEncodeExp<IndexExp, DType> &t) {
    CHECK(dim == 2)
        << "OneHotEncodeExp only support 2 dimension output";
    Shape<1> shape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<dim> ret;
    ret[0] = shape[0];
    ret[1] = t.num_choices_;
    return ret;
  }
};

template<typename IndexExp, typename DType>
struct ExpInfo<OneHotEncodeExp<IndexExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_ONE_HOT_H_
//===== EXPANDED: ../mshadow/mshadow/extension/one_hot.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/slice.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file slice.h
 * \brief support for slice a certain dimension.
 */
#ifndef MSHADOW_EXTENSION_SLICE_H_
#define MSHADOW_EXTENSION_SLICE_H_


namespace mshadow {
namespace expr {
/*!
 * \brief slice expression, slice a tensor's channel
 * \tparam SrcExp left expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 * \tparam dimsrc_m_cat dimsrc - dimcat
 */
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct SliceExp : public TRValue<SliceExp<SrcExp,
                                          Device, DType,
                                          srcdim, dimsrc_m_slice>,
                                 Device, srcdim, DType> {
  static const int dimslice = srcdim - dimsrc_m_slice;
  const SrcExp &src_;
  index_t ch_begin_;
  index_t ch_old_;
  Shape<srcdim> shape_;
  SliceExp(const SrcExp &src, index_t begin, index_t end)
      : src_(src), ch_begin_(begin) {
    shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    ch_old_ = shape_[dimslice];
    CHECK(begin < shape_[dimslice] && end <= shape_[dimslice])
        << "The slice went out of range";
    shape_[dimslice] = end - begin;
  }
  template<typename E, int etype>
  inline void
  operator=(const expr::Exp<E, DType, etype> &exp) {
    this->__assign(exp);
  }
  inline void
  operator=(const DType &exp) {
    this->__assign(exp);
  }
};  // struct Slice

/*!
 * \brief Slice a Tensor
 * \param src source tensor
 * \param begin The beginning slice.
 * \param end The end slice.
 * \return sliced tensor
 * \tparam sdim the dimension to slice on
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<int sdim, typename SrcExp,
         typename Device, typename DType, int srcdim>
inline SliceExp<SrcExp, Device, DType, srcdim, srcdim - sdim>
slice(const TRValue<SrcExp, Device, srcdim, DType> &src, index_t begin, index_t end) {
  TypeCheckPass<sdim < srcdim && ExpInfo<SrcExp>::kDim == srcdim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SliceExp<SrcExp, Device, DType, srcdim, srcdim - sdim>(src.self(), begin, end);
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct ShapeCheck<srcdim, SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  inline static Shape<srcdim> Check(const SliceExp<SrcExp,
                                    Device, DType, srcdim, dimsrc_m_slice> &t) {
    return t.shape_;
  }
};
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct StreamInfo<Device, SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  inline static Stream<Device> *
  Get(const SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> &t) {
    return StreamInfo<Device, SrcExp>::Get(t.src_);
  }
};
// static typecheck
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct ExpInfo<SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  static const int kDim = ExpInfo<SrcExp>::kDim;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct Plan<SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice>, DType> {
 public:
  static const int dimslice = srcdim - dimsrc_m_slice;
  explicit Plan(const SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> &e)
      : src_(MakePlan(e.src_)),
        height_(e.shape_.ProdShape(dimslice + 1, srcdim - 1)),
        ch_begin_(e.ch_begin_), ch_old_(e.ch_old_), ch_(e.shape_[dimslice]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_ + ch_begin_;
    const index_t b = i / ch_;
    const index_t x = j;
    return src_.Eval((b * ch_old_ + c) * height_ + y, x);
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_ + ch_begin_;
    const index_t b = i / ch_;
    const index_t x = j;
    return src_.REval((b * ch_old_ + c) * height_ + y, x);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t height_, ch_begin_, ch_old_, ch_;
};  // struct Plan

template<typename SrcExp,
         typename Device, typename DType,
         int srcdim>
struct Plan<SliceExp<SrcExp, Device, DType, srcdim, 1>, DType> {
 public:
  explicit Plan(const SliceExp<SrcExp, Device, DType, srcdim, 1> &e)
      : src_(MakePlan(e.src_)),
        ch_begin_(e.ch_begin_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(y, x + ch_begin_);
  }
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return src_.REval(y, x + ch_begin_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t ch_begin_;
};
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_SLICE_H_
//===== EXPANDED: ../mshadow/mshadow/extension/slice.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/take.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file take.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_TAKE_H_
#define MSHADOW_EXTENSION_TAKE_H_


namespace mshadow {
namespace expr {

/*! \brief Take a column from a matrix
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct TakeExp: public Exp<TakeExp<IndexExp, SrcExp, DType>,
                           DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief embediing oprand */
  const SrcExp &src_;
  /*! constructor */
  TakeExp(const IndexExp &index, const SrcExp &src)
    : index_(index), src_(src) {}
};  // struct TakeExp



template<typename IndexExp,
         typename SrcExp,
         typename DType,
         int e1, int e2>
inline TakeExp<IndexExp, SrcExp, default_real_t>
take(const Exp<IndexExp, DType, e1> &index,
     const Exp<SrcExp, DType, e2> &src) {
  return TakeExp<IndexExp, SrcExp, default_real_t>(index.self(), src.self());
}


//----------------------
// Execution plan
//----------------------

template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<TakeExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const TakeExp<IndexExp, SrcExp, DType> &e)
    : index_(MakePlan(e.index_)), src_(MakePlan(e.src_)) {
  }

  // TODO(xx): discuss W shape: in * out or out * in
  // Now I use in * out
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    return static_cast<DType>(src_.Eval(idx, x));
  }

 private:
  expr::Plan<IndexExp, DType> index_;
  expr::Plan<SrcExp, DType> src_;
};  // struct Plan

template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<TakeExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const TakeExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<TakeExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, TakeExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const TakeExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 2)
      << "TakeExp only support 2D output";
    Shape<1> dshape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<2> wshape = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> ret;
    ret[0] = dshape[0];
    ret[1] = wshape[1];
    return ret;
  }
};


template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<TakeExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow

#endif  // MSHADOW_EXTENSION_TAKE_H_
//===== EXPANDED: ../mshadow/mshadow/extension/take.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/take_grad.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file take_grad.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_TAKE_GRAD_H_
#define MSHADOW_EXTENSION_TAKE_GRAD_H_


namespace mshadow {
namespace expr {

/*! \brief Calculate embedding gradient
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */

template<typename IndexExp, typename SrcExp, typename DType>
struct TakeGradExp : public Exp<TakeGradExp<IndexExp, SrcExp, DType>,
                                DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief out gradient oprand */
  const SrcExp &src_;
  /*! \brief batch size */
  const index_t input_dim_;
  /*! \brief constructor */
  TakeGradExp(const IndexExp &index, const SrcExp &src, const index_t input_dim)
    : index_(index), src_(src), input_dim_(input_dim) {}
};  // struct TakeGradExp


template<typename IndexExp,
         typename SrcExp,
         typename DType,
         int e1, int e2>
inline TakeGradExp<IndexExp, SrcExp, default_real_t>
take_grad(const Exp<IndexExp, DType, e1> &index,
          const Exp<SrcExp, DType, e2> &src,
          const index_t input_dim) {
  return TakeGradExp<IndexExp, SrcExp, default_real_t>(index.self(),
                                                       src.self(),
                                                       input_dim);
}

//----------------------
// Execution plan
//----------------------

template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const TakeGradExp<IndexExp, SrcExp, DType> &e)
    : index_(MakePlan(e.index_)),
      src_(MakePlan(e.src_)),
      batch_size_(ShapeCheck<1, IndexExp>::Check(e.index_)[0]) {
  }

  // now return shape: in * out
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    DType ret = 0.f;
    for (index_t i = 0; i < batch_size_; ++i) {
      index_t idx = static_cast<index_t>(index_.Eval(0, i));
      if (idx == y) {
        ret += static_cast<DType>(src_.Eval(i, x));
      }
    }
    return ret;
  }

 private:
  expr::Plan<IndexExp, DType> index_;
  expr::Plan<SrcExp, DType> src_;
  const index_t batch_size_;
};  // struct Plan


template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const TakeGradExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, TakeGradExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const TakeGradExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 2)
      << "TakeGradExp only support 2D output";
    // Shape<1> dshape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<2> gshape = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> ret;
    ret[0] = t.input_dim_;
    ret[1] = gshape[1];
    return ret;
  }
};  // struct ShapeCheck

template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<TakeGradExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow

#endif  // MSHADOW_EXTENSION_TAKE_GRAD_H_
//===== EXPANDED: ../mshadow/mshadow/extension/take_grad.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/reduce_with_axis.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file reduce_with_axis.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
#define MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_


namespace mshadow {
namespace expr {

/*! \brief reduce out the dimension of src labeled by axis.
 *  \tparam Reducer type of reducer
 *  \tparam SrcExp type of source expression
 *  \tparam DType data type
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim, bool mask>
struct ReduceWithAxisExp:
    public MakeTensorExp<ReduceWithAxisExp<Reducer, SrcExp, DType, srcdim, mask>,
                         SrcExp, srcdim-1, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief size of leading dimensions */
  index_t leading_;
  /*! \brief size of trailing dimensions */
  index_t trailing_;
  /*! \brief size of axis dimension */
  index_t size_;
  /*! \brief size of last src dimension */
  index_t last_;
  /*! constructor */
  explicit ReduceWithAxisExp(const SrcExp &src, int axis)
    : src_(src) {
    CHECK(srcdim > axis) << "reduce axis out of bound";
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->leading_ = 1;
    for (index_t i = 0; i < axis; ++i) {
      this->leading_ *= src_shape[i];
      this->shape_[i] = src_shape[i];
    }
    this->size_ = src_shape[axis];
    this->trailing_ = 1;
    for (index_t i = axis + 1; i < srcdim; ++i) {
      this->trailing_ *= src_shape[i];
      this->shape_[i-1] = src_shape[i];
    }
    this->last_ = src_shape[srcdim-1];
  }
};  // struct ReduceWithAxisExp

/*!
 * \brief pooling subregion results together
 * \param lhs left oprand
 * \param rhs right oprand
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the content data type
 */
template<typename Reducer, bool mask, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask>
reduce_with_axis(const Exp<SrcExp, DType, etype> &src, int axis) {
  return ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask>(src.self(), axis);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim, bool mask>
struct Plan<ReduceWithAxisExp<Reducer, SrcExp, DType, srcdim, mask>, DType> {
 public:
  explicit Plan(const ReduceWithAxisExp<Reducer, SrcExp, DType, srcdim, mask> &e)
      : src_(MakePlan(e.src_)), leading_(e.leading_), trailing_(e.trailing_),
        size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_ + j)/trailing_;
    index_t y = (i*last_ + j)%trailing_;

    if (mask) {
      index_t idx = 0;
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        index_t z = (x*size_+k)*trailing_+y;
        DType tmp = res;
        Reducer::Reduce(res, src_.Eval(z/last_, z%last_));
        if (tmp != res) {
          idx = k;
        }
      }
      return static_cast<DType>(idx);
    } else {
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        index_t z = (x*size_+k)*trailing_+y;
        Reducer::Reduce(res, src_.Eval(z/last_, z%last_));
      }
      return res;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t leading_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
//===== EXPANDED: ../mshadow/mshadow/extension/reduce_with_axis.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/broadcast_with_axis.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file tensor_dot.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
#define MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_


namespace mshadow {
namespace expr {

/*! \brief Backward for tensor dot
 *  \tparam DataExp type of left expression
 *  \tparam TopExp type of right expression
 *  \tparam DType data type
 */
template<typename SrcExp, typename DType, int srcdim>
struct BroadcastWithAxisExp:
    public MakeTensorExp<BroadcastWithAxisExp<SrcExp, DType, srcdim>,
                         SrcExp, srcdim+1, DType> {
  /*! \brief data oprand */
  const SrcExp &src_;
  /*! \brief size of middle dimension */
  index_t leading_;
  /*! \brief size of middle dimension */
  index_t trailing_;
  /*! \brief size of middle dimension */
  index_t size_;
  /*! \brief size of middle dimension */
  index_t last_;
  /*! constructor */
  BroadcastWithAxisExp(const SrcExp &src, const int axis, const index_t size)
    : src_(src), size_(size) {
    CHECK(srcdim > axis) << "broadcast axis out of bound";
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->leading_ = 1;
    for (index_t i = 0; i <= axis; ++i) {
      this->leading_ *= src_shape[i];
      this->shape_[i] = src_shape[i];
    }
    this->shape_[axis+1] = size_;
    this->trailing_ = 1;
    for (index_t i = axis+1; i < srcdim; ++i) {
      this->trailing_ *= src_shape[i];
      this->shape_[i+1] = src_shape[i];
    }
    this->last_ = src_shape[srcdim-1];
  }
};  // struct BroadcastWithAxisExp

/*!
 * \brief pooling subregion results together
 * \param data data oprand
 * \param top top grad oprand
 * \tparam DataExp left expression
 * \tparam TopExp right expression
 * \tparam DType the content data type
 */
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
broadcast_with_axis(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axis, size);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<BroadcastWithAxisExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const BroadcastWithAxisExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), leading_(e.leading_),
        trailing_(e.trailing_), size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_+j)/trailing_/size_;
    index_t y = (i*last_+j)%trailing_;
    index_t z = x*trailing_ + y;
    return src_.Eval(z/last_, z%last_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t leading_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
//===== EXPANDED: ../mshadow/mshadow/extension/broadcast_with_axis.h =====

//===== EXPANDIND: ../mshadow/mshadow/extension/spatial_upsampling_nearest.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file spatial_upsampling.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#define MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_

namespace mshadow {
namespace expr {

/*! \brief nearest neighboor upsampling
 *         out(x, y) = in(int(x / scale_x), int(y / scale_y))
 *  \tparam SrcExp source expression
 *  \tparam DType data type
 *  \tparam srcdim source dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UpSamplingNearestExp :
  public MakeTensorExp<UpSamplingNearestExp<SrcExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief up sampling scale */
  index_t scale_;
  /*! \brief constructor */
  UpSamplingNearestExp(const SrcExp &src, index_t scale)
    : src_(src), scale_(scale) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_[srcdim - 2] *= scale_;
    this->shape_[srcdim - 1] *= scale_;
  }
};


template<typename SrcExp, typename DType, int etype>
inline UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
upsampling_nearest(const Exp<SrcExp, DType, etype> &src, index_t scale) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
    ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), scale);
}

template<typename SrcExp, typename DType, int srcdim>
struct Plan<UpSamplingNearestExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UpSamplingNearestExp<SrcExp, DType, srcdim> &e)
    : src_(MakePlan(e.src_)),
      scale_(e.scale_),
      new_height_(e.shape_[srcdim - 2]),
      src_height_(static_cast<index_t>(e.shape_[srcdim - 2] / e.scale_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = static_cast<index_t>(y / scale_);
    const index_t w = static_cast<index_t>(x / scale_);
    return src_.Eval(c * src_height_ + h, w);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t scale_;
  const index_t new_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
//===== EXPANDED: ../mshadow/mshadow/extension/spatial_upsampling_nearest.h =====

#endif  // MSHADOW_EXTENSION_H_
//===== EXPANDED: ../mshadow/mshadow/extension.h =====

//===== EXPANDIND: ../mshadow/mshadow/tensor_cpu-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_cpu-inl.h
 * \brief implementation of CPU host code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_CPU_INL_H_
#define MSHADOW_TENSOR_CPU_INL_H_

namespace mshadow {
template<>
inline void InitTensorEngine<cpu>(int dev_id) {
}
template<>
inline void ShutdownTensorEngine<cpu>(void) {
}

template<>
inline void SetDevice<cpu>(int devid) {
}
template<>
inline Stream<cpu> *NewStream<cpu>(bool create_blas_handle,
                                   bool create_dnn_handle) {
  return new Stream<cpu>();
}
template<>
inline void DeleteStream<cpu>(Stream<cpu> *stream) {
  delete stream;
}

template<int ndim>
inline std::ostream &operator<<(std::ostream &os, const Shape<ndim> &shape) { // NOLINT(*)
  os << "(";
  for (int i = 0; i < ndim; ++i) {
    if (i != 0) os << ",";
    os << shape[i];
  }
  os << ")";
  return os;
}

template<typename xpu>
inline void *AllocHost_(size_t size);
template<typename xpu>
inline void FreeHost_(void * dptr);

#ifdef __CUDACC__
template<>
inline void *AllocHost_<gpu>(size_t size) {
  void *dptr;
  MSHADOW_CUDA_CALL(cudaMallocHost(&dptr, size, cudaHostAllocPortable));
  return dptr;
}
template<>
inline void FreeHost_<gpu>(void *dptr) {
  MSHADOW_CUDA_CALL(cudaFreeHost(dptr));
}
#endif

template<>
inline void *AllocHost_<cpu>(size_t size) {
  size_t pitch;
  return packet::AlignedMallocPitch(&pitch, size, 1);
}
template<>
inline void FreeHost_<cpu>(void *dptr) {
  packet::AlignedFree(dptr);
}

template<typename xpu, int dim, typename DType>
inline void AllocHost(Tensor<cpu, dim, DType> *obj) {
  obj->stride_ = obj->size(dim - 1);
  CHECK_EQ(obj->CheckContiguous(), true) << "AllocHost";
  void *dptr = AllocHost_<xpu>(obj->MSize() * sizeof(DType));
  obj->dptr_ = reinterpret_cast<DType*>(dptr);
}
template<typename xpu, int dim, typename DType>
inline void FreeHost(Tensor<cpu, dim, DType> *obj) {
  if (obj->dptr_ == NULL) {
    LOG(FATAL) << "FreeHost:: double free";
  }
  FreeHost_<xpu>(obj->dptr_);
  obj->dptr_ = NULL;
}

template<int dim, typename DType>
inline void AllocSpace(Tensor<cpu, dim, DType> *obj, bool pad) {
  size_t pitch;
  void *dptr;
  if (pad) {
    dptr = packet::AlignedMallocPitch
        (&pitch, obj->size(dim - 1) * sizeof(DType), obj->shape_.FlatTo2D()[0]);
    obj->stride_ = static_cast<index_t>(pitch / sizeof(DType));
  } else {
    obj->stride_ = obj->size(dim - 1);
    dptr = packet::AlignedMallocPitch
        (&pitch, obj->shape_.Size() * sizeof(DType), 1);
  }
  obj->dptr_ = reinterpret_cast<DType*>(dptr);
}
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType>
NewTensor(const Shape<dim> &shape, DType initv, bool pad, Stream<Device> *stream_) {
  Tensor<Device, dim, DType> obj(shape);
  obj.stream_ = stream_;
  AllocSpace(&obj, pad);
  MapExp<sv::saveto>(&obj, expr::ScalarExp<DType>(initv));
  return obj;
}
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> *obj) {
  packet::AlignedFree(obj->dptr_);
  obj->dptr_ = NULL;
}
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> _dst,
                 const Tensor<cpu, dim, DType> &_src,
                 Stream<cpu> *stream) {
  CHECK_EQ(_dst.shape_, _src.shape_)
      << "Copy:shape mismatch:" << _dst.shape_ << " vs " << _src.shape_;
  if (_dst.CheckContiguous() && _src.CheckContiguous()) {
    memcpy(_dst.dptr_, _src.dptr_, sizeof(DType) * _dst.shape_.Size());
  } else {
    Tensor<cpu, 2, DType> dst = _dst.FlatTo2D();
    Tensor<cpu, 2, DType> src = _src.FlatTo2D();
    for (index_t y = 0; y < dst.size(0); ++y) {
      memcpy(dst[y].dptr_, src[y].dptr_, sizeof(DType) * dst.size(1));
    }
  }
}

template<typename Saver, typename R, int dim,
         typename DType, typename E>
inline void MapPlan(TRValue<R, cpu, dim, DType> *dst,
                    const expr::Plan<E, DType> &plan) {
  Shape<2> shape = expr::ShapeCheck<dim, R>::Check(dst->self()).FlatTo2D();
  expr::Plan<R, DType> dplan = expr::MakePlan(dst->self());
  for (index_t y = 0; y < shape[0]; ++y) {
    for (index_t x = 0; x < shape[1]; ++x) {
      // trust your compiler! -_- they will optimize it
      Saver::Save(dplan.REval(y, x), plan.Eval(y, x));
    }
  }
}
// code to handle SSE optimization
template<bool pass_check, typename Saver,
         typename R, int dim,
         typename DType, typename E, int etype>
struct MapExpCPUEngine {
  inline static void Map(TRValue<R, cpu, dim, DType> *dst,
                         const expr::Exp<E, DType, etype> &exp) {
    MapPlan<Saver>(dst, MakePlan(exp.self()));
  }
};

template<typename SV, int dim, typename DType, typename E, int etype>
struct MapExpCPUEngine<true, SV, Tensor<cpu, dim, DType>,
                       dim, DType, E, etype> {
  inline static void Map(Tensor<cpu, dim, DType> *dst,
                         const expr::Exp<E, DType, etype> &exp) {
    if (expr::PacketAlignCheck<dim, E, MSHADOW_DEFAULT_PACKET>::Check(exp.self()) &&
        expr::PacketAlignCheck<dim, Tensor<cpu, dim, DType>, MSHADOW_DEFAULT_PACKET>::Check(*dst)) {
      expr::MapPacketPlan<SV>(dst->self(),
                              expr::MakePacketPlan<MSHADOW_DEFAULT_PACKET>(exp.self()));
    } else {
      MapPlan<SV>(dst, MakePlan(exp.self()));
    }
  }
};


template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, cpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, dim, DType, E>::kMapPass>
      ::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
  Shape<dim> eshape = expr::ShapeCheck<dim, E>::Check(exp.self());
  Shape<dim> dshape = expr::ShapeCheck<dim, R>::Check(dst->self());
  CHECK(eshape[0] == 0 || eshape == dshape)
      << "Assignment: Shape of Tensors are not consistent with target";
  MapExpCPUEngine<expr::PacketCheck<E, MSHADOW_DEFAULT_PACKET>::kPass,
                  Saver, R, dim, DType, E, etype>
  ::Map(dst->ptrself(), exp);
}

template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, cpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, 1, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  Shape<2> eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self()).FlatTo2D();
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[1], dshape[0]) << "MapReduceKeepLowest::reduction dimension do not match";
  CHECK_NE(eshape[0], 0) << "can not reduce over empty tensor";
  // execution
  expr::Plan<R, DType> dplan = MakePlan(dst->self());
  expr::Plan<E, DType> splan = MakePlan(exp.self());
  for (index_t x = 0; x < eshape[1]; ++x) {
    DType res = splan.Eval(0, x);
    for (index_t y = 1; y < eshape[0]; ++y) {
      Reducer::Reduce(res, splan.Eval(y, x));
    }
    Saver::Save(dplan.REval(0, x), res * scale);
  }
}

template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, cpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, dimkeep, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  typedef Shape<expr::ExpInfo<E>::kDim> EShape;
  EShape eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self());
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[dimkeep], dshape[0])
    << "MapReduceKeepHighDim::reduction dimension do not match";
  // use equvalent form
  Shape<4> pshape = Shape4(eshape.ProdShape(0, dimkeep),
                           eshape[dimkeep],
                           eshape.ProdShape(dimkeep + 1, EShape::kSubdim),
                           eshape[EShape::kSubdim]);
  // execution
  expr::Plan<R, DType> dplan = MakePlan(dst->self());
  expr::Plan<E, DType> splan = MakePlan(exp.self());
  for (index_t c = 0; c < pshape[1]; ++c) {
    DType res; Reducer::SetInitValue(res);
    for (index_t n = 0; n < pshape[0]; ++n) {
      DType tres; Reducer::SetInitValue(tres);
      for (index_t y = 0; y < pshape[2]; ++y) {
        for (index_t x = 0; x < pshape[3]; ++x) {
          Reducer::Reduce(tres,
                          splan.Eval((n * pshape[1] + c) * pshape[2] + y, x));
        }
      }
      Reducer::Reduce(res, tres);
    }
    Saver::Save(dplan.REval(0, c), res * scale);
  }
}

template<typename DType>
inline void Softmax(Tensor<cpu, 1, DType> dst,
                    const Tensor<cpu, 1, DType> &energy) {
  DType mmax = energy[0];
  for (index_t x = 1; x < dst.size(0); ++x) {
    if (mmax < energy[x]) mmax = energy[x];
  }
  DType sum = 0.0f;
  for (index_t x = 0; x < dst.size(0); ++x) {
    dst[x] = std::exp(energy[x] - mmax);
    sum += dst[x];
  }
  for (index_t x = 0; x < dst.size(0); ++x) {
    dst[x] /= sum;
  }
}

template<typename DType>
inline void SoftmaxGrad(Tensor<cpu, 2, DType> dst,
                        const Tensor<cpu, 2, DType> &src,
                        const Tensor<cpu, 1, DType> &label) {
  for (index_t y = 0; y < dst.size(0); ++y) {
    const index_t k = static_cast<int>(label[y]);
    for (index_t x = 0; x < dst.size(1); ++x) {
      if (x == k) {
        dst[y][k] = src[y][k] - 1.0f;
      } else {
        dst[y][x] = src[y][x];
      }
    }
  }
}

template<typename DType>
inline void SoftmaxGrad(Tensor<cpu, 3, DType> dst,
                        const Tensor<cpu, 3, DType> &src,
                        const Tensor<cpu, 2, DType> &label) {
  for (index_t n = 0; n < dst.size(2); ++n) {
    for (index_t y = 0; y < dst.size(0); ++y) {
      const index_t k = static_cast<int>(label[y][n]);
      for (index_t x = 0; x < dst.size(1); ++x) {
        if (x == k) {
          dst[y][k][n] = src[y][k][n] - 1.0f;
        } else {
          dst[y][x][n] = src[y][x][n];
        }
      }
    }
  }
}

template<typename DType>
inline void SoftmaxGrad(Tensor<cpu, 3, DType> dst,
                        const Tensor<cpu, 3, DType> &src,
                        const Tensor<cpu, 2, DType> &label,
                        const DType &ignore_label) {
  for (index_t n = 0; n < dst.size(2); ++n) {
    for (index_t y = 0; y < dst.size(0); ++y) {
      const index_t k = static_cast<int>(label[y][n]);
      if (k == static_cast<int>(ignore_label)) {
        for (index_t x = 0; x < dst.size(1); ++x) {
          dst[y][x][n] = 0.0f;
        }
      } else {
        for (index_t x = 0; x < dst.size(1); ++x) {
          if (x == k) {
            dst[y][k][n] = src[y][k][n] - 1.0f;
          } else {
            dst[y][x][n] = src[y][x][n];
          }
        }
      }
    }
  }
}

template<typename DType>
inline void Softmax(Tensor<cpu, 2, DType> dst,
                    const Tensor<cpu, 2, DType> &energy) {
  CHECK_EQ(dst.shape_, energy.shape_) << "Softmax: shape mismatch";
  for (index_t y = 0; y < dst.size(0); ++y) {
    Softmax(dst[y], energy[y]);
  }
}

template<typename DType>
inline void Softmax(Tensor<cpu, 3, DType> dst,
                    const Tensor<cpu, 3, DType> &energy) {
  CHECK_EQ(dst.shape_, energy.shape_) << "Softmax: shape mismatch";
  for (index_t y = 0; y < dst.size(0); ++y) {
    for (index_t n = 0; n < dst.size(2); ++n) {
      DType mmax = energy[y][0][n];
      for (index_t x = 1; x < dst.size(1); ++x) {
        if (mmax < energy[y][x][n]) mmax = energy[y][x][n];
      }
      DType sum = 0.0f;
      for (index_t x = 0; x < dst.size(1); ++x) {
        dst[y][x][n] = std::exp(energy[y][x][n] - mmax);
        sum += dst[y][x][n];
      }
      for (index_t x = 0; x < dst.size(1); ++x) {
        dst[y][x][n] /= sum;
      }
    }
  }
}

// blas related
template<typename Device, typename DType>
inline void VectorDot(Tensor<Device, 1, DType> dst,
                      const Tensor<Device, 1, DType> &lhs,
                      const Tensor<Device, 1, DType> &rhs) {
  CHECK_EQ(lhs.size(0), rhs.size(0))
      << "VectorDot: Shape mismatch";
  CHECK_EQ(dst.size(0), 1)
      << "VectorDot: expect dst to be scalar";
  expr::BLASEngine<Device>::SetStream(lhs.stream_);
  mshadow::expr::BLASEngine<Device>::dot(
      lhs.stream_, lhs.size(0), lhs.dptr_, 1, rhs.dptr_, 1, dst.dptr_);
}
}  // namespace mshadow
#endif  // MSHADOW_TENSOR_CPU_INL_H_
//===== EXPANDED: ../mshadow/mshadow/tensor_cpu-inl.h =====

//===== EXPANDIND: ../mshadow/mshadow/tensor_gpu-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_gpu-inl.h
 * \brief implementation of GPU host code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_GPU_INL_H_
#define MSHADOW_TENSOR_GPU_INL_H_

namespace mshadow {
#if MSHADOW_USE_CUDA
template<>
inline void InitTensorEngine<gpu>(int dev_id) {
  cudaDeviceProp prop;
  int device_id = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  CHECK_GT(device_count, 0) << "Cannot find CUDA device. Please check CUDA-Configuration";
  if (dev_id < 0) {
    device_id = 0;
  } else {
    device_id = dev_id;
  }
  CHECK_LT(device_id, device_count) << "Incorrect Device ID";
  MSHADOW_CUDA_CALL(cudaSetDevice(device_id));
  MSHADOW_CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
}
template<>
inline void ShutdownTensorEngine<gpu>(void) {
}
template<>
inline void SetDevice<gpu>(int devid) {
  MSHADOW_CUDA_CALL(cudaSetDevice(devid));
}
template<int dim, typename DType>
inline void AllocSpace(Tensor<gpu, dim, DType> *obj, bool pad) {
  size_t pitch;
  // common choice for cuda mem align unit is 32
  if (pad && obj->size(dim - 1) >= MSHADOW_MIN_PAD_RATIO * 32) {
    MSHADOW_CUDA_CALL(cudaMallocPitch(reinterpret_cast<void**>(&(obj->dptr_)), &pitch,
                                      obj->size(dim - 1) * sizeof(DType),
                                      obj->shape_.FlatTo2D()[0]));
    obj->stride_ = static_cast<index_t>(pitch / sizeof(DType));
  } else {
    obj->stride_ = obj->size(dim - 1);
    MSHADOW_CUDA_CALL(cudaMallocPitch(reinterpret_cast<void**>(&(obj->dptr_)), &pitch,
                                      obj->shape_.Size() * sizeof(DType), 1));
  }
}
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> *obj) {
  MSHADOW_CUDA_CALL(cudaFree(obj->dptr_));
  obj->dptr_ = NULL;
}
template<typename A, typename B, int dim, typename DType>
inline void Copy(Tensor<A, dim, DType> _dst,
                 Tensor<B, dim, DType> _src,
                 cudaMemcpyKind kind,
                 Stream<gpu> *stream) {
  CHECK_EQ(_dst.shape_, _src.shape_) << "Copy:shape mismatch";
  Tensor<A, 2, DType> dst = _dst.FlatTo2D();
  Tensor<B, 2, DType> src = _src.FlatTo2D();
  MSHADOW_CUDA_CALL(cudaMemcpy2DAsync(dst.dptr_, dst.stride_ * sizeof(DType),
                                      src.dptr_, src.stride_ * sizeof(DType),
                                      dst.size(1) * sizeof(DType),
                                      dst.size(0), kind,
                                      Stream<gpu>::GetStream(stream)));
  // use synchronize call behavior for zero stream
  if (stream == NULL) {
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(0));
  }
}
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyHostToDevice, stream);
}
#endif  // MSHADOW_USE_CUDA
}  // namespace mshadow

// the following part is included only if compiler is nvcc
#ifdef __CUDACC__

namespace mshadow {
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, gpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, dim, DType, E>::kMapPass>
      ::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
  Shape<dim> eshape = expr::ShapeCheck<dim, E>::Check(exp.self());
  Shape<dim> dshape = expr::ShapeCheck<dim, R>::Check(dst->self());
  CHECK(eshape[0] == 0 || eshape == dshape)
    << "Assignment: Shape of Tensors are not consistent with target";
  cuda::MapPlan<Saver>(MakePlan(dst->self()),
                       MakePlan(exp.self()),
                       dshape.FlatTo2D(),
                       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}

template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, gpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, 1, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  Shape<2> eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self()).FlatTo2D();
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[1], dshape[0]) << "MapReduceKeepLowest::reduction dimension do not match";
  CHECK_NE(eshape[0], 0) << "can not reduce over empty tensor";
  cuda::MapReduceKeepLowest<Saver, Reducer>
      (MakePlan(dst->self()), MakePlan(exp.self()), scale, eshape,
       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}

template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, gpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, dimkeep, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  typedef Shape<expr::ExpInfo<E>::kDim> EShape;
  EShape eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self());
    Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[dimkeep], dshape[0]) << "MapReduceKeepHighDim::reduction dimension do not match";
  // use equvalent form
  Shape<4> pshape = Shape4(eshape.ProdShape(0, dimkeep),
                           eshape[dimkeep],
                           eshape.ProdShape(dimkeep + 1, EShape::kSubdim),
                           eshape[EShape::kSubdim]);
  // call equavalent map red dim 2
  cuda::MapReduceKeepDim1<Saver, Reducer>
      (MakePlan(dst->self()), MakePlan(exp.self()), scale, pshape,
       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}
template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> dst,
                    const Tensor<gpu, 2, DType>& src) {
  cuda::Softmax(dst, src);
}

template<typename DType>
inline void Softmax(Tensor<gpu, 3, DType> dst,
                    const Tensor<gpu, 3, DType>& src) {
  cuda::Softmax(dst, src);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label) {
  cuda::SoftmaxGrad(dst, src, label);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label) {
  cuda::SoftmaxGrad(dst, src, label);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const DType &ignore_label) {
  cuda::SoftmaxGrad(dst, src, label, ignore_label);
}

}  // namespace mshadow
#endif  // __CUDACC__
#endif  // MSHADOW_TENSOR_GPU_INL_H_
//===== EXPANDED: ../mshadow/mshadow/tensor_gpu-inl.h =====

//===== EXPANDIND: ../mshadow/mshadow/io.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file io.h
 * \brief definitions of I/O functions for mshadow tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_IO_H_
#define MSHADOW_IO_H_

namespace mshadow {
namespace utils {
/*!
 * \brief interface of stream I/O, used to serialize data,
 *   mshadow does not restricted to only this interface in SaveBinary/LoadBinary
 *   mshadow accept all class that implements Read and Write
 */
class IStream {
 public:
  /*!
   * \brief read data from stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   * \return usually is the size of data readed
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief write data to stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}
};
}  // namespace utils
/*!
 * \brief CPU/GPU: save a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 * \param fo output binary stream
 * \param src source data file
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<cpu, dim, DType> &src);  // NOLINT(*)
/*!
 * \brief CPU/GPU: save a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 * \param fo output binary stream
 * \param src source data file
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<gpu, dim, DType> &src); // NOLINT(*)
/*!
 * \brief CPU/GPU: load a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 *       if pre_alloc is true , then space in dst is preallocated, and must have same shape of the tensor loaded
 *       if pre_alloc is false, then dst originally does not have space allocated, LoadBinary will allocate space for dst
 * \param fi output binary stream
 * \param dst destination file
 * \param pre_alloc whether space is pre-allocated, if false, space allocation will happen
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi,  // NOLINT(*)
                       Tensor<cpu, dim, DType> *dst, bool pre_alloc);
/*!
 * \brief CPU/GPU: load a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 *       if pre_alloc is true , then space in dst is preallocated, and must have same shape of the tensor loaded
 *       if pre_alloc is false, then dst originally does not have space allocated, LoadBinary will allocate space for dst
 * \param fi output binary stream
 * \param dst destination file
 * \param pre_alloc whether space is pre-allocated, if false, space allocation will happen
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */

template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<gpu, dim, DType> *dst, bool pre_alloc);

// implementations
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<cpu, dim, DType> &src_) { // NOLINT(*)
  fo.Write(&src_.shape_, sizeof(src_.shape_));
  Tensor<cpu, 2, DType> src = src_.FlatTo2D();
  for (index_t i = 0; i < src.size(0); ++i) {
    fo.Write(src[i].dptr_, sizeof(DType) * src.size(1));
  }
}
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<gpu, dim, DType> &src) { // NOLINT(*)
  // copy to CPU, then save
  Tensor<cpu, dim, DType> tmp(src.shape_);
  AllocSpace(&tmp);
  Stream<gpu> stream;
  Copy(tmp, src, &stream);
  SaveBinary(fo, tmp);
  FreeSpace(&tmp);
}
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<cpu, dim, DType> *dst_, bool pre_alloc) {
  Shape<dim> shape;
  CHECK_NE(fi.Read(&shape, sizeof(shape)), 0) << "mshadow::LoadBinary";
  if (pre_alloc) {
    CHECK_EQ(shape, dst_->shape_) << "LoadBinary, shape do not match pre-allocated shape";
  } else {
    dst_->shape_ = shape; AllocSpace(dst_);
  }
  Tensor<cpu, 2, DType> dst = dst_->FlatTo2D();
  if (dst.size(0) == 0) return;
  for (index_t i = 0; i < dst.size(0); ++i) {
    CHECK_NE(fi.Read(dst[i].dptr_, sizeof(DType) * dst.size(1)), 0) << "mshadow::LoadBinary";
  }
}
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<gpu, dim, DType> *dst, bool pre_alloc) {
  Tensor<cpu, dim, DType> tmp;
  LoadBinary(fi, &tmp, false);
  if (pre_alloc) {
    CHECK_EQ(tmp.shape, dst->shape_) << "LoadBinary, shape do not match pre-allocated shape";
  } else {
    dst->shape = tmp.shape; AllocSpace(dst);
  }
  Stream<gpu> stream;
  Copy(*dst, tmp, &stream);
  FreeSpace(&tmp);
}
}  // namespace mshadow
#endif  // MSHADOW_IO_H_
//===== EXPANDED: ../mshadow/mshadow/io.h =====

//===== EXPANDIND: ../mshadow/mshadow/tensor_container.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_container.h
 * \brief tensor container that does memory allocation and resize like STL
 * \author Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_CONTAINER_H_
#define MSHADOW_TENSOR_CONTAINER_H_

namespace mshadow {
/*!
 * \brief tensor container that does memory allocation and resize like STL,
 *        use it to save the lines of FreeSpace in class.
 *        Do not abuse it, efficiency can come from pre-allocation and no re-allocation
 *
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 */
template<typename Device, int dimension, typename DType = default_real_t>
class TensorContainer: public Tensor<Device, dimension, DType> {
 public:
  /*!
   * \brief constructor
   * \param pad whether use padding alignment in space allocation
   */
  explicit TensorContainer(bool pad = MSHADOW_ALLOC_PAD) {
    this->pad_ = pad;
    this->dptr_ = data_.dptr_ = NULL;
    this->shape_[0] = 0;
    this->stride_ = 0;
    this->data_.stride_ = 0;
    this->data_.shape_[0] = 0;
  }
  /*!
   * \brief constructor
   * \param shape intial shape
   */
  explicit TensorContainer(const Shape<dimension> &shape) {
    this->pad_ = MSHADOW_ALLOC_PAD;
    data_.dptr_ = NULL;
    this->AllocByShape(shape);
  }
  /*!
   * \brief constructor
   * \param shape intial shape
   * \param initv intial value
   */
  explicit TensorContainer(const Shape<dimension> &shape, DType initv) {
    this->pad_ = MSHADOW_ALLOC_PAD;
    data_.dptr_ = NULL;
    this->AllocByShape(shape);
    (*this) = initv;
  }
  /*!
   * \brief copy constructor
   * \param src source value
   */
  TensorContainer
  (const TensorContainer<Device, dimension, DType> &src)
      : pad_(src.pad_) {
    this->dptr_ = data_.dptr_ = NULL;
    this->shape_[0] = 0;
    this->stride_ = 0;
    this->data_.stride_ = 0;
    this->data_.shape_[0] = 0;
    this->stream_ = src.stream_;
    if (src.dptr_ != NULL) {
      this->AllocByShape(src.shape_);
      mshadow::Copy(*this, src, this->stream_);
    }
  }
  ~TensorContainer(void) {
    this->Release();
  }
  /*!
   * \brief resize the container to given shape, content is NOT preserved
   * \param shape target shape
   */
  inline void Resize(const Shape<dimension> &shape) {
    Shape<2> s2 = shape.FlatTo2D();
    if (s2.shape_[1] > data_.stride_ || s2.shape_[0] > data_.size(0)) {
      this->AllocByShape(shape);
    } else {
      this->shape_ = shape;
      if (this->pad_) {
        this->stride_ = data_.stride_;
      } else {
        this->stride_ = s2.shape_[1];
      }
    }
  }
  /*!
   * \brief resize the container to given shape, and initialize, content is NOT preserved
   * \param shape target shape
   * \param initv initialization value
   */
  inline void Resize(const Shape<dimension> &shape, DType initv) {
    this->Resize(shape);
    (*this) = initv;
  }
  /*! \brief set whether padding is allowed in tensor */
  inline void set_pad(bool pad) {
    this->pad_ = pad;
  }
  /*!
   * \brief save by binary format
   * \param fo output binary stream
   * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
   */
  template<typename TStream>
  inline void SaveBinary(TStream &fo) const { // NOLINT(*)
    mshadow::SaveBinary(fo, *this);
  }
  /*!
   * \brief load by binary format, a temp Tensor<cpu,dim> storage will be allocated
   * \param fi input binary stream
   * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
   */
  template<typename TStream>
  inline void LoadBinary(TStream &fi) { // NOLINT(*)
    Tensor<cpu, dimension, DType> tmp;
    mshadow::LoadBinary(fi, &tmp, false);
    this->Resize(tmp.shape_);
    Stream<Device> stream;
    Copy(*this, tmp, &stream);
    mshadow::FreeSpace(&tmp);
  }
  /*!
   * \brief assign operator from TensorContainer
   * \param src source value
   * \return reference of self
   */
  inline TensorContainer &operator=
  (const TensorContainer<Device, dimension, DType> &src) {
    this->pad_ = src.pad_;
    this->stream_ = src.stream_;
    if (src.dptr_ != NULL) {
      this->Resize(src.shape_);
      mshadow::Copy(*this, src, this->stream_);
    }
    return *this;
  }
  /*!\brief functions to fit expression template */
  inline Tensor<Device, dimension, DType> &operator=(DType s) {
    return this->__assign(s);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &
  operator=(const expr::Exp<E, DType, expr::type::kMapper> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &
  operator=(const expr::Exp<E, DType, expr::type::kChainer> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &
  operator=(const expr::Exp<E, DType, expr::type::kComplex> &exp) {
    return this->__assign(exp);
  }
  /*!
   * \brief Release the llocated space,
   *  The TensorContainer is still functionable,
   *  but will restart allocating space when Resize is called.
   */
  inline void Release(void) {
    if (data_.dptr_ != NULL) {
      mshadow::FreeSpace(&data_);
      this->dptr_ = data_.dptr_ = NULL;
      this->shape_[0] = 0;
      this->stride_ = 0;
      this->data_.stride_ = 0;
      this->data_.shape_[0] = 0;
    }
  }

 private:
  /*! \brief whether we do padding in the space */
  bool pad_;
  /*! \brief the shape of data_ is actually current data space */
  Tensor<Device, 2, DType> data_;

  inline void AllocByShape(const Shape<dimension>& shape) {
    if (data_.dptr_ != NULL) this->Release();
    data_.shape_ = shape.FlatTo2D();
    mshadow::AllocSpace(&data_, pad_);
    this->dptr_ = data_.dptr_;
    this->shape_ = shape;
    if (this->pad_) {
      this->stride_ = data_.stride_;
    } else {
      this->stride_ = data_.size(1);
    }
  }
};
}  // namespace mshadow
#endif  // MSHADOW_TENSOR_CONTAINER_H_
//===== EXPANDED: ../mshadow/mshadow/tensor_container.h =====

//===== EXPANDIND: ../mshadow/mshadow/tensor_blob.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_BLOB_H_
#define MSHADOW_TENSOR_BLOB_H_
namespace mshadow {
/*!
 * \brief dynamic shape class that can hold shape
 *   of arbirary dimension
 */
struct TShape {
 public:
  /*! \brief constructor */
  TShape()
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {}
  /*!
   * \brief constructor from TShape
   * \param s the source shape
   */
  TShape(const TShape &s)
      : ndim_(s.ndim_) {
    if (ndim_ <= kStackCache) {
      data_heap_ = NULL;
      num_heap_allocated_ = 0;
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::copy(s.data_heap_, s.data_heap_ + ndim_, data_heap_);
    }
  }
  /*!
   * \brief construct the TShape from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  TShape(RandomAccessIterator begin,
         RandomAccessIterator end)
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {
    this->CopyFrom(begin, end);
  }
#if MSHADOW_IN_CXX11
  /*!
   * \brief move constructor from TShape
   * \param s the source shape
   */
  TShape(TShape &&s)
      : ndim_(s.ndim_),
        num_heap_allocated_(s.num_heap_allocated_),
        data_heap_(s.data_heap_) {
    if (ndim_ <= kStackCache) {
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    }
    // remove data heap space from s
    s.data_heap_ = NULL;
  }
  /*!
   * \brief move constructor from Shape
   * \param s the source shape
   */
  template<int dim>
  TShape(Shape<dim> &&s)  // NOLINT(*)
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {
    this->CopyFrom(s.shape_, s.shape_ + dim);
  }
#endif
  /*! \brief destructor */
  ~TShape() {
    // data_heap_ can be NULL
    delete [] data_heap_;
  }
  /*!
   * \brief copy shape from content betwen two iterators
   * \param begin the beginning of iterator
   * \param end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline void CopyFrom(RandomAccessIterator begin,
                       RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, data());
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const TShape &shape) {
    this->SetDim(shape.ndim_);
    const index_t *src = shape.data();
    std::copy(src, src + ndim_, data());
    return *this;
  }
  /*!
   * \brief assignment from vector
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const std::vector<index_t> &shape) {
    this->CopyFrom(shape.begin(), shape.end());
    return *this;
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \tparam dim shape dimension
   * \return reference of self
   */
  template<int dim>
  inline TShape &operator=(const Shape<dim> &shape) {
    this->SetDim(dim);
    index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (int i = 0; i < dim; ++i) {
      d[i] = shape[i];
    }
    return *this;
  }
  /*! \return the data content of the shape */
  inline const index_t *data() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data content of the shape */
  inline index_t *data() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \brief return number of dimension of the tensor inside */
  inline index_t ndim(void) const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline index_t &operator[](index_t i) {
    return data()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const index_t &operator[](index_t i) const {
    return data()[i];
  }
  /*! \brief total number of elements in the tensor */
  inline size_t Size(void) const {
    size_t size = 1;
    const index_t *d = this->data();
    for (index_t i = 0; i < ndim_; ++i) {
      size *= d[i];
    }
    return size;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  inline Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    if (ndim_ == 0) return Shape2(0, 0);
    const index_t *d = this->data();
    s.shape_[1] = d[ndim_ - 1];
    index_t ymax = 1;
    for (index_t i = 1; i < ndim_; ++i) {
      ymax *= d[i - 1];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*!
   * \brief get the shape of tensor specifying dim
   * \return the shape requested
   * \tparam dim dimension of the tensor
   */
  template<int dim>
  inline Shape<dim> get(void) const {
    CHECK_EQ(dim, ndim_) << "dimension do not match target dimension " << dim << " vs " << ndim_;
    const index_t *d = this->data();
    Shape<dim> s;
    for (int i = 0; i < dim; ++i) {
      s[i] = d[i];
    }
    return s;
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   */
  inline bool operator==(const TShape &s) const {
    if (ndim_ != s.ndim_) return false;
    if (ndim_ <= kStackCache) {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_stack_[i] != s.data_stack_[i]) return false;
      }
    } else {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_heap_[i] != s.data_heap_[i]) return false;
      }
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   */
  inline bool operator!=(const TShape &s) const {
    return !(*this == s);
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator==(const Shape<dim> &s) const {
    if (ndim_ != dim) return false;
    const index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (index_t i = 0; i < dim; ++i) {
      if (d[i] != s.shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator!=(const Shape<dim> &s) const {
    return !(*this == s);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   */
  template<typename TStream>
  inline void Save(TStream *strm) const {
    strm->Write(&ndim_, sizeof(ndim_));
    strm->Write(data(), sizeof(index_t) * ndim_);
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   * \return whether the load is successful
   */
  template<typename TStream>
  inline bool Load(TStream *strm) {
    if (strm->Read(&ndim_, sizeof(ndim_)) != sizeof(ndim_)) return false;
    this->SetDim(ndim_);
    size_t nread = sizeof(index_t) * ndim_;
    if (strm->Read(data(), nread) != nread) return false;
    return true;
  }

  friend std::ostream &operator<<(std::ostream &os, const TShape &shape);
  friend std::istream &operator>>(std::istream &is, TShape &shape);

 private:
  // the shape will be stored in data_stack_
  // when dimension is smaller than kStackCache
  // when it is bigger, it will be stored in data_heap_;
  /*! \brief size of in stack space */
  static const index_t kStackCache = 4;
  /*! \brief number of dimnsion of the shape */
  index_t ndim_;
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_;
  /*! \brief in stack space used to store shape when it is small */
  index_t data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  index_t *data_heap_;
  /*!
   * \brief internal function to set the dimension
   * \param dim the dimension of the shape
   */
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
        dim > num_heap_allocated_) {
      // data_heap_ can be NULL
      delete [] data_heap_;
      data_heap_ = new index_t[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*!
 * \brief allow string printing of the shape
 * \param os the output stream
 * \param shape the shape
 * \return the ostream
 */
inline std::ostream &operator<<(std::ostream &os, const TShape &shape) {
  os << '(';
  for (index_t i = 0; i < shape.ndim(); ++i) {
    if (i != 0) os << ", ";
    os << shape[i];
  }
  // python style tuple
  if (shape.ndim() == 1) os << ',';
  os << ')';
  return os;
}

/*!
 * \brief read shape from the istream
 * \param is the input stream
 * \param shape the shape
 * \return the istream
 */
inline std::istream &operator>>(std::istream &is, TShape &shape) {
  // get (
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  index_t idx;
  std::vector<index_t> tmp;
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  shape.CopyFrom(tmp.begin(), tmp.end());
  return is;
}

#define MSHADOW_TYPE_SWITCH(type, DType, ...)       \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

#define MSHADOW_REAL_TYPE_SWITCH(type, DType, ...)  \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint8"; \
    break;                                          \
  case mshadow::kInt32:                             \
    LOG(FATAL) << "This operation only support "      \
                  "floating point types, not int32";  \
    break;                                            \
  default:                                            \
    LOG(FATAL) << "Unknown type enum " << type;       \
  }

/*! \brief get data type size from type enum */
inline size_t mshadow_sizeof(int type) {
  int size = 0;
  MSHADOW_TYPE_SWITCH(type, DType, size = sizeof(DType););
  return size;
}

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself do not involve any arithmentic operations,
 *  but it can be converted to tensor of fixed dimension for further operations
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 */
class TBlob {
 public:
  /*! \brief pointer to the data */
  void *dptr_;
  /*! \brief shape of the tensor */
  TShape shape_;
  /*!
   * \brief storing the stride information in x dimension
   */
  index_t stride_;
  /*! \brief device mask of the corresponding device */
  int dev_mask_;
  /*! \brief type flag of the tensor blob */
  int type_flag_;
  /*! \brief default constructor, default copy assign will work */
  TBlob(void)
      : dptr_(NULL), dev_mask_(cpu::kDevMask),
        type_flag_(DataType<default_real_t>::kFlag) {}
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   */
  template<typename DType>
  TBlob(DType *dptr,
        const TShape &shape,
        int dev_mask)
      : dptr_(dptr), shape_(shape),
        stride_(shape[shape.ndim() - 1]),
        dev_mask_(dev_mask),
        type_flag_(DataType<DType>::kFlag) {}
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   * \param type_flag the type flag. Can be one of enum mshadow::dtype
   */
  TBlob(void *dptr,
        const TShape &shape,
        int dev_mask,
        int type_flag)
      : dptr_(dptr), shape_(shape),
        stride_(shape[shape.ndim() - 1]),
        dev_mask_(dev_mask),
        type_flag_(type_flag) {}
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  TBlob(const Tensor<Device, dim, DType> &src) {  // NOLINT(*)
    *this = src;
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   * \return reference of self
   */
  template<typename Device, int dim, typename DType>
  inline TBlob
  &operator=(const Tensor<Device, dim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    stride_ = src.stride_;
    dev_mask_ = Device::kDevMask;
    type_flag_ = DataType<DType>::kFlag;
    return *this;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return shape_[shape_.ndim() - 1] == stride_;
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline Tensor<Device, 2, DType> FlatTo2D(Stream<Device> *stream = NULL) const {
    CHECK(Device::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << DataType<DType>::kFlag;
    return Tensor<Device, 2, DType>(static_cast<DType*>(dptr_),
                                    shape_.FlatTo2D(), stride_, stream);
  }
  /*! \brief return number of dimension of the tensor inside */
  inline int ndim(void) const {
    return shape_.ndim();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param idx the dimension count from the highest dimensin
   * \return the size
   */
  inline index_t size(index_t idx) const {
    return shape_[idx];
  }
  /*! \brief total number of elements in the tensor */
  inline index_t Size(void) const {
    return shape_.Size();
  }
  /*!
   * \brief fetch the tensor, with respect to specific dimension
   * if dim do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline Tensor<Device, dim, DType> get(Stream<Device> *stream = NULL) const {
    CHECK(Device::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << DataType<DType>::kFlag;
    return Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                       shape_.get<dim>(),
                                       stride_, stream);
  }
  /*!
   * \brief fetch a tensor in given shape
   *  If size do not match the stored size, an error will be issued
   * \return the tensor requested
   * \param shape the shape required
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline Tensor<Device, dim, DType> get_with_shape(const Shape<dim> &shape,
                                                   Stream<Device> *stream = NULL) const {
    CHECK(Device::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << DataType<DType>::kFlag;
    CHECK_EQ(this->CheckContiguous(), true) << "TBlob.get_reshape: must be contiguous";
    CHECK_EQ(this->shape_.Size(), shape.Size())
      << "TBlob.get_with_shape: new and old shape do not match total elements";
    return Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                      shape,
                                      shape[dim - 1],
                                      stream);
  }
};
}  // namespace mshadow
#endif  // MSHADOW_TENSOR_BLOB_H_
//===== EXPANDED: ../mshadow/mshadow/tensor_blob.h =====

//===== EXPANDIND: ../mshadow/mshadow/random.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 *  \file random.h
 *  \brief Random inline functions for tensor.
 *  \author Bing Xu, Tianqi Chen
 *   Based on curand|MKL|stdlib
 */
#ifndef MSHADOW_RANDOM_H_
#define MSHADOW_RANDOM_H_


#if MSHADOW_IN_CXX11
#endif

#if _MSC_VER
#define rand_r(x) rand()
#endif


namespace mshadow {
/*!
 * \brief random number generator
 * \tparam Device the device of random number generator
 * \tparam DType the target data type of random number can be float for double
 */
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class Random {};

/*! \brief CPU random number generator */
template<typename DType>
class Random<cpu, DType> {
 public:
  /*!
   * \brief constructor of random engine
   * \param seed random number seed
   */
  explicit Random(int seed) {
    this->Seed(seed);
    buffer_.Resize(Shape1(kRandBufferSize));
  }
  ~Random(void) {
  }
  /*!
   * \brief seed random number generator using this seed
   * \param seed seed of prng
   */
  inline void Seed(int seed) {
#if MSHADOW_IN_CXX11
    rnd_engine_.seed(seed);
#else
    this->rseed_ = static_cast<unsigned>(seed);
#endif
  }
  /*!
   * \brief set the stream of computation
   * \param stream computation stream
   */
  inline void set_stream(Stream<cpu> *stream) {
  }
  /*!
   * \brief generate data from uniform [a,b)
   * \param dst destination
   * \param a lower bound of uniform
   * \param b upper bound of uniform
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleUniform(Tensor<cpu, dim, DType> *dst,
                            DType a = 0.0f, DType b = 1.0f) {
    if (dst->CheckContiguous()) {
      this->GenUniform(dst->dptr_, dst->shape_.Size(), a, b);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenUniform(mat[i].dptr_, mat.size(1), a, b);
      }
    }
  }
  /*!
   * \brief generate data from standard gaussian
   * \param dst destination
   * \param mu mean variable
   * \param sigma standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleGaussian(Tensor<cpu, dim, DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f) {
    if (sigma <= 0.0f) {
      *dst = mu; return;
    }
    if (dst->CheckContiguous()) {
      this->GenGaussian(dst->dptr_, dst->shape_.Size(), mu, sigma);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenGaussian(mat[i].dptr_, mat.size(1), mu, sigma);
      }
    }
  }
  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleGaussian(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }
  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = uniform(s1) * uniform(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleUniform(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }

 private:
#if MSHADOW_IN_CXX11
  /*! \brief use c++11 random engine. */
  std::mt19937 rnd_engine_;
  // implementing generators.
  inline void GenUniform(DType *dptr, index_t size, DType a, DType b) {
    std::uniform_real_distribution<DType> dist_uniform(a, b);
    for (size_t i = 0; i < size; ++i) {
      dptr[i] = dist_uniform(rnd_engine_);
    }
  }
  inline void GenGaussian(DType *dptr, index_t size, DType mu, DType sigma) {
    std::normal_distribution<DType> dist_normal(mu, sigma);
    for (size_t i = 0; i < size; ++i) {
      dptr[i] = dist_normal(rnd_engine_);
    }
  }

#else
  /*! \brief random number seed used by PRNG */
  unsigned rseed_;
  // functions
  inline void GenUniform(float *dptr, index_t size, float a, float b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<float>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenUniform(double *dptr, index_t size, double a, double b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<double>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenGaussian(float *dptr, index_t size, float mu, float sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussian(double *dptr, index_t size, double mu, double sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussianX(DType *dptr, index_t size, DType mu, DType sigma) {
    DType g1 = 0.0f, g2 = 0.0f;
    for (index_t j = 0; j < size; ++j) {
      if ((j & 1) == 0) {
        this->SampleNormal2D(&g1, &g2);
        dptr[j] = mu + g1 * sigma;
      } else {
        dptr[j] = mu + g2 * sigma;
      }
    }
  }
  /*! \brief get next random number from rand */
  inline DType RandNext(void) {
    return static_cast<DType>(rand_r(&rseed_)) /
        (static_cast<DType>(RAND_MAX) + 1.0f);
  }
  /*! \brief return a real numer uniform in (0,1) */
  inline DType RandNext2(void) {
    return (static_cast<DType>(rand_r(&rseed_)) + 1.0f) /
        (static_cast<DType>(RAND_MAX) + 2.0f);
  }
  /*!
   * \brief sample iid xx,yy ~N(0,1)
   * \param xx first  gaussian output
   * \param yy second gaussian output
   */
  inline void SampleNormal2D(DType *xx_, DType *yy_) {
    DType &xx = *xx_, &yy = *yy_;
    DType x, y, s;
    do {
      x = 2.0f * RandNext2() - 1.0f;
      y = 2.0f * RandNext2() - 1.0f;
      s = x * x + y * y;
    } while (s >= 1.0f || s == 0.0f);
    DType t = std::sqrt(-2.0f * std::log(s) / s);
    xx = x * t; yy = y * t;
  }
#endif
  /*! \brief temporal space used to store random numbers */
  TensorContainer<cpu, 1, DType> buffer_;
};  // class Random<cpu, DType>

// only allow GPU PRNG when cuda is enabled
#if MSHADOW_USE_CUDA
/*! \brief GPU random number generator */
template<typename DType>
class Random<gpu, DType> {
 public:
  /*!
   * \brief constructor of random engine
   * \param seed random number seed
   */
  explicit Random(int seed) {
    curandStatus_t status;
    status = curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Can not create CURAND Generator";
    this->Seed(seed);
    buffer_.Resize(Shape1(kRandBufferSize));
  }
  ~Random(void) DMLC_THROW_EXCEPTION {
    curandStatus_t status;
    status = curandDestroyGenerator(gen_);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Destory CURAND Gen failed";
  }
  /*!
   * \brief set the stream of computation
   * \param stream computation stream
   */
  inline void set_stream(Stream<gpu> *stream) {
    curandStatus_t status;
    status = curandSetStream(gen_, Stream<gpu>::GetStream(stream));

    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "set_stream CURAND failed";
  }
  /*!
   * \brief seed random number generator using this seed
   * \param seed seed of prng
   */
  inline void Seed(int seed) {
    curandStatus_t status;
    status = curandSetPseudoRandomGeneratorSeed(gen_, seed);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Set CURAND seed failed.";
  }
  /*!
   * \brief generate data from uniform [a,b)
   * \param dst destination
   * \param a lower bound of uniform
   * \param b upper bound of uniform
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleUniform(Tensor<gpu, dim, DType> *dst,
                            DType a = 0.0f, DType b = 1.0f);

  /*!
   * \brief generate data from standard gaussian
   * \param dst destination
   * \param mu mean variable
   * \param sigma standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleGaussian(Tensor<gpu, dim, DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f);
  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \param mu mean
   * \param sigma variance
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape, DType mu = 0.0f, DType sigma = 1.0f);
  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape);

 private:
  inline void GenGaussian(float *dptr, size_t size, float mu, float sigma) {
    curandStatus_t status;
    status = curandGenerateNormal(gen_, dptr, size, mu, sigma);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform failed";
  }
  inline void GenGaussian(double *dptr, size_t size, double mu, double sigma) {
    curandStatus_t status;
    status = curandGenerateNormalDouble(gen_, dptr, size, mu, sigma);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform failed";
  }
  inline void GenUniform(float *dptr, size_t size) {
    curandStatus_t status;
    status = curandGenerateUniform(gen_, dptr, size);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform failed";
  }
  inline void GenUniform(double *dptr, size_t size) {
    curandStatus_t status;
    status = curandGenerateUniformDouble(gen_, dptr, size);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform failed";
  }
  /*! \brief random numbeer generator */
  curandGenerator_t gen_;
  /*! \brief templ buffer */
  TensorContainer<gpu, 1, DType> buffer_;
};  // class Random<gpu, DType>
#endif  // MSHADOW_USE_CUDA

#ifdef __CUDACC__
// implementations that depends on cuda kernels
template<typename DType>
template<int dim>
inline void Random<gpu, DType>::SampleUniform(
    Tensor<gpu, dim, DType> *dst, DType a, DType b) {
  if (a == 0.0f && b == 1.0f) {
    if (dst->CheckContiguous()) {
      this->GenUniform(dst->dptr_, dst->shape_.Size());
    } else {
      *dst = this->uniform(dst->shape_);
    }
  } else {
    *dst = this->uniform(dst->shape_) * (b - a) + a;
  }
}
template<typename DType>
template<int dim>
inline void Random<gpu, DType>::SampleGaussian(
    Tensor<gpu, dim, DType> *dst, DType mu, DType sigma) {
  if (dst->CheckContiguous()) {
    this->GenGaussian(dst->dptr_, dst->shape_.Size(), mu, sigma);
  } else {
    *dst = this->gaussian(dst->shape_, mu, sigma);
  }
}

template<typename DType>
template<int dim>
inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
Random<gpu, DType>::gaussian(Shape<dim> shape, DType mu, DType sigma) {
  size_t aligned_sz = ((shape.Size() + 1UL) >> 1) << 1;
  // allocate alligned size
  buffer_.Resize(Shape1(aligned_sz));
  buffer_.Resize(Shape1(shape.Size()));
  this->GenGaussian(buffer_.dptr_, aligned_sz, mu, sigma);
  return expr::reshape(buffer_, shape);
}

template<typename DType>
template<int dim>
inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
Random<gpu, DType>::uniform(Shape<dim> shape) {
  buffer_.Resize(Shape1(shape.Size()));
  this->GenUniform(buffer_.dptr_, buffer_.size(0));
  return expr::reshape(buffer_, shape);
}
#endif  // __CUDACC__
}  // namespace mshadow
#endif  // MSHADOW_RANDOM_H_
//===== EXPANDED: ../mshadow/mshadow/random.h =====

// add definition of scalar related operators
#ifdef MSAHDOW_SCALAR_
  #error "MSHADOW_SCALAR_ must not be defined"
#endif
// enumerate all the scalar data type we aim to be good at
#define MSHADOW_SCALAR_ float
//===== EXPANDIND: ../mshadow/mshadow/expr_scalar-inl.h =====

/*!
 *  Copyright (c) 2014 by Contributors
 * \file expression-inl.h
 * \brief definitions of operators in expression with respect to scalar
 *  this file will be included several times, each time with MACRO MSHADOW_SCALAR_ to be different types
 *
 * DO NOT add pragma once or macro guard
 * \author Tianqi Chen, Bing Xu
 */
// macro guard is harmful, used to pass the cpplint
#ifndef MSHADOW_EXPR_SCALAR_INL_H_
#define MSHADOW_EXPR_SCALAR_INL_H_
// undef the guard so it can be included multiple times
#undef MSHADOW_EXPR_SCALAR_INL_H_

namespace mshadow {
namespace expr {
// DotExp
/*! \brief dot operator def */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &lhs,
          MSHADOW_SCALAR_ rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(lhs.lhs_, lhs.rhs_, lhs.scale_ * rhs);
}
/*! \brief scale of dot operation */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(MSHADOW_SCALAR_ lhs,
          const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(rhs.lhs_, rhs.rhs_, rhs.scale_ * lhs);
}

/*! \brief operator overload */
template<typename E, typename DType, typename R, int d>
inline ReduceTo1DExp<E, DType, R, d>
operator*(const ReduceTo1DExp<E, DType, R, d> &e, MSHADOW_SCALAR_ scale) {
  return ReduceTo1DExp<E, DType, R, d>(e.src_, e.scale_ * scale);
}
/*! \brief operator overload */
template<typename E, typename DType, typename R, int d>
inline ReduceTo1DExp<E, DType, R, d>
operator*(MSHADOW_SCALAR_ scale, const ReduceTo1DExp<E, DType, R, d> &e) {
  return ReduceTo1DExp<E, DType, R, d>(e.src_, e.scale_ * scale);
}

/*! \brief operator overload for const */
template<typename OP, typename TA, int ta>
inline BinaryMapExp<OP, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
F(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs, const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
/*! \brief operator overload for const */
template<typename OP, typename TB, int tb>
inline BinaryMapExp<OP, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
F(const ScalarExp<MSHADOW_SCALAR_> &lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
// constant operators
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::plus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator+(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::minus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator-(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::mul, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator*(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::div, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator/(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
// constant operators 2
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::plus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator+(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::minus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator-(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::mul, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator*(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::div, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator/(const ScalarExp<MSHADOW_SCALAR_> &lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPR_SCALAR_INL_H_
//===== EXPANDED: ../mshadow/mshadow/expr_scalar-inl.h =====

#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ double
#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ int
#undef MSHADOW_SCALAR_
#endif  // MSHADOW_TENSOR_H_
//===== EXPANDED: ../mshadow/mshadow/tensor.h =====


/*!
 *\brief whether to use opencv support
 */
#ifndef MXNET_USE_OPENCV
#define MXNET_USE_OPENCV 1
#endif

/*!
 *\brief whether to use cuda support
 */
#ifndef MXNET_USE_CUDA
#define MXNET_USE_CUDA MSHADOW_USE_CUDA
#endif

/*!
 *\brief whether to use cudnn library for convolution
 */
#ifndef MXNET_USE_CUDNN
#define MXNET_USE_CUDNN MSHADOW_USE_CUDNN
#endif

/*! \brief Error message for using gpu when MXNET_USE_CUDA==0 */
#define MXNET_GPU_NOT_ENABLED_ERROR  "GPU is not enabled"

/*!
 * \brief define compatible keywords in g++
 *  Used to support g++-4.6 and g++4.7
 */
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#error "Currently we need g++ 4.8 or higher to fully support c++11 features"
#define override
#define final
#endif
#endif

/*!
 * \brief define dllexport for Visual Studio
 */
#ifdef _MSC_VER
#ifdef MXNET_EXPORTS
#define MXNET_API __declspec(dllexport)
#else
#define MXNET_API __declspec(dllimport)
#endif
#else
#define MXNET_API
#endif

/*!
 * \brief define prediction only
 */
#ifndef MXNET_PREDICT_ONLY
#define MXNET_PREDICT_ONLY 0
#endif


/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief mxnet cpu */
typedef mshadow::cpu cpu;
/*! \brief mxnet gpu */
typedef mshadow::gpu gpu;
/*! \brief index type usually use unsigned */
typedef mshadow::index_t index_t;
/*! \brief data type that will be used to store ndarray */
typedef mshadow::default_real_t real_t;

/*! \brief dynamic shape type */
typedef mshadow::TShape TShape;
/*! \brief storage container type */
typedef mshadow::TBlob TBlob;

/*! \brief Context information about the execution enviroment */
struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = cpu::kDevMask,
    kGPU = gpu::kDevMask,
    kCPUPinned = 3
  };
  /*! \brief the device type we run the op on */
  DeviceType dev_type;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
  /*! \brief default constructor */
  Context() : dev_type(kCPU), dev_id(0) {}
  /*!
   * \brief Get corresponding device mask
   * \return cpu::kDevMask or gpu::kDevMask
   */
  inline int dev_mask() const {
    if (dev_type == kCPUPinned) return cpu::kDevMask;
    return dev_type;
  }
  /*!
   * \brief Comparator, used to enable Context as std::map key.
   * \param b another context to compare
   * \return compared result
   */
  inline bool operator<(const Context &b) const;
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_type == b.dev_type && dev_id == b.dev_id;
  }
  /*!
   * \brief check if current context not equals another one
   * \param b another context to compare
   * \return whether they are not the same
   */
  inline bool operator!=(const Context &b) const {
    return !(*this == b);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  inline void Save(dmlc::Stream *strm) const {
    strm->Write(&dev_type, sizeof(dev_type));
    strm->Write(&dev_id, sizeof(dev_id));
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  inline bool Load(dmlc::Stream *strm) {
    if (strm->Read(&dev_type, sizeof(dev_type)) != sizeof(dev_type)) return false;
    if (strm->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) return false;
    return true;
  }
  /*! \brief the maximal device type */
  static const int32_t kMaxDevType = 4;
  /*! \brief the maximal device index */
  static const int32_t kMaxDevID = 16;
  /*!
   * \brief Create a new context.
   * \param dev_type device type.
   * \param dev_id device id.
   */
  inline static Context Create(DeviceType dev_type, int32_t dev_id);
  /*! \return CPU Context */
  inline static Context CPU();
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context.
   */
  inline static Context GPU(int32_t dev_id);
  /*!
   * Create a pinned CPU context.
   * \param dev_id the device id for corresponding GPU.
   * \return Pinned CPU context.
   */
  inline static Context CPUPinned(int32_t dev_id);
};

/*!
 * \brief execution time context.
 *  The information needed in runtime for actual execution.
 */
struct RunContext {
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return static_cast<mshadow::Stream<xpu>*>(stream);
  }
};
}  // namespace mxnet

//! \cond Doxygen_Suppress
namespace mxnet {
// implementing Context
inline bool Context::operator<(const Context &b) const {
  if (dev_type == b.dev_type) {
    return dev_id < b.dev_id;
  } else {
    return dev_type < b.dev_type;
  }
}
inline Context Context::Create(DeviceType dev_type, int32_t dev_id) {
  Context ctx;
  ctx.dev_type = dev_type;
  ctx.dev_id = dev_id;
  return ctx;
}
inline Context Context::CPU() {
  return Create(kCPU, 0);
}

inline Context Context::CPUPinned(int32_t dev_id) {
  return Create(kCPUPinned, dev_id);
}

inline Context Context::GPU(int32_t dev_id) {
  return Create(kGPU, dev_id);
}
}  // namespace mxnet

namespace dmlc {
// Add a few patches to support TShape in dmlc/parameter.
DMLC_DECLARE_TYPE_NAME(mxnet::TShape, "Shape(tuple)");

namespace parameter {
template<>
class FieldEntry<mxnet::TShape>
    : public FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> {
 public:
  FieldEntry() : enforce_nonzero_(false), expect_ndim_(0) {}
  // parent class
  typedef FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> Parent;

  virtual void Check(void *head) const {
    Parent::Check(head);
    mxnet::TShape &v = this->Get(head);
    if (expect_ndim_ != 0 && v.ndim() != expect_ndim_) {
      std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " has wrong dimensions, expected dimension=" << expect_ndim_;
        throw dmlc::ParamError(os.str());
    }
    if (enforce_nonzero_) {
      for (mxnet::index_t i = 0; i < v.ndim(); ++i) {
        if (v[i] == 0U) {
          std::ostringstream os;
          os << "value " << v << "for Parameter " << this->key_
             << " is invalid, the input shape must be nonzero in all dimensions";
          throw dmlc::ParamError(os.str());
        }
      }
    }
  }
  inline FieldEntry<mxnet::TShape> &enforce_nonzero() {
    this->enforce_nonzero_ = true;
    return this->self();
  }
  inline FieldEntry<mxnet::TShape> &set_expect_ndim(mshadow::index_t ndim) {
    expect_ndim_ = ndim;
    return this->self();
  }

 private:
  // whether all the entries need to be nonzero
  bool enforce_nonzero_;
  // expected number of dimension, default = 0 means no restriction.
  mxnet::index_t expect_ndim_;
};
}  // namespace parameter
}  // namespace dmlc
//! \endcond
#endif  // MXNET_BASE_H_
//===== EXPANDED: ../include/mxnet/base.h =====

//===== EXPANDIND: ../include/mxnet/operator.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.h
 * \brief Operator interface of mxnet.
 * \author Naiyan Wang
 */
#ifndef MXNET_OPERATOR_H_
#define MXNET_OPERATOR_H_

//===== EXPANDIND: ../include/mxnet/resource.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.h
 * \brief Global resource allocation handling.
 */
#ifndef MXNET_RESOURCE_H_
#define MXNET_RESOURCE_H_

//===== EXPANDIND: ../include/mxnet/engine.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file engine.h
 * \brief Engine that schedules all the operations according to dependency.
 */
#ifndef MXNET_ENGINE_H_
#define MXNET_ENGINE_H_

#if DMLC_USE_CXX11
#endif

namespace mxnet {

// forward declare engine
class Engine;

/*! \brief namespace of engine internal types. */
namespace engine {
/*! \brief Internal representation of variable. */
struct Var;
/*! \brief Internal representation of operator.  */
struct Opr;
/*! \brief Variable pointer type, usually hold by user used to specify dependencies. */
typedef Var* VarHandle;
/*! \brief Operator pointer type, usually hold by user.*/
typedef Opr* OprHandle;
/*!
 * \brief OnComplete Callback to the engine,
 *  called by AsyncFn when action completes
 */
class CallbackOnComplete {
 public:
  // use implicit copy and assign
  /*! \brief involve the callback */
  inline void operator()() const {
    (*callback_)(engine_, param_);
  }

 private:
  /*! \brief engine can see content of callback */
  friend class ::mxnet::Engine;
  /*! \brief the real callback */
  void (*callback_)(Engine *, void *);
  /*! \brief the engine class passed to callback */
  Engine* engine_;
  /*! \brief the parameter set on callback */
  void* param_;
};
}  // namespace engine

#if DMLC_USE_CXX11
/*! \brief Function property, used to hint what action is pushed to engine. */
enum class FnProperty {
  /*! \brief Normal operation */
  kNormal,
  /*! \brief Copy operation from GPU to other devices */
  kCopyFromGPU,
  /*! \brief Copy operation from CPU to other devices */
  kCopyToGPU,
  /*! \brief Prioritized sync operation on CPU */
  kCPUPrioritized,
  /*! \brief Asynchronous function call */
  kAsync
};  // enum class FnProperty

/*!
 * \brief Dependency engine that schedules operations.
*/
class MXNET_API Engine {
 public:
  /*! \brief callback on complete*/
  typedef engine::CallbackOnComplete CallbackOnComplete;
  /*! \brief Synchronous operation to pass to engine. */
  typedef std::function<void(RunContext)> SyncFn;
  /*! \brief Asynchronous operation to pass to engine. */
  typedef std::function<void(RunContext, CallbackOnComplete)> AsyncFn;
  /*! \brief Variable pointer */
  typedef engine::VarHandle VarHandle;
  /*! \brief Operator pointer */
  typedef engine::OprHandle OprHandle;
  /*!
   * \brief Notify the engine about a shutdown,
   *  This can help engine to print less messages into display.
   *
   *  User do not have to call this function.
   * \return 0 when success, -1 when failure happens.
   */
  virtual void NotifyShutdown() = 0;
  /*!
   * \brief Allocate a new variable, the variable can then
   *        be used to schedule the operation concurrently via dependency
   *        patterns.
   * \return The new variable allocated.
   */
  virtual VarHandle NewVariable() = 0;
  /*!
   * \brief Create a new operator. The returned operator could be saved
   *        externally so that it could be resued for scheduling.
   * \param fn The execution function.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \return The new operator allocated.
   */
  virtual OprHandle NewOperator(AsyncFn fn,
                                std::vector<VarHandle> const& const_vars,
                                std::vector<VarHandle> const& mutable_vars,
                                FnProperty prop = FnProperty::kNormal) = 0;
  /*!
   * \brief Delete the given operator.
   * \param op The operator to delete.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations using this operator are completed.
   */
  virtual void DeleteOperator(OprHandle op) = 0;
  /*!
   * \brief Push an operator to the engine.
   * \param op The operator to push.
   * \param exec_ctx Execution context.
   * \param priority Priority of the action, as hint to the engine.
   */
  virtual void Push(OprHandle op, Context exec_ctx, int priority = 0) = 0;
  /*!
   * \brief Push an asynchronous operation to the engine.
   * \param exec_fun Execution function, this function takes a parameter
   *                 on_complete that must be called when the execution
   *                 completes.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \param priority Priority of the action, as hint to the engine.
   */
  virtual void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                         std::vector<VarHandle> const& const_vars,
                         std::vector<VarHandle> const& mutable_vars,
                         FnProperty prop = FnProperty::kNormal,
                         int priority = 0) = 0;
  /*!
   * \brief Schedule the deletion of a variable.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations depending on var are completed.
   *
   * \param delete_fn A function that will be called after the variable is
   *                   deleted.
   * \param exec_ctx Execution context.
   * \param var The variable to be deleted.
   */
  virtual void DeleteVariable(SyncFn delete_fn,
                              Context exec_ctx,
                              VarHandle var) = 0;
  /*!
   * \brief Wait for a variable.
   * \param var The variable we should wait for. This function returns when the
   *            variable is ready.
   */
  virtual void WaitForVar(VarHandle var) = 0;
  /*!
   * \brief Wait until all the activity of engine finishes.
   */
  virtual void WaitForAll() = 0;
  /*!\brief virtual destructor */
  virtual ~Engine() noexcept(false) {}
  /*!
   * \return Engine singleton.
   */
  static Engine* Get();
  /*!
   * \brief Get shared pointer reference to engine singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  engine to be destructed after X.
   *
   * \return A shared pointer to Engine singleton.
   */
  static std::shared_ptr<Engine> _GetSharedRef();
  /*!
   * \brief Push an synchronous operation to the engine.
   * \param exec_fn Execution function that executes the operation.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \param prop Property of the function.
   * \param priority Priority of the action, as hint to the engine.
   * \tparam SyncFn the synchronous function to be pushed.
   */
  template<typename SyncFn>
  inline void PushSync(SyncFn exec_fn, Context exec_ctx,
                       std::vector<VarHandle> const& const_vars,
                       std::vector<VarHandle> const& mutable_vars,
                       FnProperty prop = FnProperty::kNormal,
                       int priority = 0) {
    this->PushAsync([exec_fn](RunContext ctx, CallbackOnComplete on_complete) {
        exec_fn(ctx);
        on_complete();
      }, exec_ctx, const_vars, mutable_vars, prop, priority);
  }

 protected:
  /*!
   * \brief factory function to create OnComplete callback.
   * \param callback th static callback function.
   * \param param the paramter passed to callback.
   */
  inline CallbackOnComplete CreateCallback(
      void (*callback)(Engine *, void *), void *param) {
    CallbackOnComplete ret;
    ret.callback_ = callback;
    ret.engine_ = this;
    ret.param_ = param;
    return ret;
  }
};  // class Engine
#endif  // DMLC_USE_CXX11
}  // namespace mxnet
#endif  // MXNET_ENGINE_H_
//===== EXPANDED: ../include/mxnet/engine.h =====


namespace mxnet {

/*!
 * \brief The resources that can be requested by Operator
 */
struct ResourceRequest {
  /*! \brief Resource type, indicating what the pointer type is */
  enum Type {
    /*! \brief mshadow::Random<xpu> object */
    kRandom,
    /*! \brief A dynamic temp space that can be arbitrary size */
    kTempSpace
  };
  /*! \brief type of resources */
  Type type;
  /*! \brief default constructor */
  ResourceRequest() {}
  /*!
   * \brief constructor, allow implicit conversion
   * \param type type of resources
   */
  ResourceRequest(Type type)  // NOLINT(*)
      : type(type) {}
};


/*!
 * \brief Resources used by mxnet operations.
 *  A resource is something special other than NDArray,
 *  but will still participate
 */
struct Resource {
  /*! \brief The original request */
  ResourceRequest req;
  /*! \brief engine variable */
  engine::VarHandle var;
  /*! \brief identifier of id information, used for debug purpose */
  int32_t id;
  /*!
   * \brief pointer to the resource, do not use directly,
   *  access using member functions
   */
  void *ptr_;
  /*! \brief default constructor */
  Resource() : id(0) {}
  /*!
   * \brief Get random number generator.
   * \param stream The stream to use in the random number generator.
   * \return the mshadow random number generator requested.
   * \tparam xpu the device type of random number generator.
   */
  template<typename xpu, typename DType>
  inline mshadow::Random<xpu, DType>* get_random(
      mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kRandom);
    mshadow::Random<xpu, DType> *ret =
        static_cast<mshadow::Random<xpu, DType>*>(ptr_);
    ret->set_stream(stream);
    return ret;
  }
  /*!
   * \brief Get space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   * \tparam xpu the device type of random number generator.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim>
  inline mshadow::Tensor<xpu, ndim, real_t> get_space(
      mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kTempSpace);
    mshadow::TensorContainer<xpu, 1, real_t> *space =
        static_cast<mshadow::TensorContainer<xpu, 1, real_t>*>(ptr_);
    space->Resize(mshadow::Shape1(shape.Size()));
    return mshadow::Tensor<xpu, ndim, real_t>(
        space->dptr_, shape, shape[ndim - 1], stream);
  }
};

/*! \brief Global resource manager */
class ResourceManager {
 public:
  /*!
   * \brief Get resource of requested type.
   * \param ctx the context of the request.
   * \param req the resource request.
   * \return the requested resource.
   * \note The returned resource's ownership is
   *       still hold by the manager singleton.
   */
  virtual Resource Request(Context ctx, const ResourceRequest &req) = 0;
  /*!
   * \brief Seed all the allocated random numbers.
   * \param seed the seed to the random number generators on all devices.
   */
  virtual void SeedRandom(uint32_t seed) = 0;
  /*! \brief virtual destructor */
  virtual ~ResourceManager() DMLC_THROW_EXCEPTION {}
  /*!
   * \return Resource manager singleton.
   */
  static ResourceManager *Get();
};
}  // namespace mxnet
#endif  // MXNET_RESOURCE_H_
//===== EXPANDED: ../include/mxnet/resource.h =====


namespace mxnet {
/*! \brief operation request type to Forward and Backward */
enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
   * \brief perform an inplace write,
   * Target shares memory with one of input arguments.
   * This option only happen when
   */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

/*!
 * \brief All the possible information needed by Operator.Forward and Backward
 *  This is the superset of RunContext.
 *  We use this data structure to bookkeep everything needed by Forward and Backward.
 * \sa Resource
 */
struct OpContext {
  /*! \brief whether it is training phase */
  int is_train;
  /*! \brief RunContext related resources */
  RunContext run_ctx;
  /*! \brief the callback when operation completes, used by asynchronize ops */
  engine::CallbackOnComplete async_on_complete;
  /*! \brief Resources requested by the operator */
  std::vector<Resource> requested;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return run_ctx.get_stream<xpu>();
  }
};

/*!
 * \brief Operator interface.
 *  Operator defins basic operation unit of optimized computation graph in mxnet.
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward.
 *
 *  Operator is generated by OperatorProperty.
 *  To add new operator(aka. layers of neural nets) to mxnet, developer need to create
 *  a new OperatorProperty and its corresponding Operator.
 *
 * \sa TBlob, TShape, OperatorProperty
 */
class Operator {
 public:
  /*! \brief the execution type of the operator */
  enum ExecType {
    /*! \brief Forward/Backward are synchronize calls */
    kSync,
    /*!
     * \brief Forward/Backward are asynchronize,
     *  will call OpContext.async_on_complete when operation finishes.
     */
    kAsync,
    /*!
     * \brief Cross device copy operation, this is a special operator
     *  That indicates copy across devices, the input and output can sit on different device.
     *  In current implementation, copy operator is specially handled by executor.
     *  This flag is used for special case treatment and future extension of different copy ops.
     */
    kCrossDeviceCopy
  };
  /*! \brief destructor */
  virtual ~Operator() {}
  /*!
   * \brief perform a forward operation of Operator, save the output to TBlob.
   * \param ctx runtime context available to this call
   * \param in_data array of input data, it is const
   * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
   * \param out_data array of output data, pointer is used to indicate that this is holder
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   * \param aux_states Auxiliary states of operator. Normally operator doesn't
   *        need, epecial case like Batch Norm requires.
   * \sa OpReqType, OpContext
   */
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) = 0;
  /*!
   * \brief Perform a Backward Operation, write gradient to the in_grad.
   *
   * \note
   * Convention:
   *   out_grad.size() == OperatorProperty.NumVisibleOutputs()
   *   out_data.size() == OperatorProperty.NumOutputs()
   * out_data can contain additional invisible returns that remembers the
   * state carried from the Forward pass. For example mask in the dropout.
   * The gradients are passed from visible returns in this function.
   *
   * \par
   * Not all the TBlobs in the arguments will be available
   * if you override the DeclareBackwardDependency of corresponding OperatorProperty class.
   * Only the dependencies you declared will be available at corresponding position,
   * the rest of the parameters are simply dummy where you will get a nullptr.
   * You will be safe if you use the default DeclareBackwardDependency.
   * But only declare what you need will give engine more chance for optimization.
   *
   * \param ctx runtime context available to this call
   * \param out_grad the gradient value we get from of the Operator.
   * \param in_data the array of input data.
   * \param out_data the array of output data.
   * \param req request types of the saving operation, can be all types.
   * \param in_grad the array of gradient we need to write to.
   * \param aux_states Auxiliary states of operator. Normally operator doesn't need
   * \sa OperatorProperty, OpReqType, OpContext
   */
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "Backward is not implemented";
  }
  /*! \return execution type of the operator */
  virtual ExecType exec_type() const {
    return kSync;
  }
};

#if DMLC_USE_CXX11
// OperatorProperty allows C++11, while Operator do not rely on it.
/*!
 * \brief OperatorProperty is a object that stores all information about Operator.
 * It also contains method to generate context(device) specific operators.
 *
 * It also contains various functions that can be optimally overriden to
 * provide optimization chance for computation engine.
 */
class OperatorProperty {
 public:
  /*!
   * \brief virtual destructor
   */
  virtual ~OperatorProperty() {}
  /*!
   *  \brief Initialize the Operator by setting the parameters
   *  This function need to be called before all other functions.
   *  \param kwargs the keyword arguments parameters
   */
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;
  /*!
   * \brief Get a map representation of internal parameters.
   *  This can be used by Init to recover the state of OperatorProperty.
   */
  virtual std::map<std::string, std::string> GetParams() const = 0;
  /*!
   * \brief Get input arguments of the Operator.
   * \return vector of arguments.
   */
  virtual std::vector<std::string> ListArguments() const {
    return {"data"};
  }
  /*!
   * \brief Get name of output values of Operator
   * \return name of output values.
   */
  virtual std::vector<std::string> ListOutputs() const {
    return {"output"};
  }
  /*!
   * \brief Get name of auxilary states of Operator
   * \return name of return values.
   */
  virtual std::vector<std::string> ListAuxiliaryStates() const {
    return {};
  }
  /*! \return number of real return values of the Operator */
  virtual int NumOutputs() const {
    return 1;
  }
  /*!
   * \brief get number of visible return values during Symbol creation.
   *  If NumVisibleOutputs() = k, and NumOutputs() = n.
   *  The first k returns will be presented in the resulting symbol.
   *
   *  The rest of the returns can be used for auxiliary states for Backward.
   *  For example, Dropout will return [data, mask], with NumVisibleOutputs() == 1.
   *  So when user call sym = Dropout(input), only data is presented in sym.
   *  But all the returns will be presented in out_data parameter of Backward if requested.
   *
   * \return number of default return values
   */
  virtual int NumVisibleOutputs() const {
    return NumOutputs();
  }
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param aux_shape the shape of auxiliary states of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const = 0;
  /*!
   * \brief infer the data types of outputs and unknown input arguments
   * \param in_type the type of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_type allows unknown elements, which are checked by type.ndim() == 0.
   *     For unknown types, Infertype will try to fill in the correct type in in_type
   *     For known types, Infertype will check type consistency
   *
   *     common practice: set the type of data input, and usually weight's type can be infered
   *
   * \param out_type the type of outputs of the operator
   *     Infertype will modify the vector to fill output Ttype
   * \param aux_type the type of auxiliary states of the operator
   *     Infertype will modify the vector to fill output Ttype
   * \return true if the type inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_types are inconsistent.
   */
  virtual bool InferType(std::vector<int> *in_type,
                          std::vector<int> *out_type,
                          std::vector<int> *aux_type) {
    CHECK_LE(in_type->size(), this->ListArguments().size());
    int n_in = this->ListArguments().size();
    for (unsigned i = 0; i < in_type->size(); ++i) {
      CHECK(in_type->at(i) == mshadow::default_type_flag ||
            in_type->at(i) == -1) << "Unsupported data type " << in_type->at(i);
    }
    in_type->clear();
    for (int i = 0; i < n_in; ++i ) in_type->push_back(mshadow::default_type_flag);

    int n_out = this->ListOutputs().size();
    out_type->clear();
    for (int i = 0; i < n_out; ++i ) out_type->push_back(mshadow::default_type_flag);

    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(mshadow::default_type_flag);
    return true;
  }
  /*!
   * \brief Copy this OperatorProperty.
   * \return a pointer to the copied OperatorProperty
   */
  virtual OperatorProperty* Copy() const = 0;
  /*!
   * \brief Create a Operator on specific context
   */
  virtual Operator* CreateOperator(Context ctx) const = 0;
  /*!
   * \brief return the type string of the Operator
   *  subclasses override this function.
   * \return The type string.
   */
  virtual std::string TypeString() const = 0;
  //--------------------------------------------------------
  // All the below functions are optional to override.
  //--------------------------------------------------------
  /*!
   * \brief Declare additional resource required in forward pass.
   *  These additional resources will be presented in OpContext.requested
   *  in the same order of the returned Resource.
   * \param in_shape The input shape to the operator, corresponds to shapes of in_data.
   * \return Additional resource request
   */
  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const {
    return std::vector<ResourceRequest>();
  }
  /*!
   * \brief Decalre additional resource required in backward pass.
   *  These additional resources will be presented in OpContext.requested
   *  in the same order of the returned Resource.
   * \param in_shape The input shape to the operator, corresponds to shapes of in_data.
   * \return Additional resource request
   */
  virtual std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const {
    return std::vector<ResourceRequest>();
  }
  /*!
   * \brief Declare the input requirement of Backward pass.
   *
   *  Only the returned list of variables will be used in Backward.
   *  This function is used for memory optimization.
   *  It is adviced to override and only return what is actually needed.
   *  If this function is not overriden, all the variables will be valid in Backward.
   *
   * \code
   *  // The following code declares Backward need out_grad[0], in_data[0],in_data[1]
   *  vector<int> BackwardInputs(const vector<int> &out_grad,
   *                             const vector<int> &in_data,
   *                             const vector<int> &out_data) const {
   *    return {out_grad[0], in_data[0], in_data[1]};
   *  }
   * \endcode
   * \param out_grad gradient of outputs in backward pass.
   * \param in_data the input data in forward pass.
   * \param out_data the output data in forward pass.
   * \return an integer vector indicating the input requirments
   * \sa BackwardInputs
   */
  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    // By default requires to see all the things.
    // remember to override this function to get a better performance.
    std::vector<int> ret = out_grad;
    ret.insert(ret.end(), in_data.begin(), in_data.end());
    ret.insert(ret.end(), out_data.begin(), out_data.end());
    return ret;
  }
  /*!
   * \brief Get possible forward inplace options.
   *  This function enables optimization to reuse memory of inputs in output.
   *  Only override when necessary, by default in-place is disabled.
   *
   *  The reason for void* type in the out_data is to distinguish the order
   *  of mappings between the two, compiler will report error when
   *  in_data and out_data's order in the pair get reversed.
   *
   * \code
   *  // The following code says out_data[0] can share data with in_data[0]
   *  vector<pair<int, void*> > ForwardInplaceOption(const vector<int> &in_data,
   *                                                 const vector<void*> &out_data) const {
   *    return {{in_data[0], out_data[0]}};
   *  }
   * \endcode
   * \param in_data The input data in forward pass.
   * \param out_data The output data in forward pass.
   * \return list of pair of that maps input->output,
   *   indicating possible in place operations.
   */
  virtual std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const {
    return std::vector<std::pair<int, void*> >();
  }
  /*!
   * \brief Get possible backward inplace options.
   *  This function enables optimization to reuse memory of inputs in output.
   *  Only override when necessary, by default in-place is disabled.
   *
   *  The reason for void* type in the in_grad is to distinguish the order
   *  of mappings between the two, compiler will report error when
   *  in_data and out_data's order in the pair get reversed.
   *
   * \code
   *  // The following code says in_grad[0] can share data with in_data[0]
   *  vector<pair<int,int> > BackwardInplaceOption(
   *                 const std::vector<int> &out_grad,
   *                 const std::vector<int> &in_data,
   *                 const std::vector<int> &out_data,
   *                 const std::vector<int> &in_grad) const {
   *    return {in_data[0], in_grad[0]}};
   *  }
   * \endcode
   * \param in_data The input data in forward pass.
   * \param out_data The output data in forward pass.
   * \param in_grad Gradient of inputs in backward pass.
   * \param out_grad Gradient of outputs in backward pass.
   * \return list of pair of that maps input->output,
   *   indicating possible in place operations.
   */
  virtual std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const {
    return std::vector<std::pair<int, void*> >();
  }
  /*!
   * \brief Get Backward Input Dependency for generic types of data.
   *  Normally T can be pointer of Symbol::DataEntry, or NDArray.
   *  This function will select the result list of T according to DeclareBackwardDependency.
   *
   * \param in_data the input data in forward pass.
   * \param out_data the output data in forward pass.
   * \param out_grad gradient of outputs in backward pass.
   * \tparam T the generic type parameter.
   * \return vector of inputs the Backward Operation depends on.
   * \sa DeclareBackwardDependency
   */
  template<typename T>
  inline std::vector<T> BackwardInputs(const std::vector<T> &out_grad,
                                       const std::vector<T> &in_data,
                                       const std::vector<T> &out_data) const {
    int counter = 0;
    std::vector<int> out_grad_index(out_grad.size());
    std::vector<int> in_data_index(in_data.size());
    std::vector<int> out_data_index(out_data.size());
    for (size_t i = 0; i < out_grad_index.size(); ++i) {
      out_grad_index[i] = counter++;
    }
    for (size_t i = 0; i < in_data_index.size(); ++i) {
      in_data_index[i] = counter++;
    }
    for (size_t i = 0; i < out_data_index.size(); ++i) {
      out_data_index[i] = counter++;
    }
    std::vector<T> all_data;
    all_data.insert(all_data.end(), out_grad.begin(), out_grad.end());
    all_data.insert(all_data.end(), in_data.begin(), in_data.end());
    all_data.insert(all_data.end(), out_data.begin(), out_data.end());

    std::vector<int> ret_index = this->DeclareBackwardDependency(
        out_grad_index, in_data_index, out_data_index);

    std::vector<T> ret(ret_index.size());
    for (size_t i = 0; i < ret_index.size(); ++i) {
      ret[i] = all_data[ret_index[i]];
    }
    return ret;
  }
  /*!
   * \brief create OperatorProperty
   * \param type_name the type string of the OperatorProperty
   * \return a new constructed OperatorProperty
   */
  static OperatorProperty *Create(const char* type_name);
};

/*! \brief typedef the factory function of operator property */
typedef std::function<OperatorProperty *()> OperatorPropertyFactory;
/*!
 * \brief Registry entry for OperatorProperty factory functions.
 */
struct OperatorPropertyReg
    : public dmlc::FunctionRegEntryBase<OperatorPropertyReg,
                                        OperatorPropertyFactory> {
  /*!
   * \brief Set key_var_num_args
   *  When this is set, the API caller is required to pass in a
   *  argument with key=key_num_args.c_str(), and value=num_args.
   *  num_args is number of positional argument when calling the function.
   *
   *  This is used to pass in length of positional arguments
   *  for operators that can take variable length of input.
   *  Most operators do not need to set this property.
   *
   * \param key the key name to be set
   */
  inline OperatorPropertyReg& set_key_var_num_args(const std::string &key) {  // NOLINT(*)
    this->key_var_num_args = key;
    return *this;
  }
  /*!
   * \brief Check if TypeString of the type matches the registered name
   */
  inline OperatorPropertyReg& check_name() {
    OperatorProperty *p = this->body();
    std::string type = p->TypeString();
    delete p;
    CHECK_EQ(this->name, type)
        << "Register Name and TypeString mismatch, name=\"" << this->name << "\","
        << " but TypeString=\"" << type <<"\"";
    return *this;
  }

  /*! \brief The key num_args name. */
  std::string key_var_num_args;
};

//--------------------------------------------------------------
// The following part are API Registration of Operators
//--------------------------------------------------------------
/*!
 * \brief Macro to register OperatorProperty
 *
 * \code
 * // example of registering a fully connected operator
 * REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedOpProp)
 * .describe("Fully connected layer");
 *
 * \endcode
 */
#define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
  DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
  .set_body([]() { return new OperatorPropertyType(); })                \
  .check_name()

#endif  // DMLC_USE_CXX11
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
//===== EXPANDED: ../include/mxnet/operator.h =====


#if DMLC_USE_CXX11
#endif

namespace mxnet {
namespace common {
/*! \brief namespace of arguments */
namespace arg {
/*! \brief super class of all gradient function argument */
struct GradFunctionArgument {
  /*! \brief The real data */
  TBlob data;
};
/*! \brief First input to the function */
struct Input0 : GradFunctionArgument {};
/*! \brief Second input to the function */
struct Input1 : GradFunctionArgument {};

/*! \brief Ouput value of the function to the function */
struct OutValue : GradFunctionArgument {};
/*! \brief Gradient of output value */
struct OutGrad : GradFunctionArgument {};
}  // namespace arg

/*! \brief registry for function entry */
class TBlobOpRegEntry {
 public:
  typedef void (*UnaryFunction)(const TBlob &src,
                                TBlob* ret,
                                OpReqType req,
                                RunContext ctx);
  typedef TShape (*UnaryShapeInfer)(const TShape &src);
  typedef void (*UnaryGradType1)(const arg::OutGrad& out_grad,
                                 const arg::OutValue& out_value,
                                 TBlob* in_grad,
                                 OpReqType req,
                                 RunContext ctx);
  typedef void (*UnaryGradType2)(const arg::OutGrad& out_grad,
                                 const arg::Input0& in_data0,
                                 TBlob* in_grad,
                                 OpReqType req,
                                 RunContext ctx);
  /*! \brief declare self type */
  typedef TBlobOpRegEntry TSelf;
  /*! \brief name of the entry */
  std::string name;
  /*!
   * \brief set shape inference function, by default use same shape.
   * \param fshapeinfer The unary function that peforms the operation.
   */
  virtual TSelf& set_shape_infer(UnaryShapeInfer fshapeinfer) = 0;
  /*!
   * \brief set function of the function to be funary
   * \param dev_mask The device mask of the function can act on.
   * \param funary The unary function that peforms the operation.
   * \param inplace_in_out Whether do inplace optimization on in and out.
   * \param register_symbolic Whether register a symbolic operator as well.
   */
  virtual TSelf& set_function(int dev_mask,
                              UnaryFunction funary,
                              bool inplace_in_out,
                              bool register_symbolic = true) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_in_grad whether out_grad and in_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradType1 fgrad,
                              bool inplace_out_in_grad) = 0;
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradType2 fgrad,
                              bool inplace_out_in_grad) = 0;
  /*!
   * \brief Describe the function.
   * \param description The description of the function.
   * \return reference to self.
   */
  virtual TSelf& describe(const std::string &description) = 0;
  /*! \brief destructor */
  virtual ~TBlobOpRegEntry() {}
};

/*! \brief registry for TBlob functions */
class TBlobOpRegistry {
 public:
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  TBlobOpRegEntry &__REGISTER_OR_FIND__(const std::string& name);
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const TBlobOpRegEntry *Find(const std::string &name) {
    return Get()->fmap_.at(name);
  }
  /*! \return global singleton of the registry */
  static TBlobOpRegistry* Get();

 private:
  // destructor
  ~TBlobOpRegistry();
  /*! \brief internal registry map */
  std::map<std::string, TBlobOpRegEntry*> fmap_;
};

#define MXNET_REGISTER_TBLOB_FUN(Name, DEV)                             \
  static ::mxnet::common::TBlobOpRegEntry &                             \
  __make_ ## TBlobOpRegEntry ## _ ## Name ## __ ## DEV ##__ =           \
      ::mxnet::common::TBlobOpRegistry::Get()->__REGISTER_OR_FIND__(#Name)
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_TBLOB_OP_REGISTRY_H_
//===== EXPANDED: ../src/common/tblob_op_registry.h =====

//===== EXPANDIND: ../src/operator/mshadow_op.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_


namespace mxnet {
namespace op {
namespace mshadow_op {
/*! \brief identity Operation */
struct identity {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a);
  }
};

struct identity_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f);
  }
};


struct negation {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-a);
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f / (1.0f + expf(-a)));
  }
};
struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * (1.0f - a));
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a > 0.0f ? a : 0.0f);
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a > 0.0f ? 1.0f : 0.0f);
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > 0.0f ? a : a * b);
  }
};

struct xelu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > 0.0f ? 1.0f : b);
  }
};

/*! \brief Exponential Linear Unit */
struct elu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType a) {
    return DType(x > 0.0f ? x : a * (expf(x) - 1.0f));
  }
};

struct elu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType a) {
    return DType(x > 0.0f ? 1.0f : a + x);
  }
};

struct tanh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(tanhf( a ));
  }
};

struct tanh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f - a * a);
  }
};

/*! \brief SoftReLU, also known as softplus activation. */
struct softrelu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(log1pf(expf(a)));
  }
};
struct softrelu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f - expf(-a));
  }
};

struct exp {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(expf(a));
  }
};

struct log {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(logf(a));
  }
};

struct log_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f / a);
  }
};

struct cos {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(cosf(a));
  }
};

struct cos_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-sinf(a));
  }
};

struct sin {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sinf(a));
  }
};

struct sin_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(cosf(a));
  }
};
struct square {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * a);
  }
};

struct square_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(2.0f * a);
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a < b ? 1.0f : 0.0f);
  }
};

/*! \brief used for generate element of abs */
struct abs {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(fabsf(a));
  }
};

/*! \brief used for generate element of power */
struct sign {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (a < 0.0f) return DType(-1.0f);
    if (a > 0.0f) return DType(1.0f);
    return DType(0.0f);
  }
};
struct sign_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.0f);
  }
};
/*! \brief used for generate element of power */
struct power {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(powf( a, b ));
  }
};

/*! \brief used for generate element of maximum */
struct maximum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > b ? a : b);
  }
};

struct maximum_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > b ? 1 : 0);
  }
};

/*! \brief used for generate element of minimum */
struct minimum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a < b ? a : b);
  }
};
struct minimum_grad  {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a < b ? 1 : 0);
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sqrtf(a));
  }
};

struct square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.5f / a);
  }
};

/*!\ \brief used for generate element sqrt */
struct reciprocal_square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0/sqrtf(a));
  }
};

struct reciprocal_square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-(1.0 / (2.0 * a * sqrtf(a))));
  }
};

/*! \brief used for generate element of round */
struct round {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(roundf(a));
  }
};

/*! \brief used for generate element of ceil */
struct ceil {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(ceilf(a));
  }
};

/*! \brief used for generate element of floor */
struct floor {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(floorf(a));
  }
};

/*! \brief used for generate gradient of MAE loss*/
struct minus_sign {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a-b > 0.0f ? 1.0f : -1.0f);
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
//===== EXPANDED: ../src/operator/mshadow_op.h =====

//===== EXPANDIND: ../src/operator/operator_common.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file  operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_


namespace mxnet {
namespace op {
/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define Assign(out, req, exp)           \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }


/*! \brief exception throwed by InferShape error */
struct InferShapeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferShapeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*! \brief exception throwed by InferShape error */
struct InferTypeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferTypeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param shape_array the shape array to store the result
 * \param index the index of in the array
 * \param shape the infered shape
 */
#define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                   \
  {                                                                     \
    auto &out = (shape_array)[index];                                   \
    if (out.ndim() == 0) {                                              \
      out = shape;                                                      \
    } else {                                                            \
      if (out != shape) {                                               \
        std::ostringstream os;                                          \
        os << "Shape inconsistent, Provided " <<  '='<< out << ','      \
           << " inferred shape=" << shape;                              \
        throw ::mxnet::op::InferShapeError(os.str(), index);            \
      }                                                                 \
    }                                                                   \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the infered type
 */
#define TYPE_ASSIGN_CHECK(type_array, index, type)                      \
  {                                                                     \
    auto &out = (type_array)[index];                                    \
    if (out == -1) {                                                    \
      out = type;                                                       \
    } else {                                                            \
      if (out != type) {                                                \
        std::ostringstream os;                                          \
        os << "Type inconsistent, Provided " <<  '='<< out << ','       \
           << " inferred type=" << type;                                \
        throw ::mxnet::op::InferTypeError(os.str(), index);             \
      }                                                                 \
    }                                                                   \
  }

// helper macro to implement bind dispatch
#if MXNET_USE_CUDA
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_OPERATOR_COMMON_H_
//===== EXPANDED: ../src/operator/operator_common.h =====

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace ndarray {

using namespace common; // NOLINT(*)

template<typename xpu, typename OP>
void UnaryForward_(const TBlob &src,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    Assign(out, req, F<OP>(src.FlatTo2D<xpu, DType>(s)));
  });
}

// backward function that takes input value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseIn_(const arg::OutGrad& out_grad,
                         const arg::Input0& in_data0,
                         TBlob *in_grad,
                         OpReqType req,
                         RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    Assign(igrad, req,
           (F<OP>(in_data0.data.FlatTo2D<xpu, DType>(s)) *
           out_grad.data.FlatTo2D<xpu, DType>()));
  });
}

// backward function that takes output value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseOut_(const arg::OutGrad& out_grad,
                          const arg::OutValue& out_value,
                          TBlob *in_grad,
                          OpReqType req,
                          RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, out_value.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    Assign(igrad, req,
           (F<OP>(out_value.data.FlatTo2D<xpu, DType>(s)) *
           out_grad.data.FlatTo2D<xpu, DType>()));
  });
}

// return a shape of scalar
inline TShape ScalarShape(const TShape& ishape) {
  mshadow::index_t shape[] = {1};
  return TShape(shape, shape + 1);
}

template<typename xpu>
void L2Norm(const TBlob &src,
            TBlob *ret,
            OpReqType req,
            RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1> out = ret->get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 1> in =
      src.get_with_shape<xpu, 1, real_t>(mshadow::Shape1(src.shape_.Size()), s);
  mshadow::VectorDot(out, in, in);
  out = mshadow::expr::F<mxnet::op::mshadow_op::square_root>(out);
}

template<typename xpu, typename Reducer>
void Reduce(const TBlob &src,
            TBlob *ret,
            OpReqType req,
            RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1> out = ret->get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 2> in =
      src.get_with_shape<xpu, 2, real_t>(mshadow::Shape2(1, src.shape_.Size()), s);
  out = mshadow::expr::reduce_except_dim<0, Reducer>(in);
}

template<typename xpu, typename Reducer, bool get_mask>
void ReduceChannel(const TBlob &src,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> out = ret->get_with_shape<xpu, 2, real_t>(
    Shape2(src.shape_[0], src.Size()/src.shape_[0]/src.shape_[1]),
    s);
  Tensor<xpu, 3> in = src.get_with_shape<xpu, 3, real_t>(
    Shape3(src.shape_[0], src.shape_[1], src.Size()/src.shape_[0]/src.shape_[1]),
    s);
  out = reduce_with_axis<Reducer, get_mask>(in, 1);
}

// return a shape of ReduceChannel output
inline TShape ReduceChannelShape(const TShape& ishape) {
  std::vector<mshadow::index_t> shape;
  shape.push_back(ishape[0]);
  for (index_t i = 2; i < ishape.ndim(); ++i) {
    shape.push_back(ishape[i]);
  }
  return TShape(shape.begin(), shape.end());
}

// Register all unary operations here
// The true means inplace can be enabled.
// abs
MXNET_REGISTER_TBLOB_FUN(abs, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::abs>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sign>, true)
.describe("Take absolute value of the src");
// sign
MXNET_REGISTER_TBLOB_FUN(sign, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::sign>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sign_grad>, true)
.describe("Take sign value of the src");
// round
MXNET_REGISTER_TBLOB_FUN(round, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::round>, true)
.describe("Take round value of the src");
// ceil
MXNET_REGISTER_TBLOB_FUN(ceil, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::ceil>, true)
.describe("Take ceil value of the src");
// floor
MXNET_REGISTER_TBLOB_FUN(floor, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::floor>, true)
.describe("Take floor value of the src");
// square
MXNET_REGISTER_TBLOB_FUN(square, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::square_grad>, true)
.describe("Take square of the src");
// sqrt
MXNET_REGISTER_TBLOB_FUN(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square_root>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::square_root_grad>, true)
.describe("Take sqrt of the src");
// rsqrt
MXNET_REGISTER_TBLOB_FUN(rsqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::reciprocal_square_root>, true)
.set_gradient(XPU::kDevMask,
              UnaryBackwardUseIn_<XPU, op::mshadow_op::reciprocal_square_root_grad>, true)
.describe("Take rsqrt of the src");
// exp
MXNET_REGISTER_TBLOB_FUN(exp, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::exp>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::identity>, true)
.describe("Take exp of the src");
// log
MXNET_REGISTER_TBLOB_FUN(log, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::log>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::log_grad>, true)
.describe("Take log of the src");
// cos
MXNET_REGISTER_TBLOB_FUN(cos, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::cos>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::cos_grad>, true)
.describe("Take cos of the src");
// sin
MXNET_REGISTER_TBLOB_FUN(sin, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::sin>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sin_grad>, true)
.describe("Take sin of the src");
// L2 norm
MXNET_REGISTER_TBLOB_FUN(norm, XPU)
.set_function(XPU::kDevMask, L2Norm<XPU>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take L2 norm of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Max
MXNET_REGISTER_TBLOB_FUN(max, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::maximum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take max of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Min
MXNET_REGISTER_TBLOB_FUN(min, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::minimum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take min of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Sum
MXNET_REGISTER_TBLOB_FUN(sum, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::sum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take sum of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// argmax channel
MXNET_REGISTER_TBLOB_FUN(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum, true>, false, false)
.set_shape_infer(ReduceChannelShape)
.describe("Take sum of the src."
          "The result will be ndarray of shape (1,) on the same device.");
}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
//===== EXPANDED: ../src/ndarray/unary_function-inl.h =====

//===== EXPANDED: ../src/ndarray/unary_function.cc =====

//===== EXPANDIND: ../src/ndarray/ndarray_function.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function_cpu.cc
 * \brief CPU Implementation of ndarray function.
 */

// this will be invoked by gcc and compile CPU version
//===== EXPANDIND: ../src/ndarray/ndarray_function.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_op.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_H_


namespace mxnet {
/*! \brief namespace to support all possible Ndarray operator */
namespace ndarray {
struct BinaryBase {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape == rshape) << "operands shape mismatch";
    CHECK(lshape.ndim() != 0) << "source operand have zero dimension shape";
    return lshape;
  }
};

// operators
struct Plus : public BinaryBase {
  typedef mshadow::op::plus mshadow_op;
};

struct Minus : public BinaryBase {
  typedef mshadow::op::minus mshadow_op;
};

struct Mul : public BinaryBase {
  typedef mshadow::op::mul mshadow_op;
};

struct Div : public BinaryBase {
  typedef mshadow::op::div mshadow_op;
};

struct ClipMin : public BinaryBase {
  struct mshadow_op {
    template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a, DType b) {
      if (a < b) {
        return b;
      } else {
        return a;
      }
    }
  };
};

struct ClipMax : public BinaryBase {
  struct mshadow_op {
    template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a, DType b) {
      if (a > b) {
        return b;
      } else {
        return a;
      }
    }
  };
};

struct Dot {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape.ndim() == 2 && rshape.ndim() == 2) << "dot only support 2D Array";
    CHECK_EQ(lshape[1], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    size_t target_shape[] = {lshape[0], rshape[1]};
    return TShape(target_shape, target_shape + 2);
  }
};


struct OneHotEncode {
  inline static TShape GetShape(const TShape &index, const TShape &proptype) {
    CHECK(index.ndim() == 1 && proptype.ndim() == 2) << "OneHotEncode only support 1d index.";
    CHECK_EQ(index[0], proptype[0]) << "OneHotEncode shape inconsistent";
    return proptype;
  }
};

struct MatChooseRowElem {
  inline static TShape GetShape(const TShape &lshape, const TShape &rshape) {
    CHECK(lshape.ndim() == 2 && rshape.ndim() == 1)
        << "choose_row_element only support 2D Matrix and 1D index";
    CHECK_EQ(lshape[0], rshape[0]) << "choose_row_element index and matrix shape mismatch";
    return rshape;
  }
};

// type holder for random number generators
struct UniformDistribution {};

struct GaussianDistribution {};

template<typename Device>
void EvalClip(const TBlob &src, const real_t &a_min, const real_t &a_max,
              TBlob *ret, RunContext ctx);

template<typename Device, typename OP>
void Eval(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx);

template<typename Device, typename OP>
void Eval(const TBlob &src, TBlob *ret, RunContext ctx);

template<typename Device, typename OP, bool reverse>
void Eval(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx);

template<typename Device>
void Eval(const real_t &rhs, TBlob *ret, RunContext ctx);

template<typename Device, typename Distribution>
void EvalRandom(const real_t &a,
                const real_t &b,
                const Resource &resource,
                TBlob *ret,  RunContext ctx);

// copy function when only cpu is involved
template<typename DeviceFrom, typename DeviceTo>
void Copy(const TBlob &from, TBlob *to,
          Context from_ctx, Context to_ctx,
          RunContext ctx);

template<typename Device>
void ElementwiseSum(const std::vector<TBlob> source,
                    TBlob *out,
                    RunContext ctx);

}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_H_
//===== EXPANDED: ../src/ndarray/ndarray_function.h =====

//===== EXPANDIND: ../src/ndarray/ndarray_function-inl.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function-inl.h
 * \brief The real implementation of NDArray functions.
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_

// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function
#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                       \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                   \
  }
#endif

#ifndef DECL_SCALAR
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                              \
  template<>                                                            \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                          \
  }
#endif

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {
// true implementation
template<typename xpu, typename OP>
inline void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s),
                                   rhs.FlatTo2D<xpu, DType>(s));
  });
}

template<typename xpu, typename OP>
inline void EvalDot_(const TBlob &lhs, const TBlob &rhs,
                     TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s) = dot(lhs.FlatTo2D<xpu, DType>(s),
                                       rhs.FlatTo2D<xpu, DType>(s));
  });
}

template<typename xpu, typename OP>
inline void EvalOneHot_(const TBlob &index, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type encoding, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(index.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  ret->get<xpu, 2, real_t>(s) =
    one_hot_encode(index.get<xpu, 1, real_t>(s),
                   rhs.shape_[1]);
}

template<typename xpu, typename OP>
inline void EvalMatChooseRowElem_(const TBlob &lhs, const TBlob &rhs,
                                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type choose, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  ret->get<xpu, 1, real_t>(s)
      = mat_choose_row_element(lhs.get<xpu, 2, real_t>(s),
                               rhs.get<xpu, 1, real_t>(s));
}

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  if (reverse) {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(scalar(DType(rhs)), lhs.FlatTo2D<xpu, DType>(s));
    });
  } else {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s), scalar(DType(rhs)));
    });
  }
}

template<>
void EvalClip<DEVICE>(const TBlob &src, const real_t &a_min, const real_t &a_max,
                      TBlob *ret, RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<ClipMax::mshadow_op>(
          F<ClipMin::mshadow_op>(src.FlatTo2D<xpu, DType>(s), scalar(DType(a_min))),
          scalar(DType(a_max)));
  });
}

template<>
void EvalRandom<DEVICE, UniformDistribution>(
    const real_t &a,
    const real_t &b,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleUniform(&tmp, float(a), float(b));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleUniform(&tmp, double(a), double(b));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGaussian(&tmp, float(mu), float(sigma));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGaussian(&tmp, double(mu), double(sigma));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void Eval<DEVICE>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  mshadow::Stream<DEVICE> *s = ctx.get_stream<DEVICE>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<DEVICE, DType>(s) = DType(rhs);
  });
}

template<>
void ElementwiseSum<DEVICE>(const std::vector<TBlob> source,
                            TBlob *dst,
                            RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  for (size_t i = 1; i < source.size(); ++i) {
    CHECK_EQ(source[i].type_flag_, dst->type_flag_)
      << "Only support input/output with the same data type";
  }
  MSHADOW_TYPE_SWITCH(dst->type_flag_, DType, {
    Tensor<xpu, 2, DType> out = dst->FlatTo2D<xpu, DType>(s);

    switch (source.size()) {
      case 2: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1;
        break;
      }
      case 3: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2;
        break;
      }
      case 4: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_3 = source[3].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2 + in_3;
        break;
      }
      default: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        out = F<mshadow::op::identity>(in_0);
        for (size_t i = 1; i < source.size(); ++i) {
          out += source[i].FlatTo2D<xpu, DType>(s);
        }
        break;
      }
    }
  });
}

// declarations
DECL_BINARY(DEVICE, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_BINARY(DEVICE, Dot, EvalDot_)
DECL_BINARY(DEVICE, OneHotEncode, EvalOneHot_)
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, true)
DECL_SCALAR(DEVICE, Div, EvalScalar_, true)
// for reverse seq
DECL_SCALAR(DEVICE, Plus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, false)
DECL_SCALAR(DEVICE, Div, EvalScalar_, false)
}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
//===== EXPANDED: ../src/ndarray/ndarray_function-inl.h =====


namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo2D<cpu, DType>(),
                      from.FlatTo2D<cpu, DType>());
    } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
            to->FlatTo2D<cpu, DType>() =
                mshadow::expr::tcast<DType>(from.FlatTo2D<cpu, SrcDType>());
        })
    }
  })
}
}  // namespace ndarray
}  // namespace mxnet
//===== EXPANDED: ../src/ndarray/ndarray_function.cc =====

//===== EXPANDIND: ../src/ndarray/ndarray.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief ndarry module of mxnet
 */
//===== EXPANDIND: ../include/mxnet/ndarray.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

//===== EXPANDIND: ../include/mxnet/storage.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_


namespace mxnet {

/*!
 * \brief Storage manager across multiple devices.
 */
class MXNET_API Storage {
 public:
  /*!
   * \brief Storage handle.
   */
  struct Handle {
    /*!
     * \brief Pointer to the data.
     */
    void* dptr;
    /*!
     * \brief Size of the storage.
     */
    size_t size;
    /*!
     * \brief Context information about device and ID.
     */
    Context ctx;
  };
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param size Total size of memory in bytes.
   * \param ctx Context information about the device and ID.
   * \return Handle struct.
   */
  virtual Handle Alloc(size_t size, Context ctx) = 0;
  /*!
   * \brief Free storage.
   * \param handle Handle struect.
   */
  virtual void Free(Handle handle) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~Storage() {}
  /*!
   * \return Storage singleton.
   */
  static Storage* Get();
  /*!
   * \brief Get shared pointer reference to engine singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  Storage to be destructed after X.
   *
   * \return A shared pointer to Storage singleton.
   */
  static std::shared_ptr<Storage> _GetSharedRef();
};  // class Storage
}  // namespace mxnet
#endif  // MXNET_STORAGE_H_
//===== EXPANDED: ../include/mxnet/storage.h =====


// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {
/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default cosntructor */
  NDArray() {}
  /*!
   * \brief constructing a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const TShape &shape, Context ctx,
          bool delay_alloc = false, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(shape.Size(), ctx, delay_alloc, dtype)),
        shape_(shape), offset_(0), dtype_(dtype) {
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_), offset_(0),
        dtype_(data.type_flag_) {
  }
  /*!
   * \return the shape of current NDArray
   */
  inline const TShape &shape() const {
    return shape_;
  }
  /*!
   * \return the data TBlob
   */
  inline TBlob data() const {
    MSHADOW_TYPE_SWITCH(dtype_, DType, {
      return TBlob(static_cast<DType*>(ptr_->shandle.dptr)
        + offset_, shape_, ptr_->shandle.ctx.dev_mask());
    });
    return TBlob();
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*!
   * \return the data type of NDArray, this function is only valid when the NDArray is not empty
   */
  inline int dtype() const {
    return dtype_;
  }
  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  inline void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  inline void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushSync([](RunContext) {}, Context{}, {}, {ptr_->var});
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the ndarray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);
  /*!
   * \brief return transpose of current NDArray
   * \return a new transposed NDArray
   */
  NDArray T() const;
  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the size of the source array, in sizeof(DType) not raw btyes.
   */
  void SyncCopyFromCPU(const void *data, size_t size) const;
  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into, in sizeof(DType) not raw btyes.
   */
  void SyncCopyToCPU(void *data, size_t size) const;
  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim
   * \param end end index in first dim
   * \return sliced NDArray
   */
  inline NDArray Slice(index_t begin, index_t end) const {
    NDArray ret = *this;
    CHECK(!is_none()) << "NDArray is not initialized";
    CHECK_GE(shape_[0], end) << "Slice end index out of range";
    size_t length = 1;
    for (index_t i = 1; i < shape_.ndim(); ++i) {
      length *= shape_[i];
    }
    ret.offset_ += begin * length;
    ret.shape_[0] = end - begin;
    return ret;
  }
  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  inline NDArray Reshape(const TShape &shape) const {
    CHECK_GE(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape size is different from current shape";
    NDArray ret = *this;
    ret.shape_ = shape;
    return ret;
  }
  /*!
   * \brief Allocate the space if it is delayed allocated.
   * This is an internal function used by system that normal user should not use
   */
  inline void CheckAndAlloc() const {
    ptr_->CheckAndAlloc();
  }
  /*!
   * \brief Save list of narray into the Stream.x
   * \param fo The stream of output.
   * \param data the NDArrays to be saved.
   * \param names the name of the NDArray, optional, can be zero length.
   */
  static void Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names);
  /*!
   * \brief Load list of narray into from the stream.
   * \param fi The stream of the input file.
   * \param data the NDArrays to be loaded
   * \param keys the name of the NDArray, if saved in the file.
   */
  static void Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys);

 private:
  /*! \brief the real data chunk that backs NDArray */
  struct Chunk {
    /*! \brief storage handlefrom storage engine */
    Storage::Handle shandle;
    /*! \brief variable from engine */
    Engine::VarHandle var;
    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    bool static_data;
    /*! \brief whether allocation is delayed */
    bool delay_alloc;
    /*! \brief default cosntructor */
    Chunk() : static_data(true), delay_alloc(false) {
      var  = Engine::Get()->NewVariable();
    }
    /*! \brief construct from static data */
    Chunk(const TBlob &data, int dev_id)
        : static_data(true),
          delay_alloc(false) {
      var = Engine::Get()->NewVariable();
      if (data.dev_mask_ == cpu::kDevMask) {
        shandle.ctx = Context::CPU();
      } else {
        CHECK_EQ(data.dev_mask_, gpu::kDevMask);
        shandle.ctx = Context::GPU(dev_id);
      }
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * mshadow::mshadow_sizeof(data.type_flag_);
    }
    /*! \brief construct a new chunk */
    Chunk(uint64_t size, Context ctx, bool delay_alloc_, int dtype)
        : static_data(false), delay_alloc(true) {
      var = Engine::Get()->NewVariable();
      shandle.size = size * mshadow::mshadow_sizeof(dtype);
      shandle.ctx = ctx;
      if (!delay_alloc_) this->CheckAndAlloc();
    }
    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(shandle.size, shandle.ctx);
        delay_alloc = false;
      }
    }
    /*! \brief destructor */
    ~Chunk() {
      if (static_data || delay_alloc) {
        Engine::Get()->DeleteVariable([](RunContext s) {}, shandle.ctx, var);
      } else {
        Storage::Handle h = this->shandle;
        Engine::Get()->DeleteVariable([h](RunContext s) {
            Storage::Get()->Free(h);
          }, shandle.ctx, var);
      }
    }
  };
  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_;
  /*! \brief shape of current NDArray */
  TShape shape_;
  /*! \brief offset in chunk */
  size_t offset_;
  /*! \brief type of data */
  int dtype_;
};

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, NDArray *to, int priority = 0);

/*!
 * \brief Perform elementwise sum over each data from source, store result into out.
 * \param source the ndarray we want to sum
 * \param out the target ndarray
 * \param priority Priority of the action.
 */
void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority = 0);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const NDArray &rhs); \
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const real_t &rhs);

/*!
 * \brief Seed the random number generator.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
/*!
 * \brief Sample uniform distribution for each elements of out.
 * \param begin lower bound of distribution.
 * \param end upper bound of distribution.
 * \param out output NDArray.
 */
void SampleUniform(real_t begin, real_t end, NDArray *out);

/*!
 * \brief Sample gaussian distribution for each elements of out.
 * \param mu mean of gaussian distribution.
 * \param sigma standard deviation of gaussian distribution.
 * \param out output NDArray.
 */
void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
//--------------------------------------------------------------
// The following part are API Registration of NDArray functions.
//--------------------------------------------------------------
/*! \brief definition of NDArray function */
typedef std::function<void (NDArray **used_vars,
                            real_t *scalars,
                            NDArray **mutate_vars,
                            int num_params,
                            char **param_keys,
                            char **param_vals)> NDArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NDArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNDArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNDArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NDArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NDArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NDArrayFunction */
struct NDArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NDArrayFunctionReg,
                                        NDArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NDArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NDArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fsetvalue)(const real_t &rhs,
                                                            NDArray *out)) {
    body = [fsetvalue] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                        int num_params, char **param_keys, char **param_vals) {
      (*fsetvalue)(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    this->add_argument("src", "real_t", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fbinary)(const NDArray &lhs,
                                                          const NDArray &rhs,
                                                          NDArray *out)) {
    body = [fbinary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fbinary)(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fscalar)(const NDArray &lhs,
                                                          const real_t &rhs,
                                                          NDArray *out)) {
    body = [fscalar] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fscalar)(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*funary)(const NDArray &src,
                                                         NDArray *out)) {
    body = [funary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                     int num_params, char **param_keys, char **param_vals) {
      (*funary)(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NDArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param fgeneric function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(
    void (*fgeneric)(NDArray **used_vars,
                     real_t *s,
                     NDArray **mutate_vars,
                     const std::map<std::string, std::string>& param)) {
    body = [fgeneric] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                       int num_params, char **param_keys, char **param_vals) {
      std::map<std::string, std::string> param;
      for (int i = 0; i < num_params; ++i) {
        param[param_keys[i]] = param_vals[i];
      }
      fgeneric(used_vars, s, mutate_vars, param);
    };
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NDArrayFunctionReg

/*!
 * \brief Macro to register NDArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NDARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NDARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NDArrayFunctionReg, NDArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NDArray, true);
}  // namespace dmlc
#endif  // MXNET_NDARRAY_H_
//===== EXPANDED: ../include/mxnet/ndarray.h =====


namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NDArrayFunctionReg);
}  // namespace dmlc

namespace mxnet {
/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP>
void BinaryOp(const NDArray &lhs,
              const NDArray &rhs,
              NDArray *out) {
  // no check if both of them are on cpu
  if (lhs.ctx().dev_mask() != cpu::kDevMask || rhs.ctx().dev_mask() != cpu::kDevMask) {
    CHECK(lhs.ctx() == rhs.ctx()) << "operands context mismatch";
  }
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NDArray(OP::GetShape(lhs.shape(), rhs.shape()), lhs.ctx(), true, lhs.dtype());
  } else {
    // no check if both of them are on cpu
    if (lhs.ctx().dev_mask() != cpu::kDevMask ||
        out->ctx().dev_mask() != cpu::kDevMask) {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    }
    CHECK(out->shape() == OP::GetShape(lhs.shape(), rhs.shape()))
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
  if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void SetValueOp(const real_t &rhs, NDArray *out) {
  CHECK_NE(out->is_none(), true) << "Set value target must not be empty";
  // important: callback must always capture by value
  NDArray ret = *out;
  switch (ret.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu>(rhs, &tmp, ctx);
        }, ret.ctx(), {}, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu>(rhs, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, ret.ctx(), {}, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP, bool reverse>
void ScalarOp(const NDArray &lhs,
              const real_t &rhs,
              NDArray *out) {
  if (out->is_none()) {
    *out = NDArray(lhs.shape(), lhs.ctx(), true, lhs.dtype());
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == lhs.shape()) << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void CopyFromTo(const NDArray &from, NDArray *to, int priority) {
  CHECK(from.shape() == to->shape())
      << "operands shape mismatch";
  CHECK(from.shape().ndim() != 0)
      << "source operands have zero dimension shape";
  // important: callback must always capture by value
  NDArray ret = *to;
  int a = from.ctx().dev_mask();
  int b = to->ctx().dev_mask();

  std::vector<Engine::VarHandle> const_vars;
  if (from.var() != ret.var()) const_vars.push_back(from.var());

  if (a == cpu::kDevMask && b == cpu::kDevMask) {
    Engine::Get()->PushSync([from, ret](RunContext ctx) {
        ret.CheckAndAlloc();
        TBlob tmp = ret.data();
        ndarray::Copy<cpu, cpu>(from.data(), &tmp,
                                from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), const_vars, {ret.var()},
      FnProperty::kNormal, priority);
  } else {
#if MXNET_USE_CUDA
    if (a == cpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushSync([from, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Copy<cpu, gpu>(from.data(), &tmp,
                                  from.ctx(), ret.ctx(), ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, ret.ctx(), const_vars, {ret.var()},
        FnProperty::kCopyToGPU, priority);
    } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
      Engine::Get()->PushSync([from, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Copy<gpu, cpu>(from.data(), &tmp,
                                  from.ctx(), ret.ctx(), ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, from.ctx(), const_vars, {ret.var()},
        FnProperty::kCopyFromGPU, priority);
    } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushSync([from, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Copy<gpu, gpu>(from.data(), &tmp,
                                  from.ctx(), ret.ctx(), ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, from.ctx(), const_vars, {ret.var()},
        FnProperty::kCopyFromGPU, priority);
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority) {
  std::vector<Engine::VarHandle> const_vars;
  const_vars.reserve(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    if (source[i].var() != out->var()) {
      const_vars.push_back(source[i].var());
    }
    CHECK_EQ(source[i].shape() , out->shape())
        << "operands shape mismatch";
    if (out->ctx().dev_mask() == cpu::kDevMask) {
      CHECK_EQ(source[i].ctx().dev_mask(),  cpu::kDevMask)
          << "operands context mismatch";
    } else {
      CHECK(source[i].ctx() == out->ctx())
          << "operands context mismatch";
    }
  }
  // important: callback must always capture by value
  NDArray ret = *out;

  switch (out->ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([source, ret](RunContext ctx) {
          std::vector<TBlob> source_tblob(source.size());
          for (size_t i = 0; i < source.size(); ++i) {
            source_tblob[i] = source[i].data();
          }
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::ElementwiseSum<cpu>(source_tblob, &tmp, ctx);
        }, out->ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, priority);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([source, ret](RunContext ctx) {
          std::vector<TBlob> source_tblob(source.size());
          for (size_t i = 0; i < source.size(); ++i) {
            source_tblob[i] = source[i].data();
          }
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::ElementwiseSum<gpu>(source_tblob, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, out->ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, priority);
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void ClipOp(const NDArray &src,
            const real_t &a_min, const real_t &a_max,
            NDArray *out) {
  if (out->is_none()) {
    *out = NDArray(src.shape(), src.ctx(), true, src.dtype());
  } else {
    CHECK(out->ctx() == src.ctx()) << "target context mismatch";
    CHECK(out->shape() == src.shape()) << "target shape mismatch";
  }
  NDArray ret = *out;
  std::vector<Engine::VarHandle> const_vars;
  if (src.var() != ret.var()) const_vars.push_back(src.var());
  switch (src.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([src, a_min, a_max, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::EvalClip<cpu>(src.data(), a_min, a_max, &tmp, ctx);
        }, src.ctx(), const_vars, {ret.var()});
      break;
    }
    #if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([src, a_min, a_max, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::EvalClip<gpu>(src.data(), a_min, a_max, &tmp, ctx);
        }, src.ctx(), const_vars, {ret.var()});
      break;
    }
    #endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

inline void CopyFromToSimple(const NDArray &from, NDArray *to) {
  CopyFromTo(from, to, 0);
}

template<typename Distribution>
void SampleOP(const real_t &a,
              const real_t &b,
              NDArray *out) {
  CHECK(!out->is_none());
  Resource resource = ResourceManager::Get()->Request(
      out->ctx(), ResourceRequest::kRandom);
  // important: callback must always capture by value
  NDArray ret = *out;
  // redirect everything to mshadow operations
  switch (out->ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([a, b, resource, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::EvalRandom<cpu, Distribution>(a, b, resource, &tmp, ctx);
        }, out->ctx(), {}, {ret.var(), resource.var});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([a, b, resource, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::EvalRandom<gpu, Distribution>(a, b, resource, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, out->ctx(), {}, {ret.var(), resource.var});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void SampleUniform(real_t begin, real_t end, NDArray *out) {
  SampleOP<ndarray::UniformDistribution>(begin, end, out);
}

void SampleGaussian(real_t mu, real_t sigma, NDArray *out) {
  SampleOP<ndarray::GaussianDistribution>(mu, sigma, out);
}

void RandomSeed(uint32_t seed) {
  ResourceManager::Get()->SeedRandom(seed);
}

template<typename OP>
inline NDArray BinaryOpRet(const NDArray &lhs,
                           const NDArray &rhs) {
  NDArray ret;
  BinaryOp<OP>(lhs, rhs, &ret);
  return ret;
}

template<typename OP, bool reverse>
inline NDArray ScalarOpRet(const NDArray &lhs,
                           const real_t &rhs) {
  NDArray ret;
  ScalarOp<OP, reverse>(lhs, rhs, &ret);
  return ret;
}

template<typename OP>
inline NDArray &BinaryOpApply(NDArray *dst,
                              const NDArray &src) {
  BinaryOp<OP>(*dst, src, dst);
  return *dst;
}

template<typename OP>
inline NDArray &ScalarOpApply(NDArray *dst,
                             const real_t &src) {
  ScalarOp<OP, false>(*dst, src, dst);
  return *dst;
}

// Binary
NDArray operator+(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Plus>(lhs, rhs);
}
NDArray operator-(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Minus>(lhs, rhs);
}
NDArray operator*(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Mul>(lhs, rhs);
}
NDArray operator/(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Div>(lhs, rhs);
}
// Scalar
NDArray operator+(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Plus, false>(lhs, rhs);
}
NDArray operator-(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Minus, false>(lhs, rhs);
}
NDArray operator*(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Mul, false>(lhs, rhs);
}
NDArray operator/(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Div, false>(lhs, rhs);
}

// Binary
NDArray &NDArray::operator=(real_t scalar) {
  SetValueOp(scalar, this);
  return *this;
}

NDArray &NDArray::operator+=(const NDArray &src) {
  return BinaryOpApply<ndarray::Plus>(this, src);
}
NDArray &NDArray::operator-=(const NDArray &src) {
  return BinaryOpApply<ndarray::Minus>(this, src);
}
NDArray &NDArray::operator*=(const NDArray &src) {
  return BinaryOpApply<ndarray::Mul>(this, src);
}
NDArray &NDArray::operator/=(const NDArray &src) {
  return BinaryOpApply<ndarray::Div>(this, src);
}
// Scalar
NDArray &NDArray::operator+=(const real_t &src) {
  return ScalarOpApply<ndarray::Plus>(this, src);
}
NDArray &NDArray::operator-=(const real_t &src) {
  return ScalarOpApply<ndarray::Minus>(this, src);
}
NDArray &NDArray::operator*=(const real_t &src) {
  return ScalarOpApply<ndarray::Mul>(this, src);
}
NDArray &NDArray::operator/=(const real_t &src) {
  return ScalarOpApply<ndarray::Div>(this, src);
}

void NDArray::Save(dmlc::Stream *strm) const {
  // save shape
  shape_.Save(strm);
  if (is_none()) return;
  // save context
  Context ctx = this->ctx();
  ctx.Save(strm);
  TBlob save_data;
  NDArray temp;
  if (ctx.dev_mask() != cpu::kDevMask) {
    temp = this->Copy(Context::CPU());
    temp.WaitToRead();
    save_data = temp.data();
  } else {
    this->WaitToRead();
    save_data = this->data();
  }
  // save type flag
  int32_t type_flag = save_data.type_flag_;
  strm->Write(&type_flag, sizeof(type_flag));
  CHECK(save_data.CheckContiguous());
  size_t type_size = mshadow::mshadow_sizeof(type_flag);
  strm->Write(save_data.dptr_, type_size * shape_.Size());
}

bool NDArray::Load(dmlc::Stream *strm) {
  // load shape
  TShape shape;
  if (!shape.Load(strm)) return false;
  if (shape.ndim() == 0) {
    *this = NDArray(); return true;
  }
  // load context
  Context ctx;
  if (!ctx.Load(strm)) return false;
  // load type flag
  int32_t type_flag;
  if (strm->Read(&type_flag, sizeof(type_flag)) != sizeof(type_flag)) return false;
  // load data into CPU
  NDArray temp(shape, Context::CPU(), false, type_flag);
  TBlob load_data = temp.data();
  size_t type_size = mshadow::mshadow_sizeof(type_flag);
  size_t nread = type_size * shape.Size();

  if (strm->Read(load_data.dptr_, nread) != nread) return false;
  if (ctx.dev_mask() == cpu::kDevMask) {
    *this = std::move(temp); return true;
  } else {
    *this = temp.Copy(ctx); return true;
  }
}


const uint64_t kMXAPINDArrayListMagic = 0x112;

void NDArray::Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names) {
  uint64_t header = kMXAPINDArrayListMagic, reserved = 0;
  fo->Write(&header, sizeof(header));
  fo->Write(&reserved, sizeof(reserved));
  fo->Write(data);
  fo->Write(names);
}

void NDArray::Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys) {
  uint64_t header, reserved;
  CHECK(fi->Read(&header))
      << "Invalid NDArray file format";
  CHECK(fi->Read(&reserved))
      << "Invalid NDArray file format";
  CHECK(header == kMXAPINDArrayListMagic)
      << "Invalid NDArray file format";
  CHECK(fi->Read(data))
      << "Invalid NDArray file format";
  CHECK(fi->Read(keys))
      << "Invalid NDArray file format";
  CHECK(keys->size() == 0 || keys->size() == data->size())
      << "Invalid NDArray file format";
}

NDArray NDArray::Copy(Context ctx) const {
  NDArray ret(shape(), ctx, true, dtype_);
  CopyFromTo(*this, &ret);
  return ret;
}

void NDArray::SyncCopyFromCPU(const void *data, size_t size) const {
  this->WaitToWrite();
  TShape dshape = this->shape();
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  Context ctx = this->ctx();
  TBlob dst = this->data();
  TBlob src((void*)data, dshape, cpu::kDevMask, this->dtype_); // NOLINT(*)

  RunContext run_ctx;
  run_ctx.stream = nullptr;
  if (ctx.dev_mask() == cpu::kDevMask) {
    ndarray::Copy<cpu, cpu>(src, &dst, Context::CPU(), ctx, run_ctx);
  } else {
#if MXNET_USE_CUDA
    // use empty stream to do sync copy
    // TODO(bing, yutian) consider use a Real Stream, so it is not blocking others
    // Maybe move to engine part
    mshadow::Stream<gpu> zero_stream;
    run_ctx.stream = &zero_stream;
    ndarray::Copy<cpu, gpu>(src, &dst, Context::CPU(), ctx, run_ctx);
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
}

void NDArray::SyncCopyToCPU(void *data, size_t size) const {
  this->WaitToRead();
  TShape dshape = this->shape();
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  Context ctx = this->ctx();
  TBlob src = this->data();
  TBlob dst(data, dshape, cpu::kDevMask, this->dtype_); // NOLINT(*)

  RunContext run_ctx;
  run_ctx.stream = nullptr;
  if (ctx.dev_mask() == cpu::kDevMask) {
    ndarray::Copy<cpu, cpu>(src, &dst, ctx, Context::CPU(), run_ctx);
  } else {
#if MXNET_USE_CUDA
    // use empty stream to do sync copy
    // TODO(bing, yutian) consider use a Real Stream, so it is not blocking others
    // Maybe move to engine part
    mshadow::Stream<gpu> zero_stream;
    run_ctx.stream = &zero_stream;
    ndarray::Copy<gpu, cpu>(src, &dst, ctx, Context::CPU(), run_ctx);
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
}

#if MXNET_PREDICT_ONLY == 0
// register API function
// those with underscore will be registered at NDArray
MXNET_REGISTER_NDARRAY_FUN(_set_value).set_function(SetValueOp);


MXNET_REGISTER_NDARRAY_FUN(_plus).set_function(BinaryOp<ndarray::Plus>);
MXNET_REGISTER_NDARRAY_FUN(_minus).set_function(BinaryOp<ndarray::Minus>);
MXNET_REGISTER_NDARRAY_FUN(_mul).set_function(BinaryOp<ndarray::Mul>);
MXNET_REGISTER_NDARRAY_FUN(_div).set_function(BinaryOp<ndarray::Div>);

MXNET_REGISTER_NDARRAY_FUN(dot).set_function(BinaryOp<ndarray::Dot>)
.describe("Calcuate 2D matrix multiplication");

MXNET_REGISTER_NDARRAY_FUN(_onehot_encode).set_function(BinaryOp<ndarray::OneHotEncode>);

MXNET_REGISTER_NDARRAY_FUN(choose_element_0index)
.set_function(BinaryOp<ndarray::MatChooseRowElem>)
.describe("Choose one element from each line(row for python, column for R/Julia)"
          " in lhs according to index indicated by rhs."
          " This function assume rhs uses 0-based index.");

// register API function
// those with underscore will be registered at NDArray
MXNET_REGISTER_NDARRAY_FUN(_plus_scalar).set_function(ScalarOp<ndarray::Plus, false>);
MXNET_REGISTER_NDARRAY_FUN(_minus_scalar).set_function(ScalarOp<ndarray::Minus, false>);
MXNET_REGISTER_NDARRAY_FUN(_mul_scalar).set_function(ScalarOp<ndarray::Mul, false>);
MXNET_REGISTER_NDARRAY_FUN(_div_scalar).set_function(ScalarOp<ndarray::Div, false>);
// register API function
// scalar, reverse scalar
MXNET_REGISTER_NDARRAY_FUN(_rminus_scalar).set_function(ScalarOp<ndarray::Minus, true>);
MXNET_REGISTER_NDARRAY_FUN(_rdiv_scalar).set_function(ScalarOp<ndarray::Div, true>);

// copy function is special
// that we need to remove kAcceptEmptyMutateTarget from it
MXNET_REGISTER_NDARRAY_FUN(_copyto)
.set_function(CopyFromToSimple)
.set_type_mask(kNDArrayArgBeforeScalar);

// register random number generators
MXNET_REGISTER_NDARRAY_FUN(_random_uniform)
.set_body([](NDArray **u, real_t *s, NDArray **out,
             int num_params, char **param_keys, char **param_vals) {
    SampleUniform(s[0], s[1], out[0]);
  })
.set_num_scalars(2)
.set_num_mutate_vars(1);

MXNET_REGISTER_NDARRAY_FUN(_random_gaussian)
.set_body([](NDArray **u, real_t *s, NDArray **out,
             int num_params, char **param_keys, char **param_vals) {
    SampleGaussian(s[0], s[1], out[0]);
  })
.set_num_scalars(2)
.set_num_mutate_vars(1);

MXNET_REGISTER_NDARRAY_FUN(clip)
.set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
.set_body([](NDArray **u, real_t *s, NDArray **out,
             int num_params, char **param_keys, char **param_vals) {
    ClipOp(*u[0], s[0], s[1], out[0]);
  })
.set_num_use_vars(1)
.set_num_scalars(2)
.set_num_mutate_vars(1)
.describe("Clip ndarray elements to range (a_min, a_max)")
.add_argument("src", "NDArray", "Source input")
.add_argument("a_min", "real_t", "Minimum value")
.add_argument("a_max", "real_t", "Maximum value");
#endif
}  // namespace mxnet
//===== EXPANDED: ../src/ndarray/ndarray.cc =====

//===== EXPANDIND: ../src/engine/engine.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file engine.cc
 * \brief Implementation of engine.
 */
//===== EXPANDIND: ../src/engine/engine_impl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file engine_impl.h
 * \brief Internal implementation header of engine components.
 */
#ifndef MXNET_ENGINE_ENGINE_IMPL_H_
#define MXNET_ENGINE_ENGINE_IMPL_H_


/*! \brief MACRO on whether or not enable debug option*/
#define ENGINE_DEBUG 0

namespace mxnet {
namespace engine {

/*! \brief base class of engine variables, used for type checking */
struct Var {
#if ENGINE_DEBUG
  virtual ~Var() = default;
#endif  // ENGINE_DEBUG
  /*!
   * \brief cast variable to derived type T
   * \tparam T the type we want to cast into.
   * \return A casted variable.
   */
  template <typename T>
  inline T* Cast();
};  // struct Var

/*! \brief base class of engine operators, used for type checking */
struct Opr {
#if ENGINE_DEBUG
  virtual ~Opr() = default;
#endif
  /*!
   * \brief cast variable to derived type T
   * \tparam T the type we want to cast into.
   * \return A casted variable.
   */
  template <typename T>
  inline T* Cast();
};  // struct Opr

// implementation of the inline functions
template <typename T>
inline T* Var::Cast() {
  static_assert(std::is_base_of<Var, T>::value,
                "must inherit `mxnet::engine::Var`");
#if ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else
  return static_cast<T*>(this);
#endif
}

template <typename T>
inline T* Opr::Cast() {
  static_assert(std::is_base_of<Opr, T>::value,
                "must inherit `mxnet::engine::Opr`");
#if ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else
  return static_cast<T*>(this);
#endif
}

/*! \brief Maximum number of GPUs */
static constexpr std::size_t kMaxNumGPUs = 16;

// predeclare factory function for each type of engine
/*! \return NaiveEngine instance */
Engine *CreateNaiveEngine();
#if MXNET_PREDICT_ONLY == 0
/*! \return ThreadedEnginePooled instance */
Engine *CreateThreadedEnginePooled();
/*! \return ThreadedEnginePerDevie instance */
Engine *CreateThreadedEnginePerDevice();
#endif
}  // namespace engine
}  // namespace mxnet
#endif  // MXNET_ENGINE_ENGINE_IMPL_H_
//===== EXPANDED: ../src/engine/engine_impl.h =====


namespace mxnet {
namespace engine {
inline Engine* CreateEngine() {
  const char *type = getenv("MXNET_ENGINE_TYPE");
  const bool default_engine = (type == nullptr);
  if (type == nullptr) type = "ThreadedEnginePerDevice";
  std::string stype = type;

  Engine *ret = nullptr;
  #if MXNET_PREDICT_ONLY == 0
  if (stype == "NaiveEngine") {
    ret = CreateNaiveEngine();
  } else if (stype == "ThreadedEngine") {
    ret = CreateThreadedEnginePooled();
  } else if (stype == "ThreadedEnginePerDevice") {
    ret = CreateThreadedEnginePerDevice();
  }
  #else
  ret = CreateNaiveEngine();
  #endif

  if (ret ==nullptr) {
    LOG(FATAL) << "Cannot find Engine " << type;
  }
  if (!default_engine) {
    LOG(INFO) << "MXNet start using engine: " << type;
  }
  return ret;
}
}  // namespace engine

std::shared_ptr<Engine> Engine::_GetSharedRef() {
  static std::shared_ptr<Engine> sptr(engine::CreateEngine());
  return sptr;
}

Engine* Engine::Get() {
  static Engine *inst = _GetSharedRef().get();
  return inst;
}
}  // namespace mxnet
//===== EXPANDED: ../src/engine/engine.cc =====

//===== EXPANDIND: ../src/engine/naive_engine.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file naive_engine.cc
 * \brief Implementation of NaiveEngine
 */

namespace mxnet {
namespace engine {

// implement naive engine
class NaiveEngine final : public Engine {
 public:
  struct NaiveOpr : public Opr {
    AsyncFn fn;
    std::vector<VarHandle> const_vars;
    std::vector<VarHandle> mutable_vars;
    FnProperty prop;
  };

  NaiveEngine() {
  }
  // virtual destructor
  virtual ~NaiveEngine() {
#if MXNET_USE_CUDA
    LOG(INFO) << "Engine shutdown";
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream(streams_[i]));
        streams_[i] = nullptr;
      }
    }
#endif
  }

  // new variables
  VarHandle NewVariable() override {
    return nullptr;
  }
  OprHandle NewOperator(AsyncFn fn,
                        std::vector<VarHandle> const& const_vars,
                        std::vector<VarHandle> const& mutable_vars,
                        FnProperty prop) override {
    NaiveOpr *opr = new NaiveOpr();
    opr->fn = fn;
    opr->const_vars = const_vars;
    opr->mutable_vars = mutable_vars;
    opr->prop = prop;
    return opr;
  }
  void DeleteOperator(OprHandle op) override {
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    delete opr;
  }
  void Push(OprHandle op, Context exec_ctx, int priority) override {
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    this->PushAsync(opr->fn,
                    exec_ctx,
                    opr->const_vars,
                    opr->mutable_vars,
                    opr->prop);
  }
  void PushAsync(AsyncFn exec_fun,
                 Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop,
                 int priority = 0) override {
    CallbackOnComplete callback = CreateCallback(
        NaiveEngine::OnComplete, nullptr);
    this->req_completed_ = false;

    if (exec_ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(exec_ctx.dev_id));
      if (streams_.size() <= dev_id) {
        streams_.resize(dev_id + 1, nullptr);
      }
      if (streams_[dev_id] == nullptr) {
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
      }
      ctx_.stream = streams_[dev_id];
      exec_fun(ctx_, callback);
#else
      LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      ctx_.stream = &cpu_stream_;
      exec_fun(ctx_, callback);
    }
    CHECK(this->req_completed_)
        << "NaiveEngine only support synchronize Push so far";
  }
  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override {
    this->PushSync(delete_fn, exec_ctx, {}, {var}, FnProperty::kNormal);
  }
  void WaitForVar(VarHandle var) override {
  }
  void WaitForAll() override {
  }
  void NotifyShutdown() override {
    shutdown_phase_.store(true);
  }

 private:
  // callback to oncomplete
  static void OnComplete(Engine *engine, void *param) {
    static_cast<NaiveEngine*>(engine)->req_completed_ = true;
  }
  // runtime contetxt
  RunContext ctx_;
  // whether action is completed
  bool req_completed_;
  /*! \brief whether it is during shutdown phase*/
  std::atomic<bool> shutdown_phase_{false};
  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
};  // class NaiveEngine


Engine *CreateNaiveEngine() {
  return new NaiveEngine();
}
}  // namespace engine
}  // namespace mxnet
//===== EXPANDED: ../src/engine/naive_engine.cc =====

//===== EXPANDIND: ../src/symbol/graph_executor.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief Executor to execute the Graph.
 */
//===== EXPANDIND: ../include/mxnet/symbolic.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file symbolic.h
 * \brief Symbolic interface of mxnet.
 * \author Min Lin, Bing Xu
 */
#ifndef MXNET_SYMBOLIC_H_
#define MXNET_SYMBOLIC_H_

//===== EXPANDIND: ../include/mxnet/c_api.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_C_API_H_
#define MXNET_C_API_H_

#ifdef __cplusplus
#define MXNET_EXTERN_C extern "C"
#endif

/*! \brief MXNET_DLL prefix for windows */
#ifdef _WIN32
#ifdef MXNET_EXPORTS
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllexport)
#else
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllimport)
#endif
#else
#define MXNET_DLL MXNET_EXTERN_C
#endif

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define unsigned int */
typedef float mx_float;
// all the handles are simply void *
// will be casted internally to specific pointers types
// these typedefs are mainly used for readablity reasons
/*! \brief handle to NDArray */
typedef void *NDArrayHandle;
/*! \brief handle to a mxnet narray function that changes NDArray */
typedef const void *FunctionHandle;
/*! \brief handle to a function that takes param and creates symbol */
typedef void *AtomicSymbolCreator;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to a AtomicSymbol */
typedef void *AtomicSymbolHandle;
/*! \brief handle to an Executor */
typedef void *ExecutorHandle;
/*! \brief handle a dataiter creator */
typedef void *DataIterCreator;
/*! \brief handle to a DataIterator */
typedef void *DataIterHandle;
/*! \brief handle to KVStore */
typedef void *KVStoreHandle;
/*! \brief handle to RecordIO */
typedef void *RecordIOHandle;
/*! \brief handle to MXRtc*/
typedef void *RtcHandle;
/*! \brief handle to a function that takes param and creates optimizer*/
typedef void *OptimizerCreator;
/*! \brief handle to Optimizer*/
typedef void *OptimizerHandle;

MXNET_EXTERN_C typedef void (*ExecutorMonitorCallback)(const char*,
                                                       NDArrayHandle,
                                                       void *);

MXNET_EXTERN_C {
struct NativeOpInfo {
  void (*forward)(int, float**, int*, unsigned**, int*, void*);
  void (*backward)(int, float**, int*, unsigned**, int*, void*);
  void (*infer_shape)(int, int*, unsigned**, void*);
  void (*list_outputs)(char***, void*);
  void (*list_arguments)(char***, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
};

struct NDArrayOpInfo {
  bool (*forward)(int, void**, int*, void*);
  bool (*backward)(int, void**, int*, void*);
  bool (*infer_shape)(int, int*, unsigned**, void*);
  bool (*list_outputs)(char***, void*);
  bool (*list_arguments)(char***, void*);
  bool (*declare_backward_dependency)(const int*, const int*, const int*,
                                      int*, int**, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
  void* p_declare_backward_dependency;
};
}
/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  MXGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
MXNET_DLL const char *MXGetLastError();

//-------------------------------------
// Part 0: Global State setups
//-------------------------------------
/*!
 * \brief Seed the global random number generators in mxnet.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXRandomSeed(int seed);
/*!
 * \brief Notify the engine about a shutdown,
 *  This can help engine to print less messages into display.
 *
 *  User do not have to call this function.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXNotifyShutdown();
//-------------------------------------
// Part 1: NDArray creation and deletion
//-------------------------------------
/*!
 * \brief create a NDArray handle that is not initialized
 *  can be used to pass in as mutate variables
 *  to hold the result of NDArray
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateNone(NDArrayHandle *out);
/*!
 * \brief create a NDArray with specified shape
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreate(const mx_uint *shape,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              NDArrayHandle *out);

/*!
 * \brief create a NDArray with specified shape and data type
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param dtype data type of created array
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateEx(const mx_uint *shape,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              int dtype,
                              NDArrayHandle *out);
/*!
 * \brief create a NDArray handle that is loaded from raw bytes.
 * \param buf the head of the raw bytes
 * \param size size of the raw bytes
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoadFromRawBytes(const void *buf,
                                        size_t size,
                                        NDArrayHandle *out);
/*!
 * \brief save the NDArray into raw bytes.
 * \param handle the NDArray handle
 * \param out_size size of the raw bytes
 * \param out_buf the head of returning memory bytes.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySaveRawBytes(NDArrayHandle handle,
                                    size_t *out_size,
                                    const char **out_buf);
/*!
 * \brief Save list of narray into the file.
 * \param fname name of the file.
 * \param num_args number of arguments to save.
 * \param args the array of NDArrayHandles to be saved.
 * \param keys the name of the NDArray, optional, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySave(const char* fname,
                            mx_uint num_args,
                            NDArrayHandle* args,
                            const char** keys);
/*!
 * \brief Load list of narray from the file.
 * \param fname name of the file.
 * \param out_size number of narray loaded.
 * \param out_arr head of the returning narray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoad(const char* fname,
                            mx_uint *out_size,
                            NDArrayHandle** out_arr,
                            mx_uint *out_name_size,
                            const char*** out_names);
/*!
 * \brief Perform a synchronize copy from a continugous CPU memory region.
 *
 *  This function will call WaitToWrite before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy from.
 * \param size the memory size we want to copy from.
 */
MXNET_DLL int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                       const void *data,
                                       size_t size);
/*!
 * \brief Perform a synchronize copyto a continugous CPU memory region.
 *
 *  This function will call WaitToRead before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy into.
 * \param size the memory size we want to copy into.
 */
MXNET_DLL int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                                     void *data,
                                     size_t size);
/*!
 * \brief Wait until all the pending writes with respect NDArray are finished.
 *  Always call this before read data out synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToRead(NDArrayHandle handle);
/*!
 * \brief Wait until all the pending read/write with respect NDArray are finished.
 *  Always call this before write data into NDArray synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToWrite(NDArrayHandle handle);
/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitAll();
/*!
 * \brief free the narray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayFree(NDArrayHandle handle);
/*!
 * \brief Slice the NDArray along axis 0.
 * \param handle the handle to the narraya
 * \param slice_begin The beginning index of slice
 * \param slice_end The ending index of slice
 * \param out The NDArrayHandle of sliced NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySlice(NDArrayHandle handle,
                             mx_uint slice_begin,
                             mx_uint slice_end,
                             NDArrayHandle *out);
/*!
 * \brief get the shape of the array
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetShape(NDArrayHandle handle,
                                mx_uint *out_dim,
                                const mx_uint **out_pdata);
/*!
 * \brief get the content of the data in NDArray
 * \param handle the handle to the narray
 * \param out_pdata pointer holder to get pointer of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetData(NDArrayHandle handle,
                               mx_float **out_pdata);
/*!
 * \brief get the type of the data in NDArray
 * \param handle the handle to the narray
 * \param out_dtype pointer holder to get type of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetDType(NDArrayHandle handle,
                               int *out_dtype);
/*!
 * \brief get the context of the NDArray
 * \param handle the handle to the narray
 * \param out_dev_type the output device type
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetContext(NDArrayHandle handle,
                                  int *out_dev_type,
                                  int *out_dev_id);

//--------------------------------
// Part 2: functions on NDArray
//--------------------------------
/*!
 * \brief list all the available functions handles
 *   most user can use it to list all the needed functions
 * \param out_size the size of returned array
 * \param out_array the output function array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListFunctions(mx_uint *out_size,
                              FunctionHandle **out_array);
/*!
 * \brief get the function handle by name
 * \param name the name of the function
 * \param out the corresponding function handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetFunction(const char *name,
                            FunctionHandle *out);
/*!
 * \brief Get the information of the function handle.
 * \param fun The function handle.
 * \param name The returned name of the function.
 * \param description The returned description of the function.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXFuncGetInfo(FunctionHandle fun,
                            const char **name,
                            const char **description,
                            mx_uint *num_args,
                            const char ***arg_names,
                            const char ***arg_type_infos,
                            const char ***arg_descriptions);
/*!
 * \brief get the argument requirements of the function
 * \param fun input function handle
 * \param num_use_vars how many NDArrays to be passed in as used_vars
 * \param num_scalars scalar variable is needed
 * \param num_mutate_vars how many NDArrays to be passed in as mutate_vars
 * \param type_mask the type mask of this function
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncInvoke
 */
MXNET_DLL int MXFuncDescribe(FunctionHandle fun,
                             mx_uint *num_use_vars,
                             mx_uint *num_scalars,
                             mx_uint *num_mutate_vars,
                             int *type_mask);
/*!
 * \brief invoke a function, the array size of passed in arguments
 *   must match the values in the
 * \param fun the function
 * \param use_vars the normal arguments passed to function
 * \param scalar_args the scalar qarguments
 * \param mutate_vars the mutate arguments
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncDescribeArgs
 */
MXNET_DLL int MXFuncInvoke(FunctionHandle fun,
                           NDArrayHandle *use_vars,
                           mx_float *scalar_args,
                           NDArrayHandle *mutate_vars);
/*!
 * \brief invoke a function, the array size of passed in arguments
 *   must match the values in the
 * \param fun the function
 * \param use_vars the normal arguments passed to function
 * \param scalar_args the scalar qarguments
 * \param mutate_vars the mutate arguments
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncDescribeArgs
 */
MXNET_DLL int MXFuncInvokeEx(FunctionHandle fun,
                             NDArrayHandle *use_vars,
                             mx_float *scalar_args,
                             NDArrayHandle *mutate_vars,
                             int num_params,
                             char **param_keys,
                             char **param_vals);
//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------
/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                               AtomicSymbolCreator **out_array);
/*!
 * \brief Get the detailed information about atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param key_var_num_args The keyword argument for specifying variable number of arguments.
 *            When this parameter has non-zero length, the function allows variable number
 *            of positional arguments, and will need the caller to pass it in in
 *            MXSymbolCreateAtomicSymbol,
 *            With key = key_var_num_args, and value = number of positional arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                          const char **name,
                                          const char **description,
                                          mx_uint *num_args,
                                          const char ***arg_names,
                                          const char ***arg_type_infos,
                                          const char ***arg_descriptions,
                                          const char **key_var_num_args);
/*!
 * \brief Create an AtomicSymbol.
 * \param creator the AtomicSymbolCreator
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                         mx_uint num_param,
                                         const char **keys,
                                         const char **vals,
                                         SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateGroup(mx_uint num_symbols,
                                  SymbolHandle *symbols,
                                  SymbolHandle *out);
/*!
 * \brief Load a symbol from a json file.
 * \param fname the file name.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out);
/*!
 * \brief Load a symbol from a json string.
 * \param json the json string.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out);
/*!
 * \brief Save a symbol into a json file.
 * \param symbol the input symbol.
 * \param fname the file name.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname);
/*!
 * \brief Save a symbol into a json string
 * \param symbol the input symbol.
 * \param out_json output json string.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolFree(SymbolHandle symbol);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolPrint(SymbolHandle symbol, const char **out_str);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAttr(SymbolHandle symbol,
                              const char* key,
                              const char** out,
                              int *success);
/*!
 * \brief Set string attribute from symbol.
 *  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
 *
 *  Safe recommendaton: use  immutable graph
 *  - Only allow set attributes during creation of new symbol as optional parameter
 *
 *  Mutable graph (be careful about the semantics):
 *  - Allow set attr at any point.
 *  - Mutating an attribute of some common node of two graphs can cause confusion from user.
 *
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param value The value to be saved.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSetAttr(SymbolHandle symbol,
                              const char* key,
                              const char* value);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListArguments(SymbolHandle symbol,
                                    mx_uint *out_size,
                                    const char ***out_str_array);
/*!
 * \brief List returns in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListOutputs(SymbolHandle symbol,
                                  mx_uint *out_size,
                                  const char ***out_str_array);
/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetInternals(SymbolHandle symbol,
                                   SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetOutput(SymbolHandle symbol,
                                mx_uint index,
                                SymbolHandle *out);
/*!
 * \brief List auxiliary states in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                          mx_uint *out_size,
                                          const char ***out_str_array);
/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param args arguments to sym
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCompose(SymbolHandle sym,
                              const char *name,
                              mx_uint num_args,
                              const char** keys,
                              SymbolHandle* args);
/*!
 * \brief Get the gradient graph of the symbol
 *
 * \param sym the symbol to get gradient
 * \param num_wrt number of arguments to get gradient
 * \param wrt the name of the arguments to get gradient
 * \param out the returned symbol that has gradient
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGrad(SymbolHandle sym,
                           mx_uint num_wrt,
                           const char** wrt,
                           SymbolHandle* out);
/*!
 * \brief infer shape of unknown input shapes given the known one.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param out_shape_data returning array of pointers to head of the input shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShape(SymbolHandle sym,
                                 mx_uint num_args,
                                 const char** keys,
                                 const mx_uint *arg_ind_ptr,
                                 const mx_uint *arg_shape_data,
                                 mx_uint *in_shape_size,
                                 const mx_uint **in_shape_ndim,
                                 const mx_uint ***in_shape_data,
                                 mx_uint *out_shape_size,
                                 const mx_uint **out_shape_ndim,
                                 const mx_uint ***out_shape_data,
                                 mx_uint *aux_shape_size,
                                 const mx_uint **aux_shape_ndim,
                                 const mx_uint ***aux_shape_data,
                                 int *complete);
/*!
 * \brief infer type of unknown input types given the known one.
 *  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
 *  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_type_data the content of the CSR
 * \param in_type_size sizeof the returning array of in_types
 * \param in_type_data returning array of pointers to head of the input type.
 * \param out_type_size sizeof the returning array of out_types
 * \param out_type_data returning array of pointers to head of the input type.
 * \param aux_type_size sizeof the returning array of aux_types
 * \param aux_type_data returning array of pointers to head of the auxiliary type.
 * \param complete whether infer type completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferType(SymbolHandle sym,
                                mx_uint num_args,
                                const char** keys,
                                const int *arg_type_data,
                                mx_uint *in_type_size,
                                const int **in_type_data,
                                mx_uint *out_type_size,
                                const int **out_type_data,
                                mx_uint *aux_type_size,
                                const int **aux_type_data,
                                int *complete);
//--------------------------------------------
// Part 4: Executor interface
//--------------------------------------------
/*!
 * \brief Delete the executor
 * \param handle the executor.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorFree(ExecutorHandle handle);
/*!
 * \brief Print the content of execution plan, used for debug.
 * \param handle the executor.
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorPrint(ExecutorHandle handle, const char **out_str);
/*!
 * \brief Executor forward method
 *
 * \param handle executor handle
 * \param is_train bool value to indicate whether the forward pass is for evaluation
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorForward(ExecutorHandle handle, int is_train);
/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NDArray handle for heads' gradient
 *
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBackward(ExecutorHandle handle,
                                 mx_uint len,
                                 NDArrayHandle *head_grads);

/*!
 * \brief Get executor's head NDArray
 *
 * \param handle executor handle
 * \param out_size output narray vector size
 * \param out out put narray handles
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorOutputs(ExecutorHandle handle,
                                mx_uint *out_size,
                                NDArrayHandle **out);

/*!
 * \brief Generate Executor from symbol
 *
 * \param symbol_handle symbol handle
 * \param dev_type device type
 * \param dev_id device id
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param aux_states_len length of auxiliary states
 * \param aux_states auxiliary states array
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBind(SymbolHandle symbol_handle,
                             int dev_type,
                             int dev_id,
                             mx_uint len,
                             NDArrayHandle *in_args,
                             NDArrayHandle *arg_grad_store,
                             mx_uint *grad_req_type,
                             mx_uint aux_states_len,
                             NDArrayHandle *aux_states,
                             ExecutorHandle *out);
/*!
 * \brief Generate Executor from symbol,
 *  This is advanced function, allow specify group2ctx map.
 *  The user can annotate "ctx_group" attribute to name each group.
 *
 * \param symbol_handle symbol handle
 * \param dev_type device type of default context
 * \param dev_id device id of default context
 * \param num_map_keys size of group2ctx map
 * \param map_keys keys of group2ctx map
 * \param map_dev_types device type of group2ctx map
 * \param map_dev_ids device id of group2ctx map
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param aux_states_len length of auxiliary states
 * \param aux_states auxiliary states array
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBindX(SymbolHandle symbol_handle,
                              int dev_type,
                              int dev_id,
                              mx_uint num_map_keys,
                              const char** map_keys,
                              const int* map_dev_types,
                              const int* map_dev_ids,
                              mx_uint len,
                              NDArrayHandle *in_args,
                              NDArrayHandle *arg_grad_store,
                              mx_uint *grad_req_type,
                              mx_uint aux_states_len,
                              NDArrayHandle *aux_states,
                              ExecutorHandle *out);
/*!
 * \brief set a call back to notify the completion of operation
 */
MXNET_DLL int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                           ExecutorMonitorCallback callback,
                                           void* callback_handle);
//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
/*!
 * \brief List all the available iterator entries
 * \param out_size the size of returned iterators
 * \param out_array the output iteratos entries
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListDataIters(mx_uint *out_size,
                              DataIterCreator **out_array);
/*!
 * \brief Init an iterator, init with parameters
 * the array size of passed in arguments
 * \param handle of the iterator creator
 * \param num_param number of parameter
 * \param keys parameter keys
 * \param vals parameter values
 * \param out resulting iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterCreateIter(DataIterCreator handle,
                                   mx_uint num_param,
                                   const char **keys,
                                   const char **vals,
                                   DataIterHandle *out);
/*!
 * \brief Get the detailed information about data iterator.
 * \param creator the DataIterCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetIterInfo(DataIterCreator creator,
                                    const char **name,
                                    const char **description,
                                    mx_uint *num_args,
                                    const char ***arg_names,
                                    const char ***arg_type_infos,
                                    const char ***arg_descriptions);
/*!
 * \brief Free the handle to the IO module
 * \param handle the handle pointer to the data iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterFree(DataIterHandle handle);
/*!
 * \brief Move iterator to next position
 * \param handle the handle to iterator
 * \param out return value of next
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterNext(DataIterHandle handle,
                             int *out);
/*!
 * \brief Call iterator.Reset
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterBeforeFirst(DataIterHandle handle);

/*!
 * \brief Get the handle to the NDArray of underlying data
 * \param handle the handle pointer to the data iterator
 * \param out handle to underlying data NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetData(DataIterHandle handle,
                                NDArrayHandle *out);
/*!
 * \brief Get the image index by array.
 * \param handle the handle pointer to the data iterator
 * \param out_index output index of the array.
 * \param out_size output size of the array.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetIndex(DataIterHandle handle,
                                 uint64_t **out_index,
                                 uint64_t *out_size);
/*!
 * \brief Get the padding number in current data batch
 * \param handle the handle pointer to the data iterator
 * \param pad pad number ptr
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetPadNum(DataIterHandle handle,
                                  int *pad);

/*!
 * \brief Get the handle to the NDArray of underlying label
 * \param handle the handle pointer to the data iterator
 * \param out the handle to underlying label NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetLabel(DataIterHandle handle,
                                 NDArrayHandle *out);
//--------------------------------------------
// Part 5: basic KVStore interface
//--------------------------------------------
/*!
 * \brief Create a kvstore
 * \param type the type of KVStore
 * \param out The output type of KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreCreate(const char *type,
                              KVStoreHandle *out);
/*!
 * \brief Delete a KVStore handle.
 * \param handle handle to the kvstore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreFree(KVStoreHandle handle);
/*!
 * \brief Init a list of (key,value) pairs in kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreInit(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals);

/*!
 * \brief Push a list of (key,value) pairs to kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePush(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePull(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief user-defined updater for the kvstore
 * It's this updater's responsibility to delete \a recv and \a local
 * \param the key
 * \param recv the pushed value on this key
 * \param local the value stored on local on this key
 * \param handle The additional handle to the updater
 */
typedef void (MXKVStoreUpdater)(int key,
                                NDArrayHandle recv,
                                NDArrayHandle local,
                                void *handle);
/*!
 * \brief register an push updater
 * \param handle handle to the KVStore
 * \param updater udpater function
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetUpdater(KVStoreHandle handle,
                                  MXKVStoreUpdater updater,
                                  void *updater_handle);
/*!
 * \brief get the type of the kvstore
 * \param handle handle to the KVStore
 * \param type a string type
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetType(KVStoreHandle handle,
                               const char** type);
//--------------------------------------------
// Part 6: advanced KVStore for multi-machines
//--------------------------------------------

/**
 * \brief return The rank of this node in its group, which is in [0, GroupSize).
 *
 * \param handle handle to the KVStore
 * \param ret the node rank
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetRank(KVStoreHandle handle,
                               int *ret);

/**
 * \brief return The number of nodes in this group, which is
 * - number of workers if if `IsWorkerNode() == true`,
 * - number of servers if if `IsServerNode() == true`,
 * - 1 if `IsSchedulerNode() == true`,
 * \param handle handle to the KVStore
 * \param ret the group size
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetGroupSize(KVStoreHandle handle,
                                    int *ret);

/**
 * \brief return whether or not this process is a worker node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsWorkerNode(int *ret);


/**
 * \brief return whether or not this process is a server node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsServerNode(int *ret);


/**
 * \brief return whether or not this process is a scheduler node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsSchedulerNode(int *ret);

/**
 * \brief global barrier among all worker machines
 *
 * \param handle handle to the KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreBarrier(KVStoreHandle handle);

/**
 * \brief the prototype of a server controller
 * \param head the head of the command
 * \param body the body of the command
 */
typedef void (MXKVStoreServerController)(int head,
                                         const char* body);

/**
 * \return Run as server (or scheduler)
 *
 * \param handle handle to the KVStore
 * \param controller the user-defined server controller
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreRunServer(KVStoreHandle handle,
                                 MXKVStoreServerController controller);

/**
 * \return Send a command to all server nodes
 *
 * \param handle handle to the KVStore
 * \param cmd_id the head of the command
 * \param cmd_body the body of the command
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
                                             int cmd_id,
                                             const char* cmd_body);

/**
 * \brief Create a RecordIO writer object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO writer object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf buffer to write
 * \param size size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterWriteRecord(RecordIOHandle *handle,
                                          const char *buf, size_t size);

/**
 * \brief Create a RecordIO reader object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO reader object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderFree(RecordIOHandle *handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf pointer to return buffer
 * \param size point to size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderReadRecord(RecordIOHandle *handle,
                                        char const **buf, size_t *size);

/**
 * \brief Create a MXRtc object
*/
MXNET_DLL int MXRtcCreate(char* name, mx_uint num_input, mx_uint num_output,
                          char** input_names, char** output_names,
                          NDArrayHandle* inputs, NDArrayHandle* outputs,
                          char* kernel, RtcHandle *out);

/**
 * \brief Run cuda kernel
*/
MXNET_DLL int MXRtcPush(RtcHandle handle, mx_uint num_input, mx_uint num_output,
                        NDArrayHandle* inputs, NDArrayHandle* outputs,
                        mx_uint gridDimX,
                        mx_uint gridDimY,
                        mx_uint gridDimZ,
                        mx_uint blockDimX,
                        mx_uint blockDimY,
                        mx_uint blockDimZ);

/**
 * \brief Delete a MXRtc object
*/
MXNET_DLL int MXRtcFree(RtcHandle handle);

MXNET_DLL int MXOptimizerFindCreator(const char *key,
                                     OptimizerCreator *out);

MXNET_DLL int MXOptimizerCreateOptimizer(OptimizerCreator creator,
                                         mx_uint num_param,
                                         const char **keys,
                                         const char **vals,
                                         OptimizerHandle *out);

MXNET_DLL int MXOptimizerFree(OptimizerHandle handle);

MXNET_DLL int MXOptimizerUpdate(OptimizerHandle handle,
                                int index,
                                NDArrayHandle weight,
                                NDArrayHandle grad,
                                mx_float lr);

#endif  // MXNET_C_API_H_
//===== EXPANDED: ../include/mxnet/c_api.h =====


// check c++11
#if DMLC_USE_CXX11 == 0
#error "CXX11 was required for symbolic module"
#endif

namespace mxnet {
/*!
 * \brief Internal data structure used for
 *  graph serializaion and graph algorithms.
 */
class StaticGraph;
/*!
 * \brief Symbol is used to represent dynamically generated symbolic computation graph.
 *
 *   This class is used as a tool to generate computation graphs(aka. configuration) of the network.
 *   Symbol is always composite, the head Node is the output node of the symbol.
 *   An atomic symbol can be seen as a special case of the composite symbol with only the head node.
 */
class Symbol {
 public:
  /*!
   * \brief copy the symbol
   * \return a deep copy of the graph
   */
  Symbol Copy() const;
  /*!
   * \brief print the symbol info to output stream.
   * \param os the output stream we like to print to
   */
  void Print(std::ostream &os) const; // NOLINT(*)
  /*!
   * \brief List the arguments names.
   *
   * The position of the returned list also corresponds to calling position in operator()
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  std::vector<std::string> ListArguments() const;
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListOutputs() const;
  /*! \return get the descriptions of auxiliary data for this symbol */
  std::vector<std::string> ListAuxiliaryStates() const;
  /*!
   * \brief get the index th element from the returned tuple.
   * \param index index of multi output
   * \return the symbol corresponds to the indexed element.
   */
  Symbol operator[] (size_t index) const;
  /*!
   * \brief Compose the symbol with arguments, this changes current symbol.
   *
   * The positional arguments passed in must be complete(contain all arguments).
   *
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::vector<Symbol>& args,
               const std::string& name);
  /*!
   * \brief Compose the symbol with arguments, this changes the current symbol.
   * The kwargs passed in can be in-complete,
   *
   * The rest of the symbols will remain the same name.
   *
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::unordered_map<std::string, Symbol>& kwargs,
               const std::string& name);
  /*!
   * \brief set additional attributes of the symbol,
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param key the key of the attribute
   * \param value the value of the attribute.
   */
  void SetAttr(const std::string &key, const std::string& value);
  /*!
   * \brief Get attributes from the symbol.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param key Key of the attribute.
   * \param out the output value of the attribute.
   * \return true if the attribute exists, false if the attribute do not exist.
   */
  bool GetAttr(const std::string& key, std::string* out);
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  Symbol operator () (const std::vector<Symbol>& args, const std::string& name) const;
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  Symbol operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                      const std::string& name) const;
  /*
   * \brief Get all the internal nodes of the symbol.
   * \return symbol A new symbol whose output contains all the outputs of the symbols
   *  Including input variables and intermediate outputs.
   */
  Symbol GetInternals() const;
  /*!
   * \brief get the gradient graph
   * \param wrt with respect to the input
   * \return the new symbol with gradient graph
   */
  Symbol Grad(const std::vector<std::string>& wrt) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param arg_shapes the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shapes Use to store the infered shapes of outputs.
   * \param aux_shapes Use to store the infered shapes of auxiliary states
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  bool InferShape(std::vector<TShape> *arg_shapes,
                  std::vector<TShape> *out_shapes,
                  std::vector<TShape> *aux_shapes) const;
  /*!
   * \brief infer the shapes by providing shapes of known arguments.
   * \param known_arg_shapes map of argument name to shape of arguments with known shapes.
   * \param arg_shapes used to store infered shapes of arguments.
   * \param out_shapes used to store infered shapes of outputs.
   * \param aux_shapes Use to store the infered shapes of auxiliary states
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  bool InferShape(const std::unordered_map<std::string, TShape> &known_arg_shapes,
                  std::vector<TShape> *arg_shapes,
                  std::vector<TShape> *out_shapes,
                  std::vector<TShape> *aux_shapes) const;

  /*!
   * \brief infer the types of outputs and unknown input arguments
   * \param arg_types the type of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_type allows unknown elements, which are checked by type.ndim() == 0.
   *     For unknown types, Infertype will try to fill in the correct type in in_type
   *     For known types, Infertype will check type consistency
   *
   *     common practice: set the type of data input, and usually weight's type can be infered
   *
   * \param out_types Use to store the infered types of outputs.
   * \param aux_types Use to store the infered types of auxiliary states
   * \return true if the type inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_types are inconsistent.
   */
  bool InferType(std::vector<int> *arg_types,
                  std::vector<int> *out_types,
                  std::vector<int> *aux_types) const;
  /*!
   * \brief infer the types by providing types of known arguments.
   * \param known_arg_types map of argument name to type of arguments with known types.
   * \param arg_types used to store infered types of arguments.
   * \param out_types used to store infered types of outputs.
   * \param aux_types Use to store the infered types of auxiliary states
   * \return true if the type inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_types are inconsistent.
   */
  bool InferType(const std::unordered_map<std::string, int> &known_arg_types,
                  std::vector<int> *arg_types,
                  std::vector<int> *out_types,
                  std::vector<int> *aux_types) const;
  /*!
   * \brief interface for json serialization.
   * \param writer the JSON writer write json.
   */
  void Save(dmlc::JSONWriter *writer) const;
  /*!
   * \brief interface for json serialization.
   * \param reader the JSON read to read json.
   */
  void Load(dmlc::JSONReader *reader);
  /*!
   * \brief get number of outputs of this symbol
   * \return number of outputs
   */
  inline size_t NumOutputs() const {
    return heads_.size();
  }
  /*!
   * \brief create Symbol by wrapping OperatorProperty
   * This function takes the ownership of op
   *
   * \param op the OperatorProperty of the Operator
   * \return Symbol
   * \sa OperatorProperty::Create
   */
  static Symbol Create(OperatorProperty *op);
  /*!
   * \brief create equivalence of symbol by grouping the symbols together
   * \param symbols list of symbols
   * \return the grouped symbol
   */
  static Symbol CreateGroup(const std::vector<Symbol> &symbols);
  /*!
   * \brief create variable symbol node
   * \param name name of the variable
   * \return the new variable
   */
  static Symbol CreateVariable(const std::string &name);

 protected:
  // Decalre node, internal data structure.
  struct Node;
  /*! \brief an entry that represents output data from a node */
  struct DataEntry {
    /*! \brief the source node of this data */
    std::shared_ptr<Node> source;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief enabled default copy constructor */
    DataEntry() {}
    /*! \brief constructor from index */
    DataEntry(std::shared_ptr<Node> source, uint32_t index)
        : source(source), index(index) {}
  };
  /*!
   * \brief the head nodes of Symbols
   * This head is only effective when
   */
  std::vector<DataEntry> heads_;

 private:
  /*! \return whwther the symbol is atomic */
  inline bool is_atomic() const;
  /*!
   * \brief Visit all the nodes in left-to-right depth first order.
   *
   *  This function will visit the graph in DFS order, call fvisit exactly once
   *  for each Node, and store the result in out_result.
   *
   * \param fvisit function applied for each visit.
   * \tparam FVisit visiting function type
   */
  template<typename FVisit>
  inline void DFSVisit(FVisit fvisit) const;
  /*!
   * \brief Find duplicate arguments in the composition
   * \param out the map of argument-name -> occurence count
   * \return maximum number of duplication factor
   */
  int FindDuplicateArgs(std::unordered_map<std::string, int> *out) const;
  /*!
   * \brief Convert symbol into internal static graph
   *
   * \param out_graph the pointer holder of the output graph
   */
  void ToStaticGraph(StaticGraph *out_graph) const;
  /*!
   * \brief create equivalence of symbol from static graphs.
   *  This operation will change the content of current symbol.
   * \param graph the static graph
   */
  void FromStaticGraph(const StaticGraph &graph);
  /*! \brief let static graph know the contents */
  friend class StaticGraph;
};

/*!
 * \brief Executor of a computation graph.
 *  Executor can be created by Binding a symbol.
 */
class Executor {
 public:
  /*! \brief destructor */
  virtual ~Executor() {}
  /*!
   * \brief Perform a Forward operation of Operator
   *  After this operation, user can get the result by using function head.
   */
  virtual void Forward(bool is_train) = 0;
  /*!
   * \brief Perform a Partial Forward operation of Operator.
   *  Only issue operation specified by step.
   *  The caller must keep calling PartialForward with increasing steps, until step_left=0.
   * \param is_train Whether this is training phase.
   * \param step current step, user can always start from 0
   * \param step_left Number of steps left to finish the forward.
   */
  virtual void PartialForward(bool is_train, int step, int *step_left) = 0;
  /*!
   * \brief Perform a Backward operation of the Operator.
   *  This must be called after Forward.
   *  After this operation, NDArrays specified by grad_in_args_store will be updated accordingly.
   *  User is allowed to pass in an empty Array if the head node is
   *  loss function and head gradeitn is not needed.
   *
   * \param head_grads the gradient of head nodes to be backproped.
   */
  virtual void Backward(const std::vector<NDArray> &head_grads) = 0;
  /*!
   * \brief print the execution plan info to output stream.
   * \param os the output stream we like to print to.
   */
  virtual void Print(std::ostream &os) const {} // NOLINT(*)
  /*!
   * \brief get array of outputs in the executor.
   * \return array of outputs in the executor.
   */
  virtual const std::vector<NDArray> &outputs() const = 0;
  /*!
   * \brief Create an operator by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *
   * \param default_ctx the default context of binding.
   * \param group2ctx Context mapping group to context.
   * \param symbol the symbol that specifies the output of Forward pass.
   * \param in_args the NDArray that stores the input arguments to the symbol.
   * \param arg_grad_store NDArray that is used to store the gradient output of the input arguments.
   * \param grad_req_type requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   * \param aux_states NDArray that is used as internal state in op
   * \return a new executor.
   */
  static Executor *Bind(Symbol symbol,
                        const Context& default_ctx,
                        const std::map<std::string, Context>& group2ctx,
                        const std::vector<NDArray> &in_args,
                        const std::vector<NDArray> &arg_grad_store,
                        const std::vector<OpReqType> &grad_req_type,
                        const std::vector<NDArray> &aux_states);
  /*!
   * \brief the prototype of user-defined monitor callback
   */
  typedef std::function<void(const char*, void*)> MonitorCallback;
  /*!
   * \brief Install a callback to notify the completion of operation.
   */
  virtual void SetMonitorCallback(const MonitorCallback& callback) {}
};  // class operator
}  // namespace mxnet
#endif  // MXNET_SYMBOLIC_H_
//===== EXPANDED: ../include/mxnet/symbolic.h =====

//===== EXPANDIND: ../src/symbol/graph_executor.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.h
 * \brief Executor to execute the Forward and Backward on Composition Graph.
*/
#ifndef MXNET_SYMBOL_GRAPH_EXECUTOR_H_
#define MXNET_SYMBOL_GRAPH_EXECUTOR_H_

//===== EXPANDIND: ../src/symbol/static_graph.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file static_graph.h
 * \brief A memory compact representation of symbolic graph
 *   Used for serialization, and helper data structure.
 * \author Naiyan Wang
 */
#ifndef MXNET_SYMBOL_STATIC_GRAPH_H_
#define MXNET_SYMBOL_STATIC_GRAPH_H_


namespace mxnet {
/*!
 * \brief StaticGraph is the configuration of computation graphs.
 *  This is the "configuration file" of mxnet.
 *  It can be converted to/from Symbol, and can be used to bind to operators.
 *   The symbol can be converted from/to StaticGraph, the actual configuration used by mxnet.
 *   Symbol offers more flexible way to composite nodes than StaticGraph, which makes it good
 *   tool to generate configurations from language bindings such as python.
 * \sa Symbol
 */
class StaticGraph {
 public:
  /*! \brief represents a data in the graph */
  struct DataEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t source_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief default constructor */
    DataEntry() {}
    /*!
     * \brief constructor with source and index
     * \param source_id source id
     * \param index node index
     */
    DataEntry(uint32_t source_id, uint32_t index)
        : source_id(source_id), index(index) {}
    /*!
     * \brief compare equality
     * \param other the other entry to compare
     * \return whether two entries equals to each other
     */
    inline bool operator==(const DataEntry &other) const {
      return source_id == other.source_id && index == other.index;
    }
    /*!
     * \brief comparator, allows to use map
     * \param other the other entry to compare
     * \return whether two entries is smaller than the other
     */
    inline bool operator<(const DataEntry &other) const {
      if (source_id == other.source_id) return index < other.index;
      return source_id < other.source_id;
    }
    /*!
     * \brief interface for json serialization.
     * \param writer the JSON writer to write json into.
     */
    inline void Save(dmlc::JSONWriter *writer) const {
      writer->BeginArray(false);
      writer->WriteArrayItem(source_id);
      writer->WriteArrayItem(index);
      writer->EndArray();
    }
    /*!
     * \brief interface for json serialization.
     * \param reader the JSON reader to read json from.
     */
    inline void Load(dmlc::JSONReader *reader) {
      std::pair<uint32_t, uint32_t> p;
      reader->Read(&p);
      *this = DataEntry(p.first, p.second);
    }
  };
  /*!
   * \brief Operation Node in static graphs.
   *  There are two types of node, Forward and Backward Node.
   *
   *  - Forward node corresponds to the op.Forward
   *  - Backward node corresponds to the Backward pass,
   *    where the corresponding forward node is indicated by backward_source_id.
   *    The op field in Backward node is nullptr
   *
   *  The reason we explicit support Backward node is to allow special treatment
   *  such as shape inference and state sharing with Forward pass.
   */
  struct Node {
    /*! \brief wrapped operator property */
    std::unique_ptr<OperatorProperty> op;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs (node_id, index) for of the nodes*/
    std::vector<DataEntry> inputs;
    /*!
     * \brief If this field is nonnegative, this indicates this
     *  Node is corresponds to a Backward Operation of Operator.
     *  backward_source_id will points to the corresponding Forward Node.
     *
     *  For normal node, this field is -1.
     *  When the node is a Backward node, the op field will be nullptr
     */
    int32_t backward_source_id;
    /*! \brief additional attributes about the node */
    std::map<std::string, std::string> attr;
    /*! \brief default constructor */
    Node() : backward_source_id(-1) {}
    /*! \brief copy constructor in favor of serialization. */
    Node(const Node& another)
        : op(another.op.get() ? another.op.get()->Copy() : nullptr),
          name(another.name),
          inputs(another.inputs),
          backward_source_id(another.backward_source_id),
          attr(another.attr) {}

    inline Node& operator=(Node another) {
      op = std::move(another.op);
      name = std::move(another.name);
      inputs = std::move(another.inputs);
      backward_source_id = std::move(another.backward_source_id);
      attr = std::move(another.attr);
      return *this;
    }
    /*! \return whether the node is forward op node */
    inline bool is_forward() const {
      return op != nullptr;
    }
    /*! \return whether the node is backward op node */
    inline bool is_backward() const {
      return backward_source_id != -1;
    }
    /*! \return whether the node is variable node */
    inline bool is_variable() const {
      return op == nullptr && !is_backward();
    }
    /*!
     * \brief interface for json serialization.
     * \param writer the JSON writer write json.
     */
    void Save(dmlc::JSONWriter *writer) const;
    /*!
     * \brief interface for json serialization.
     * \param reader the JSON read to read json.
     */
    void Load(dmlc::JSONReader *reader);
  };
  /*! \brief all nodes in the graph */
  std::vector<Node> nodes;
  /*! \brief index of nodes that correspods to arguments */
  std::vector<uint32_t> arg_nodes;
  /*! \brief heads outputs of the graph */
  std::vector<DataEntry> heads;
  /*!
   * \brief interface for json serialization.
   * \param writer the JSON writer write json.
   */
  void Save(dmlc::JSONWriter *writer) const;
  /*!
   * \brief interface for json serialization.
   * \param reader the JSON read to read json.
   */
  void Load(dmlc::JSONReader *reader);
  // funtions to help inference in static graph
  /*!
   * \brief Perform a topological sort on the graph
   * \return a topological order of node indices.
   */
  std::vector<uint32_t> TopoSort() const;
  /*!
   * \brief Get a post DFS order traversal order from the head nodes.
   *  Post DFS order is a special case of Topological order.
   * \param heads The head of the node.
   * \param banned The banned map, used to ban some nodes from the graph.
   * \return a post DFS visit order of nodes that can reach heads.
   */
  std::vector<uint32_t> PostDFSOrder(const std::vector<uint32_t>& head_nodes,
                                     const std::unordered_set<uint32_t>& banned
                                     = std::unordered_set<uint32_t>()) const;
  /*!
   * \brief infer the node shapes in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_shapes The shapes of the each outputs of nodes in the graph.
   * \param node_aux_shapes The shapes of the each auxiliary states of nodes in the graph.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeShapes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<TShape> > *node_out_shapes,
                       std::vector<std::vector<TShape> > *node_aux_shapes) const;
  /*!
   * \brief infer the node types in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_types The types of the each outputs of nodes in the graph.
   * \param node_aux_types The types of the each auxiliary states of nodes in the graph.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeTypes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<int> > *node_out_types,
                       std::vector<std::vector<int> > *node_aux_types) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param aux_shape the shape of auxiliary states of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferShape(std::vector<TShape>* in_shape,
                  std::vector<TShape>* out_shape,
                  std::vector<TShape>* aux_shape) const;

  /*!
   * \brief infer the types of outputs and unknown input arguments
   * \param in_type the type of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_type allows unknown elements, which are checked by type.ndim() == 0.
   *     For unknown types, Infertype will try to fill in the correct type in in_type
   *     For known types, Infertype will check type consistency
   *
   *     common practice: set the type of data input, and usually weight's type can be infered
   *
   * \param out_type the type of outputs of the operator
   *     Infertype will modify the vector to fill output int
   * \param aux_type the type of auxiliary states of the operator
   *     Infertype will modify the vector to fill output int
   * \return if the type inference is successful, return true, else return false.
   */
  bool InferType(std::vector<int>* in_type,
                  std::vector<int>* out_type,
                  std::vector<int>* aux_type) const;
  /*!
   * \brief Add a full backward pass in the static graph.
   *  This function will add gradient nodes for each heads,
   *  and add the backward pass to backprop the gradients all
   *  the way to the arguments.
   *
   *  This will change the nodes field in the StaticGraph, but will not change other fields.
   *  The head and input of Backward pass will be returned by head_grad_nodes and arg_grads.
   *
   * \param head_grad_nodes used to store the created head gradient inputs for backward pass.
   * \param arg_grads used to store gradients to args, can be multiple one if an argument is used by operator
   * \param out_mirror_map The mirror map of the backward plan.
   */
  void MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                        std::vector<DataEntry> *arg_grads,
                        std::map<uint32_t, uint32_t>* out_mirror_map);
  /*!
   * \brief Convert symbol into static graph.
   * \param symbol the symbol to convert from.
   */
  inline void FromSymbol(const Symbol &symbol) {
    symbol.ToStaticGraph(this);
  }
  /*!
   * \brief create a sum node that aggregates gradient together
   * \param grad_source the source of the inputs.
   * \return a created ElementWiseSum node
   */
  static Node CreateSumNode(const std::vector<DataEntry> &grad_source);
  /*!
   * \brief create a copy node.
   * \param source the Source data
   * \return a created _CrossDeviceCopy node
   */
  static Node CreateCopyNode(const DataEntry& source);
};
}  // namespace mxnet

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, ::mxnet::StaticGraph::DataEntry, true);
}
#endif  //  MXNET_SYMBOL_STATIC_GRAPH_H_
//===== EXPANDED: ../src/symbol/static_graph.h =====

//===== EXPANDIND: ../src/symbol/graph_memory_allocator.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_memory_allocator.h
 * \brief Memory allocator for graph executor.
*/
#ifndef MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
#define MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_

//===== EXPANDIND: ../src/symbol/graph_algorithm.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_allocation_helper.h
 * \brief This header contains graph algorithms on StaticGraph.
 *  It is used  compute informations such as whether two
 *  operations can run in parallel, and helps allocation.
*/
#ifndef MXNET_SYMBOL_GRAPH_ALGORITHM_H_
#define MXNET_SYMBOL_GRAPH_ALGORITHM_H_


namespace mxnet {
namespace graph {
/*!
 * \brief Find best path in the DAG, with reward defined
 *  by sum of reward of each node along the path.
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_reward the reward of each node.
 * \param path the output path of nodes.
 * \return the total reward of best path.
 */
inline uint32_t FindBestPath(
    const StaticGraph &graph,
    const std::vector<uint32_t> &topo_order,
    const std::vector<uint32_t> &node_reward,
    std::vector<uint32_t> *path) {
  const uint32_t num_nodes = static_cast<uint32_t>(graph.nodes.size());
  CHECK_EQ(graph.nodes.size(), node_reward.size());
  CHECK_EQ(graph.nodes.size(), topo_order.size());

  std::vector<uint32_t> best_reward(node_reward.size(), 0);
  std::vector<uint32_t> next_node(node_reward.size(), num_nodes);
  uint32_t best_solution = 0, best_start_node = 0;

  // traverse in reverse topo order
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    const uint32_t nid = *it;
    best_reward[nid] += node_reward[nid];
    if (best_reward[nid] > best_solution) {
      best_solution = best_reward[nid];
      best_start_node = nid;
    }
    for (const StaticGraph::DataEntry& e : graph.nodes[nid].inputs) {
      const uint32_t prev = e.source_id;
      if (best_reward[nid] > best_reward[prev]) {
        best_reward[prev] = best_reward[nid];
        next_node[prev] = nid;
      }
    }
  }
  path->clear();
  uint32_t reward = 0;
  for (uint32_t nid = best_start_node; nid < num_nodes; nid = next_node[nid]) {
    path->push_back(nid); reward += node_reward[nid];
  }
  CHECK_EQ(reward, best_solution);
  return best_solution;
}

/*!
 * \brief Color the nodes in the graph into index.
 *  The coloring algorithm tries to assign node group
 *  such that node in the same group cannot run in parallel.
 *
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_importance The importance of the node
 * \param max_ncolor maximum number of colors allowed.
 * \param color the color index of each of the node.
 * \return the total number of colors.
 */
inline uint32_t ColorNodeGroup(
    const StaticGraph &graph,
    const std::vector<uint32_t> &topo_order,
    std::vector<uint32_t> node_importance,
    uint32_t max_ncolor,
    std::vector<uint32_t> *color) {
  CHECK_NE(max_ncolor, 0);
  CHECK_EQ(graph.nodes.size(), topo_order.size());
  CHECK_EQ(graph.nodes.size(), node_importance.size());

  color->clear();
  color->resize(topo_order.size(), max_ncolor);
  uint32_t cindex;
  // greedy algorithm, every time
  // find a path with best reward and assign a new color
  // All the nodes in the path cannot run in parallel.
  for (cindex = 0; cindex < max_ncolor - 1; ++cindex) {
    std::vector<uint32_t> path;
    uint32_t reward = FindBestPath(graph, topo_order, node_importance, &path);
    if (reward == 0) break;
    for (uint32_t nid : path) {
      if (node_importance[nid] != 0) {
        CHECK_EQ(color->at(nid), max_ncolor);
        color->at(nid) = cindex;
        // make the importance 0 after color is decided.
        node_importance[nid] = 0;
      }
    }
  }
  // assign i for rest of the node
  for (size_t i = 0; i < topo_order.size(); ++i) {
    if (color->at(i) == max_ncolor) {
      color->at(i) = cindex;
    }
  }
  return cindex + 1;
}
}  // namespace graph
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_ALGORITHM_H_
//===== EXPANDED: ../src/symbol/graph_algorithm.h =====

//===== EXPANDIND: ../src/common/utils.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_COMMON_UTILS_H_
#define MXNET_COMMON_UTILS_H_

#if DMLC_USE_CXX11
#endif  // DMLC_USE_CXX11


namespace mxnet {
namespace common {

#if DMLC_USE_CXX11

// heuristic to dermine number of threads per GPU
inline int GetNumThreadPerGPU() {
  // This is resource efficient option.
  return dmlc::GetEnv("MXNET_GPU_WORKER_NTHREADS", 2);
}

// heuristic to get number of matching colors.
// this decides how much parallelism we can get in each GPU.
inline int GetExecNumMatchColor() {
  // This is resource efficient option.
  int num_match_color = dmlc::GetEnv("MXNET_EXEC_NUM_TEMP", 1);
  return std::min(num_match_color, GetNumThreadPerGPU());
}

/*!
 * \brief Random Engine
 */
typedef std::mt19937 RANDOM_ENGINE;

/*!
 * \brief Helper functions.
 */
namespace helper {

/*!
 * \brief Helper for non-array type `T`.
 */
template <class T>
struct UniqueIf {
  /*!
   * \brief Type of `T`.
   */
  using SingleObject = std::unique_ptr<T>;
};

/*!
 * \brief Helper for an array of unknown bound `T`.
 */
template <class T>
struct UniqueIf<T[]> {
  /*!
   * \brief Type of `T`.
   */
  using UnknownBound = std::unique_ptr<T[]>;
};

/*!
 * \brief Helper for an array of known bound `T`.
 */
template <class T, size_t kSize>
struct UniqueIf<T[kSize]> {
  /*!
   * \brief Type of `T`.
   */
  using KnownBound = void;
};

}  // namespace helper

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs a non-array type `T`. The arguments `args` are passed to the
 * constructor of `T`. The function does not participate in the overload
 * resolution if `T` is an array type.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::SingleObject MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param n The size of the array to construct.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs an array of unknown bound `T`. The function does not participate
 * in the overload resolution unless `T` is an array of unknown bound.
 */
template <class T>
typename helper::UniqueIf<T>::UnknownBound MakeUnique(size_t n) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[n]{});
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 *
 * Constructs an arrays of known bound is disallowed.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::KnownBound MakeUnique(Args&&... args) = delete;

#endif  // DMLC_USE_CXX11

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_UTILS_H_
//===== EXPANDED: ../src/common/utils.h =====


namespace mxnet {
/*!
 * \brief Memory allocators for the GraphExecutor.
 *  This class is intended to be used by GraphExecutor
 *  to allocate the memory for each DataEntryInfo.
 *
 *  The class algorithm works in two phase:
 *  (1) Planning Phase: GraphExecutor call Request and Release
 *      to request and release resources according to dependency.
 *      - Each call to Request will get a ResourceID that is used to
 *        identify the memory block assigned to each DataEntryInfo.
 *  (2) Allocating phase: GraphExecutor call InitMemory.
 *      - Then each DataEntry will call Get to get the real NDArray.
 *  (3) All the memory will be freed up when reference to all the related NDArray ends.
 */
class GraphStorageAllocator {
 public:
  /*! \brief resource index */
  typedef int64_t StorageID;
  /*! \brief bad storage id */
  static const StorageID kBadStorageID = -1;
  /*! \brief constructor to the graph memory allocator */
  explicit GraphStorageAllocator(
      StaticGraph *graph,
      const std::vector<uint32_t>& topo_order) noexcept(false);
  /*!
   * \brief Request a memory.
   * \param ctx the context of the graph
   * \param shape shape of the NDArray we want
   * \param node_id the node that is requesting the memory, used as hint.
   */
  StorageID Request(Context ctx, int type_flag, TShape shape, uint32_t node_id);
  /*!
   * \brief Release a memory.
   * \param id the storage ID of the memory.
   * \param node_id the node id in the graph that is releasing the memory.
   */
  void Release(StorageID id, uint32_t node_id);
  /*!
   * \brief Initialize all the memories requested
   * \return size of memory allocated.
   */
  size_t InitStorages();
  /*!
   * \brief Get the the memory allocated in planning phase.
   * \param id the storage id allocated in planning phase.
   * \param shape the shape of the NDArray requested.
   */
  NDArray Get(StorageID id, TShape shape);

 protected:
  /*! \brief internal storage entry */
  struct StorageEntry {
    /*! \brief id of the storage */
    StorageID id;
    /*! \brief the context of the storage */
    Context ctx;
    /*! \brief the data type enum of the storage */
    int type_flag;
    /*! \brief maximum size of the storage that is requested */
    size_t max_size;
    /*! \brief node index that released it last time */
    uint32_t released_by_node;
    /*! \brief the actual NDArray to hold the data */
    NDArray data;
    /*! \brief constructor */
    StorageEntry() : max_size(0), released_by_node(0) {}
  };
  /*!
   * \brief Allocate a StorageID when Request cannot found existing ones.
   * \param ctx the context of the graph
   * \param shape shape of the NDArray we want
   */
  StorageID Alloc(Context ctx, int type_flag, size_t size);
  /*!
   * \brief Initialize the colors of graph nodes.
   * \param topo_order the topological order in the graph.
   */
  void InitColor(const std::vector<uint32_t> &topo_order);
  /*! \brief reference to the computation graph */
  StaticGraph *graph_;
  /*! \brief all the resources available */
  std::vector<std::unique_ptr<StorageEntry> > data_;
  /*! \brief scale used for rough match */
  size_t match_range_;
  /*!
   * \brief free list of storage entries, maps size to free list
   */
  std::multimap<size_t, StorageEntry*> free_;
  /*!
   * \brief color of nodes in the graph, used for auxiliary policy making.
  */
  std::vector<uint32_t> node_color_;
  /*! \brief whether use color based match algorithm */
  uint32_t num_match_color_;
};

// put implementation in header files for now
GraphStorageAllocator::GraphStorageAllocator(
    StaticGraph *graph,
    const std::vector<uint32_t>& topo_order) noexcept(false)
    : graph_(graph) , num_match_color_(0) {
  match_range_ = dmlc::GetEnv("MXNET_EXEC_MATCH_RANGE", 16);
  // if we set this to 1, this means no color based match.
  // color based match will cost a bit more memory usually
  // but also enables more parallelization.
  num_match_color_ = static_cast<uint32_t>(common::GetExecNumMatchColor());
  this->InitColor(topo_order);
}

void GraphStorageAllocator::InitColor(const std::vector<uint32_t>& topo_order) {
  std::vector<uint32_t> importance(graph_->nodes.size(), 0);
  for (size_t i = 0; i < topo_order.size(); ++i) {
    uint32_t nid = topo_order[i];
    if (graph_->nodes[nid].is_variable()) continue;
    importance[nid] = 1;
  }
  num_match_color_ = graph::ColorNodeGroup(
      *graph_, topo_order,
      importance, num_match_color_,
      &node_color_);
}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Alloc(Context ctx, int type_flag, size_t size) {
  StorageID id = static_cast<StorageID>(data_.size());
  std::unique_ptr<StorageEntry> ptr(new StorageEntry());
  ptr->id = id;
  ptr->ctx = ctx;
  ptr->type_flag = type_flag;
  ptr->max_size = size;
  data_.push_back(std::move(ptr));
  return id;
}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Request(Context ctx, int type_flag, TShape shape, uint32_t node_id) {
  // search memory block in [size / match_range_, size * match_range_)
  size_t size = shape.Size();
  if (match_range_ == 0) return this->Alloc(ctx, type_flag, size);
  auto begin = free_.lower_bound(size / match_range_);
  auto mid = free_.lower_bound(size);
  auto end = free_.upper_bound(size * match_range_);
  // TODO(bing, min) consider better strategy
  // search for memory blocks larger than requested
  for (auto it = mid; it != end; ++it) {
    StorageEntry *e = it->second;
    if (e->ctx != ctx) continue;
    if (e->type_flag != type_flag) continue;
    if (node_color_[e->released_by_node] != node_color_[node_id]) continue;
    // Use exect matching strategy
    e->max_size = std::max(size, e->max_size);
    // find a exact match, erase from map and return
    free_.erase(it);
    return e->id;
  }
  // then search for memory blocks smaller than requested space
  for (auto it = mid; it != begin;) {
    --it;
    StorageEntry *e = it->second;
    if (e->ctx != ctx) continue;
    if (e->type_flag != type_flag) continue;
    if (node_color_[e->released_by_node] != node_color_[node_id]) continue;
    // Use exect matching strategy
    e->max_size = std::max(size, e->max_size);
    // find a exact match, erase from map and return
    free_.erase(it);
    return e->id;
  }
  // cannot find anything return a new one.
  return this->Alloc(ctx, type_flag, size);
}

void GraphStorageAllocator::Release(StorageID id, uint32_t node_id) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  e->released_by_node = node_id;
  free_.insert({e->max_size, e});
}

size_t GraphStorageAllocator::InitStorages() {
  size_t total = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    StorageEntry *e = data_[i].get();
    TShape shape = mshadow::Shape1(e->max_size);
    e->data = NDArray(shape, e->ctx, false, e->type_flag);
    total += e->max_size * mshadow::mshadow_sizeof(e->type_flag);
  }
  return total;
}

NDArray GraphStorageAllocator::Get(StorageID id, TShape shape) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  return e->data.Slice(0, shape.Size()).Reshape(shape);
}
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
//===== EXPANDED: ../src/symbol/graph_memory_allocator.h =====


namespace mxnet {
/*!
 * \brief Executor of a computation graph.
 */
class GraphExecutor : public Executor {
 public:
  GraphExecutor() {}
  virtual ~GraphExecutor();
  void Forward(bool is_train) override;
  void PartialForward(bool is_train, int step, int *step_left) override;
  void Backward(const std::vector<NDArray> &head_grads) override;
  const std::vector<NDArray> &outputs() const override {
    return heads_ndarray_;
  }
  void Print(std::ostream &os) const override; // NOLINT(*)
  // install callback
  void SetMonitorCallback(const MonitorCallback& callback) {
    CHECK(callback) << "invalid callback";
    monitor_callback_ = callback;
  }
  // implement Executor::Bind, only call it once.
  inline void Init(Symbol symbol,
                   const Context& default_ctx,
                   const std::map<std::string, Context>& ctx_map,
                   const std::vector<NDArray> &in_args,
                   const std::vector<NDArray> &arg_grad_store,
                   const std::vector<OpReqType> &grad_req_type,
                   const std::vector<NDArray> &aux_states) {
    enable_inplace_allocation_ = dmlc::GetEnv("MXNET_EXEC_ENABLE_INPLACE", true);

    CHECK_EQ(grad_req_type.size(), arg_grad_store.size());
    bool need_backward = false;
    for (auto req : grad_req_type) {
      if (req != kNullOp) need_backward = true;
    }
    this->InitGraph(symbol, default_ctx, ctx_map,
                    in_args, arg_grad_store, grad_req_type,
                    need_backward);
    this->InitDataEntryInfo(in_args, arg_grad_store, grad_req_type, aux_states);
    this->InitDataEntryMemory();
    this->InitResources();
    this->InitOpNodes();
  }

 protected:
  // internal class of wrapping BackwardOp as ForwardOp
  class BackwardOpWrapper;
  // type of data entry
  enum DataEntryType {
    // memory is binded by external NDArray in Bind
    kBindByExternal,
    // to be binded by external NDArray in Forward and Backward
    kTobeBindByExternal,
    // internal memory, allocated
    kInternalAllocated,
    // internal memory, to be allocated
    kNotInitialized
  };
  // Additional information about each data entry
  struct DataEntryInfo {
    // the actual data for the entry
    NDArray data;
    // write request to this entry
    OpReqType op_req;
    // the operatio node that will take
    // this DataEntry as inplace input
    int inplace_op_id;
    // data entry type
    DataEntryType type;
    // shape of this entry
    TShape shape;
    // data type of this entry
    int type_flag;
    // storage id from allocator if it is internal allocation.
    GraphStorageAllocator::StorageID storage_id;
    // reference count on how many times this entry is being used.
    // That is how many operators and heads need this DataEntry
    // this is a temporal variable that is used during initialization.
    uint32_t temp_ref_count;
    // real permanent ref count
    uint32_t ref_count;
    // constructor
    DataEntryInfo()
        : op_req(kNullOp),
          inplace_op_id(-1),
          type(kNotInitialized),
          storage_id(GraphStorageAllocator::kBadStorageID),
          temp_ref_count(0), ref_count(0) {}
  };
  // all the information needed to push the op to engine
  struct OpExecEntry {
    // execution function for
    Engine::AsyncFn exec_fun;
    // variables to read from
    std::vector<Engine::VarHandle> use_vars;
    // variables to mutate
    std::vector<Engine::VarHandle> mutate_vars;
    // constructor
    OpExecEntry() : exec_fun(nullptr) {}
  };
  // Information about operational node
  struct OpNode {
    // whether this op node is activated
    bool activated;
    // the context of the node
    Context ctx;
    // data entry information about outputs of op
    std::vector<DataEntryInfo> outputs;
    // auxiliary data information of op
    std::vector<DataEntryInfo> aux_states;
    // The following parts are constructed in InitOpNodes
    // the real operator
    std::shared_ptr<Operator> op;
    // op context, that is defined for this op.
    OpContext op_ctx;
    // executor, this is only allocated for nodes
    // whose inputs, outputs are pre-defined.
    // otherwise cached_exec.exec_fun == nullptr
    OpExecEntry cached_exec;
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
    // constructor
    OpNode() : activated(false) {}
    // Manual option for delete operator
    // need to do this before delete NDArrays
    inline void DeleteOperator() {
      if (cached_opr != nullptr) {
        Engine::Get()->DeleteOperator(cached_opr);
        cached_opr = nullptr;
      }
    }
  };
  /*!
   * \brief Get input option of a node.
   *  This function is overriden for both Forward and Backward node.
   *
   * \param node_id node index of node in StaticGraph
   * \param in_data the input data entry to the node
   * \param out_data the output data entry in the graph
   * \return the paired inplace option.
   */
  template<typename T>
  inline std::vector<std::pair<T, T> > GetInplaceOption(
      uint32_t node_id,
      const std::vector<T> &in_data,
      const std::vector<T> &out_data) const;
  /*!
   * \brief Get resource requirement of a node.
   *  This function is overriden for both Forward and Backward node.
   * \param node_id node index of node in StaticGraph
   * \return the desired resource request.
   */
  inline std::vector<ResourceRequest> GetResource(uint32_t node_id) const;
  /*!
   * \brief Get number of outputs of a node.
   *  This function is overriden for both Forward and Backward node.
   * \param node_id node index of node in StaticGraph
   * \return the number of outputs of the node.
   */
  inline int GetNumOutputs(uint32_t node_id) const;
  /*!
   * \brief get execution entry for an OpNode.
   *  This function can only be called after initialization is done.
   * \param node_id the id of operational node.
   * \return the execution entry.
   */
  inline OpExecEntry GetOpExecEntry(uint32_t node_id);
  // initialize the internal graph structure
  void InitGraph(const Symbol &symbol,
                 const Context& default_ctx,
                 const std::map<std::string, Context>& ctx_map,
                 const std::vector<NDArray> &in_args,
                 const std::vector<NDArray> &arg_grad_store,
                 const std::vector<OpReqType> &grad_req_type,
                 bool need_backward);
  // initialize internal DataEntryInfo, reference counting
  void InitDataEntryInfo(const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states);
  // initialize internal data entries NDArray
  void InitDataEntryMemory();
  // initialize the internal resources for each op
  void InitResources();
  // initialize OpNode data structure
  void InitOpNodes();
  // assign context to the graph, this will mutate the graph.
  void AssignContext(const Context default_ctx,
                     const std::map<std::string, Context>& ctx_map,
                     const std::vector<NDArray> &in_args,
                     const std::vector<NDArray> &arg_grad_store,
                     const std::vector<OpReqType> &grad_req_type,
                     std::vector<Context> *ctx_plan);
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
  // internal computational graph
  StaticGraph graph_;
  // topological order of nodes in computation graph
  // backward nodes always follow forward nodes
  std::vector<uint32_t> topo_order_;
  // whether to enable inplace space
  bool enable_inplace_allocation_;
  // total allocated space in bytes
  size_t total_allocated_bytes_;
  // total allocated temp space
  size_t total_allocated_temp_;
  // number of forward nodes in the graph
  size_t num_forward_nodes_;
  // head gradient node in the graph, if there is backward pass
  std::vector<uint32_t> head_grad_nodes_;
  // mirror map of nodes, experimental feature, normally can be ignored.
  std::map<uint32_t, uint32_t> mirror_source_map_;
  // argument node in the graph, if there is backward pass
  std::vector<StaticGraph::DataEntry> arg_grads_;
  // operational nodes
  std::vector<OpNode> op_nodes_;
  // head NDArrays
  std::vector<NDArray> heads_ndarray_;
  // monitor call back
  std::function<void(const char*, void*)> monitor_callback_;
};  // class GraphExecutor
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_EXECUTOR_H_
//===== EXPANDED: ../src/symbol/graph_executor.h =====


namespace mxnet {
/*!
 * \brief wrapper class that wraps Backward operation as Forward.
 */
class GraphExecutor::BackwardOpWrapper : public Operator {
 public:
  /*!
   * \brief create a backward Operator wrapper given forward op.
   * \param prop pointer to the property of forward wrapper
   * \param forward_op the shared ptr to Forward operator
   * \return the created wrapper.
   */
  explicit BackwardOpWrapper(const OperatorProperty *prop,
                             std::shared_ptr<Operator> forward_op)
      : op_(forward_op) {
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    out_data_.resize(prop->NumOutputs());

    std::vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    std::vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    std::vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }
  // implement forward
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    // set things correctly
    CHECK(arg_data_ptr_.size() == in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      *(arg_data_ptr_[i]) = in_data[i];
    }
    // redirect internally
    op_->Backward(ctx, out_grad_, in_data_, out_data_, req, out_data, aux_states);
  }
  virtual ExecType exec_type() const {
    return op_->exec_type();
  }

 private:
  /*! \brief internal forward operator */
  std::shared_ptr<Operator> op_;
  /*! \brief internal space for out_grad */
  std::vector<TBlob> out_grad_;
  /*! \brief internal space for in_data */
  std::vector<TBlob> in_data_;
  /*! \brief internal space for out_data */
  std::vector<TBlob> out_data_;
  /*!
   * \brief pointer to places in the internal space.
   *  arg_data_ptr_ maps in_data in Forward to the internal space.
   */
  std::vector<TBlob*> arg_data_ptr_;
};

// get resource
inline std::vector<ResourceRequest>
GraphExecutor::GetResource(uint32_t node_id) const {
  const StaticGraph::Node &node = graph_.nodes[node_id];
  // use input shape
  std::vector<TShape> in_shapes;
  for (StaticGraph::DataEntry e : node.inputs) {
    in_shapes.push_back(op_nodes_[e.source_id].outputs[e.index].shape);
  }

  if (node.is_forward()) {
    return node.op->ForwardResource(in_shapes);
  } else {
    CHECK(node.is_backward());
    return graph_.nodes[node.backward_source_id]
        .op->BackwardResource(in_shapes);
  }
}

inline int GraphExecutor::GetNumOutputs(uint32_t node_id) const {
  const StaticGraph::Node &node = graph_.nodes[node_id];
  if (node.is_forward()) {
    return node.op->NumOutputs();
  } else if (node.is_backward()) {
    return static_cast<int>(
        graph_.nodes[node.backward_source_id].op->ListArguments().size());
  } else {
    CHECK(node.is_variable());
    return 1;
  }
}

// implement get input option
template<typename T>
inline std::vector<std::pair<T, T> > GraphExecutor::GetInplaceOption(
    uint32_t node_id,
    const std::vector<T> &in_data,
    const std::vector<T> &out_data) const {
  // get the node
  const StaticGraph::Node &node = graph_.nodes[node_id];

  if (node.is_forward()) {
    std::vector<int> in_data_index(in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      in_data_index[i] = static_cast<int>(i);
    }
    std::vector<void*> out_data_ptr(out_data.size());
    for (size_t i = 0; i < out_data.size(); ++i) {
      out_data_ptr[i] = (void*)&out_data[i];  // NOLINT(*)
    }
    auto rmap_index = node.op->ForwardInplaceOption(in_data_index, out_data_ptr);
    std::vector<std::pair<T, T> > remap(rmap_index.size());
    for (size_t i = 0; i < remap.size(); ++i) {
      remap[i].first = in_data[rmap_index[i].first];
      remap[i].second = *static_cast<const T*>(rmap_index[i].second);
    }
    return remap;
  } else {
    CHECK(node.is_backward());
    // forward property
    const OperatorProperty *fwd = graph_.nodes[node.backward_source_id].op.get();

    std::vector<int> out_grad_index(fwd->NumVisibleOutputs());
    std::vector<int> in_data_index(fwd->ListArguments().size());
    std::vector<int> out_data_index(fwd->NumOutputs());
    CHECK_EQ(in_data_index.size(), out_data.size());
    int counter = 0;
    for (size_t i = 0; i < out_grad_index.size(); ++i) {
      out_grad_index[i] = counter++;
    }
    for (size_t i = 0; i < in_data_index.size(); ++i) {
      in_data_index[i] = counter++;
    }
    for (size_t i = 0; i < out_data_index.size(); ++i) {
      out_data_index[i] = counter++;
    }
    auto args_index = fwd->DeclareBackwardDependency(
        out_grad_index, in_data_index, out_data_index);
    std::vector<const T*> args_array(counter, nullptr);
    CHECK_EQ(args_index.size(), in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      args_array[args_index[i]] = &in_data[i];
    }
    std::vector<void*> in_grad_ptr(out_data.size());
    for (size_t i = 0; i < in_grad_ptr.size(); ++i) {
      in_grad_ptr[i] = (void*)&out_data[i];  // NOLINT(*)
    }
    auto remap_index = fwd->BackwardInplaceOption(
        out_grad_index, in_data_index, out_data_index, in_grad_ptr);
    std::vector<std::pair<T, T> > remap(remap_index.size());
    for (size_t i = 0; i < remap_index.size(); ++i) {
      if (args_array[remap_index[i].first] == nullptr) {
        LOG(FATAL) << "BackwardInplaceOption not consistent with DeclareBackwardDependency";
      }
      remap[i].first = *args_array[remap_index[i].first];
      remap[i].second = *static_cast<T*>(remap_index[i].second);
    }
    return remap;
  }
}

inline GraphExecutor::OpExecEntry
GraphExecutor::GetOpExecEntry(uint32_t nid) {
  OpNode& op_node = op_nodes_[nid];
  std::vector<OpReqType> req;
  std::vector<NDArray> in_array, out_array, aux_array;
  in_array.reserve(graph_.nodes[nid].inputs.size());
  out_array.reserve(op_node.outputs.size());
  req.reserve(op_node.outputs.size());
  aux_array.reserve(op_node.aux_states.size());

  OpExecEntry exec;
  // output
  for (const DataEntryInfo& out : op_node.outputs) {
    out_array.push_back(out.data);
    exec.mutate_vars.push_back(out.data.var());
    req.push_back(out.op_req);
  }
  // aux
  for (const DataEntryInfo& aux : op_node.aux_states) {
    aux_array.push_back(aux.data);
    exec.mutate_vars.push_back(aux.data.var());
  }
  // input
  for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
    const DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    in_array.push_back(info.data);
    // skip inplace since they already appear in mutate vars
    if (info.inplace_op_id != static_cast<int>(nid)) {
      exec.use_vars.push_back(info.data.var());
    }
  }
  // de-duplicate the used vars
  std::sort(exec.use_vars.begin(), exec.use_vars.end());
  exec.use_vars.resize(std::unique(exec.use_vars.begin(), exec.use_vars.end()) -
                       exec.use_vars.begin());

  // start setup exec function.
  for (const Resource& r : op_node.op_ctx.requested) {
    exec.mutate_vars.push_back(r.var);
  }

  Operator* op = op_node.op.get();
  OpContext* op_ctx_ptr = &op_node.op_ctx;
  bool is_gpu = op_node.ctx.dev_mask() == gpu::kDevMask;
  bool is_async = op->exec_type() == Operator::kAsync;
  exec.exec_fun = [op, is_gpu, is_async, op_ctx_ptr, in_array, req, out_array, aux_array]
      (RunContext ctx, Engine::CallbackOnComplete on_complete) {
    std::vector<TBlob> in_data(in_array.size());
    std::vector<TBlob> out_data(out_array.size());
    std::vector<TBlob> aux_data(aux_array.size());
    std::transform(in_array.begin(), in_array.end(), in_data.begin(), [](const NDArray& nd) {
      return nd.data();
      });
    std::transform(out_array.begin(), out_array.end(), out_data.begin(), [](const NDArray& nd) {
        return nd.data();
      });
    std::transform(aux_array.begin(), aux_array.end(), aux_data.begin(), [](const NDArray& nd) {
        return nd.data();
      });
    op_ctx_ptr->run_ctx = ctx;
    if (is_async) {
      op_ctx_ptr->async_on_complete = on_complete;
    }
    op->Forward(*op_ctx_ptr, in_data, req, out_data, aux_data);
    // call on complete only if it is async op
    if (!is_async) {
      if (is_gpu) {
        #if MXNET_USE_CUDA
        // Wait GPU kernel to finish.
        ctx.get_stream<gpu>()->Wait();
        #else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
      }
      on_complete();
    }
  };
  return exec;
}

GraphExecutor::~GraphExecutor() {
  Engine::Get()->WaitForAll();
  // need to delete the operators before delete the NDArray they referenced.
  for (OpNode& node : op_nodes_) {
    node.DeleteOperator();
  }
}

void GraphExecutor::InitGraph(const Symbol &symbol,
                              const Context& default_ctx,
                              const std::map<std::string, Context>& ctx_map,
                              const std::vector<NDArray> &in_args,
                              const std::vector<NDArray> &arg_grad_store,
                              const std::vector<OpReqType> &grad_req_type,
                              bool need_backward) {
  // initialize all internal data structures
  graph_.FromSymbol(symbol);
  if (need_backward) {
    std::map<uint32_t, uint32_t> mirror;
    graph_.MakeBackwardPass(&head_grad_nodes_, &arg_grads_, &mirror);
    for (auto kv : mirror) {
      if (kv.first != kv.second) {
        mirror_source_map_[kv.second] = kv.first;
      }
    }
  }

  // assign context, this will change the graph.
  std::vector<Context> ctx_assignment;
  this->AssignContext(default_ctx, ctx_map,
                      in_args, arg_grad_store, grad_req_type,
                      &ctx_assignment);

  // organize topo order so that backward node always falls after forward.
  std::vector<uint32_t> head_nodes;
  for (const auto& head : graph_.heads) {
    head_nodes.push_back(head.source_id);
  }
  std::vector<uint32_t> fwd_nodes = graph_.PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
  num_forward_nodes_ = fwd_nodes.size();

  std::unordered_set<uint32_t> fwd_set(fwd_nodes.begin(), fwd_nodes.end());
  std::vector<uint32_t> topo = graph_.TopoSort();
  std::unordered_set<uint32_t> fwd_bwd_set(topo.begin(), topo.end());
  std::vector<uint32_t> backward;

  for (uint32_t nid : fwd_nodes) {
    if (fwd_bwd_set.count(nid) != 0) {
      topo_order_.push_back(nid);
    }
  }
  for (uint32_t nid : topo) {
    if (fwd_set.count(nid) == 0) {
      // TODO(tqchen) find less hacky way to decide mirror node.
      const std::string& name = graph_.nodes[nid].name;
      bool is_mirror = graph_.nodes[nid].is_forward() &&
          name.substr(name.length() - 7, 7) ==  "_mirror";
      if (!is_mirror) backward.push_back(nid);
    }
  }
  std::unordered_set<uint32_t> finished(fwd_nodes.begin(), fwd_nodes.end());
  for (uint32_t nid : backward) {
    std::vector<uint32_t> pass = graph_.PostDFSOrder({nid}, finished);
    topo_order_.insert(topo_order_.end(), pass.begin(), pass.end());
    finished.insert(pass.begin(), pass.end());
  }
  for (uint32_t nid : topo) {
    if (finished.count(nid) == 0) topo_order_.push_back(nid);
  }
  // setup all the operator nodes data structure
  op_nodes_.resize(graph_.nodes.size());
  for (size_t i = 0; i < graph_.nodes.size(); ++i) {
    op_nodes_[i].ctx = ctx_assignment[i];
    op_nodes_[i].outputs.resize(GetNumOutputs(i));
  }
}

void GraphExecutor::AssignContext(const Context default_ctx,
                                  const std::map<std::string, Context>& ctx_map,
                                  const std::vector<NDArray> &in_args,
                                  const std::vector<NDArray> &arg_grad_store,
                                  const std::vector<OpReqType> &grad_req_type,
                                  std::vector<Context> *ctx_plan) {
  ctx_plan->resize(graph_.nodes.size());
  std::vector<bool> assigned(graph_.nodes.size(), false);
  // assign context of node to the binded version
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    uint32_t nid = graph_.arg_nodes[i];
    assigned[nid] = true;
    ctx_plan->at(nid) = in_args[i].ctx();
  }
  if (arg_grads_.size() != 0) {
    for (size_t i = 0; i < arg_grads_.size(); ++i) {
      if (grad_req_type[i] == kNullOp) continue;
      auto& e = arg_grads_[i];
      if (!assigned[e.source_id]) {
        assigned[e.source_id] = true;
        ctx_plan->at(e.source_id) = arg_grad_store[i].ctx();
      } else {
        CHECK(ctx_plan->at(e.source_id) == arg_grad_store[i].ctx())
            << "Inconsistent gradient context requirment";
      }
    }
  }

  // topological sort
  std::vector<uint32_t> topo = graph_.TopoSort();
  // forward prop
  for (uint32_t nid : topo) {
    if (assigned[nid]) continue;
    auto it = graph_.nodes[nid].attr.find("ctx_group");
    if (it != graph_.nodes[nid].attr.end()) {
      const std::string& group = it->second;
      if (ctx_map.count(group) != 0) {
        assigned[nid] = true;
        ctx_plan->at(nid) = ctx_map.at(group);
      } else {
        CHECK(ctx_map.size() == 0)
            << "Context for group " << group << " is not provided in group2ctx map";
      }
    }
    if (assigned[nid]) continue;
    const StaticGraph::Node& node = graph_.nodes[nid];
    if (node.is_backward() && assigned[node.backward_source_id]) {
      ctx_plan->at(nid) = ctx_plan->at(node.backward_source_id);
      assigned[nid] = true;
      continue;
    }
    for (const StaticGraph::DataEntry& e : node.inputs) {
      if (assigned[e.source_id]) {
        ctx_plan->at(nid) = ctx_plan->at(e.source_id);
        assigned[nid] = true;
        break;
      }
    }
  }
  for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
    auto& e = graph_.heads[i];
    uint32_t nid = head_grad_nodes_[i];
    if (assigned[e.source_id]) {
      ctx_plan->at(nid) = ctx_plan->at(e.source_id);
      assigned[nid] = true;
    }
  }
  // backward prop
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    const uint32_t nid = *it;
    if (!assigned[nid]) continue;
    for (const StaticGraph::DataEntry& e : graph_.nodes[nid].inputs) {
      if (!assigned[e.source_id]) {
        ctx_plan->at(e.source_id) = ctx_plan->at(nid);
        assigned[nid] = true;
      }
    }
  }
  // assign rest to default context
  for (uint32_t nid : topo) {
    if (!assigned[nid]) ctx_plan->at(nid) = default_ctx;
  }
  // make sure head gradient is consitent with final operator
  for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
    auto& e = graph_.heads[i];
    uint32_t nid = head_grad_nodes_[i];
    ctx_plan->at(nid) = ctx_plan->at(e.source_id);
  }
  // automatically create copy node
  std::map<StaticGraph::DataEntry, std::map<Context, uint32_t> > copy_node;
  std::vector<StaticGraph::Node> new_nodes;

  for (uint32_t nid : topo) {
    Context curr_ctx = ctx_plan->at(nid);
    for (StaticGraph::DataEntry& e : graph_.nodes[nid].inputs) {
      if (ctx_plan->at(e.source_id) == curr_ctx) continue;

      // create copy node
      std::map<Context, uint32_t>& rmap = copy_node[e];
      if (rmap.count(curr_ctx) == 0) {
        uint32_t new_node_id = static_cast<uint32_t>(graph_.nodes.size() + new_nodes.size());
        // add a new node
        StaticGraph::Node new_node = StaticGraph::CreateCopyNode(e);
        std::ostringstream os;
        os << graph_.nodes[e.source_id].name << '_' << e.index << "_copynode";
        new_node.name = os.str();
        new_nodes.push_back(new_node);
        rmap[curr_ctx] = new_node_id;
        ctx_plan->push_back(curr_ctx);
        CHECK_EQ(ctx_plan->size(), new_node_id + 1);
      }
      // muttate e
      e = StaticGraph::DataEntry(rmap[curr_ctx], 0);
    }
  }
  graph_.nodes.insert(graph_.nodes.end(), new_nodes.begin(), new_nodes.end());
  CHECK_EQ(graph_.nodes.size(), ctx_plan->size());
}

void GraphExecutor::InitDataEntryInfo(const std::vector<NDArray> &in_args,
                                      const std::vector<NDArray> &arg_grad_store,
                                      const std::vector<OpReqType> &grad_req_type,
                                      const std::vector<NDArray> &aux_states) {
  CHECK_EQ(arg_grad_store.size(), grad_req_type.size());
  CHECK_EQ(in_args.size(), graph_.arg_nodes.size());
  // bind inputs
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    DataEntryInfo &info = op_nodes_[graph_.arg_nodes[i]].outputs[0];
    info.type = kBindByExternal;
    info.data = in_args[i];
    CHECK(info.data.ctx() == op_nodes_[graph_.arg_nodes[i]].ctx)
        << "Argument NDArray's context must match the operator's context assignment";
  }
  // setup ref for head nodes
  for (StaticGraph::DataEntry e : graph_.heads) {
    DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    ++info.ref_count;
    op_nodes_[e.source_id].activated = true;
  }
  // need Backward pass
  if (arg_grads_.size() != 0) {
    CHECK_EQ(arg_grads_.size(), arg_grad_store.size());
    CHECK_EQ(arg_grads_.size(), grad_req_type.size());
    // setup gradient placeholders
    for (size_t i = 0; i < arg_grads_.size(); ++i) {
      if (grad_req_type[i] == kNullOp) continue;
      CHECK_NE(grad_req_type[i], kWriteInplace)
          << "Gradient request can only be nullop, add, write";
      StaticGraph::DataEntry &grad_source = arg_grads_[i];
      DataEntryInfo &info = op_nodes_[grad_source.source_id].outputs[grad_source.index];
      info.type = kBindByExternal;
      info.op_req = grad_req_type[i];
      info.data = arg_grad_store[i];
      CHECK(info.data.ctx() == op_nodes_[grad_source.source_id].ctx)
          << "Gradient holder NDArray's context must match the operator's context assignment";
      ++info.ref_count;
      op_nodes_[grad_source.source_id].activated = true;
    }
    // setup head gradient
    for (uint32_t nid : head_grad_nodes_) {
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      info.type = kTobeBindByExternal;
    }
  }
  // update ref counters for all other nodes, in reverse topo order
  for (auto it = topo_order_.rbegin(); it != topo_order_.rend(); ++it) {
    uint32_t nid = *it;
    if (op_nodes_[nid].activated) {
      for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
        DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
        ++info.ref_count;
        op_nodes_[e.source_id].activated = true;
      }
    }
    if (graph_.nodes[nid].is_backward()) {
      op_nodes_[graph_.nodes[nid].backward_source_id].activated = true;
    }
  }
  // shape inference
  std::vector<std::vector<TShape> > out_shapes(op_nodes_.size());
  std::vector<std::vector<TShape> > aux_shapes(op_nodes_.size());
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    out_shapes[i].resize(op_nodes_[i].outputs.size());
  }
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    out_shapes[graph_.arg_nodes[i]][0] = in_args[i].shape();
  }
  CHECK(graph_.InferNodeShapes(topo_order_, &out_shapes, &aux_shapes))
      << "Shape inference cannot be complete in bind";
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    for (size_t j = 0; j < out_shapes[i].size(); ++j) {
      op_nodes_[i].outputs[j].shape = out_shapes[i][j];
    }
  }
  // type inference
  std::vector<std::vector<int> > out_types(op_nodes_.size());
  std::vector<std::vector<int> > aux_types(op_nodes_.size());
  for (size_t i = 0; i < out_types.size(); ++i) {
    out_types[i].resize(op_nodes_[i].outputs.size(), -1);
  }
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    out_types[graph_.arg_nodes[i]][0] = in_args[i].dtype();
  }
  CHECK(graph_.InferNodeTypes(topo_order_, &out_types, &aux_types))
      << "Type inference cannot be complete in bind";
  for (size_t i = 0; i < out_types.size(); ++i) {
    for (size_t j = 0; j < out_types[i].size(); ++j) {
      op_nodes_[i].outputs[j].type_flag = out_types[i][j];
    }
  }
  // bind aux args
  size_t aux_ndarray_idx = 0;
  for (auto i : topo_order_) {
    op_nodes_[i].aux_states.resize(aux_shapes[i].size());
    for (size_t j = 0; j < aux_shapes[i].size(); ++j) {
      DataEntryInfo &info = op_nodes_[i].aux_states[j];
      info.shape = aux_shapes[i][j];
      info.type_flag = aux_types[i][j];
      info.type = kBindByExternal;
      if (mirror_source_map_.count(i) == 0) {
        if (graph_.nodes[i].backward_source_id == -1) {
          info.data = aux_states[aux_ndarray_idx++];
          CHECK(info.data.ctx() == op_nodes_[i].ctx)
              << "Auxiliary NDArray's context must match the operator's context assignment";
        } else {
          CHECK_NE(graph_.nodes[i].backward_source_id, -1)
              << "Input auxiliary NDArray is less than required";
          info.data = op_nodes_[graph_.nodes[i].backward_source_id].aux_states[j].data;
        }
      } else {
        info.data = op_nodes_[mirror_source_map_[i]].aux_states[j].data;
      }
      CHECK_EQ(info.data.data().shape_, info.shape)
          << "Incorrect NDArray shape"
          << " Input: " << info.data.data().shape_
          << " Desired: " << info.shape;
      CHECK_EQ(info.data.dtype(), info.type_flag)
          << "Incorrect NDArray type"
          << " Input: " << info.data.dtype()
          << " Desired: " << info.type_flag;
    }
  }
}

void GraphExecutor::InitDataEntryMemory() {
  // setup the temp ref counter for allocator algorithms
  for (OpNode &op : op_nodes_) {
    for (DataEntryInfo &node : op.outputs) {
      node.temp_ref_count = node.ref_count;
    }
  }

  // use allocator to allocate memory.
  GraphStorageAllocator allocator(&graph_, topo_order_);
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;

    // check inplace option
    std::vector<DataEntryInfo*> in_data;
    in_data.reserve(graph_.nodes[nid].inputs.size());
    // check inputs are ready.
    for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
      DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
      CHECK_NE(info.type, kNotInitialized);
      CHECK_NE(info.temp_ref_count, 0);
      in_data.push_back(&info);
    }
    std::vector<DataEntryInfo*> out_data(op_nodes_[nid].outputs.size());
    for (size_t i = 0; i < op_nodes_[nid].outputs.size(); ++i) {
      out_data[i] = &op_nodes_[nid].outputs[i];
      CHECK_NE(out_data[i]->type, kInternalAllocated);
    }
    auto inplace = GetInplaceOption(nid, in_data, out_data);

    for (std::pair<DataEntryInfo*, DataEntryInfo*> kv : inplace) {
      DataEntryInfo* in = kv.first;
      DataEntryInfo* out = kv.second;
      if (enable_inplace_allocation_ &&
          in->temp_ref_count == 1 &&
          in->type == kInternalAllocated &&
          out->type == kNotInitialized) {
        // we can only do inplace if we are last user of in
        // and out is not initialized.
        out->type = kInternalAllocated;
        out->op_req = kWriteInplace;
        out->storage_id = in->storage_id;
        // set inplace op id
        in->temp_ref_count = 0;
        in->inplace_op_id = static_cast<int>(nid);
      }
    }
    // allocate output,
    for (DataEntryInfo *out : out_data) {
      if (out->op_req == kNullOp && out->temp_ref_count != 0) {
        out->op_req = kWriteTo;
      }
      if (out->type == kNotInitialized) {
        out->storage_id = allocator.Request(
            op_nodes_[nid].ctx, out->type_flag, out->shape, nid);
        out->type = kInternalAllocated;
      }
    }
    // then free inputs
    for (DataEntryInfo *in : in_data) {
      // temp_ref_count == 0 means it is taken by inplace op
      if (in->temp_ref_count == 0) {
        CHECK_EQ(in->inplace_op_id, static_cast<int>(nid));
        continue;
      }
      // if we decrease it to zero, means we are ready to relase
      --in->temp_ref_count;
      if (in->temp_ref_count == 0 && in->type == kInternalAllocated) {
        allocator.Release(in->storage_id, nid);
      }
    }
    // check out again, if there is temp_ref_count == 0, release it
    for (DataEntryInfo *out : out_data) {
      if (out->temp_ref_count == 0 && out->type == kInternalAllocated) {
        allocator.Release(out->storage_id, nid);
      }
    }
  }
  // one pass complete, allocate real memory
  this->total_allocated_bytes_ = allocator.InitStorages();
  // get the real data NDArray into the DataEntryInfo
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    for (DataEntryInfo &out : op_nodes_[nid].outputs) {
      CHECK_NE(out.type, kNotInitialized);
      if (out.type == kInternalAllocated) {
        out.data = allocator.Get(out.storage_id, out.shape);
      }
    }
  }
  // setup heads
  for (StaticGraph::DataEntry e : graph_.heads) {
    DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    CHECK_EQ(info.type, kInternalAllocated);
    heads_ndarray_.push_back(info.data);
  }
}

void GraphExecutor::InitResources() {
  // prepare for temp space allocation
  std::vector<uint32_t> req_temp_cnt(topo_order_.size(), 0);
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    uint32_t cnt = 0;
    for (const ResourceRequest& req : GetResource(nid)) {
      if (req.type == ResourceRequest::kTempSpace) ++cnt;
    }
    CHECK_LE(cnt, 1) << "Node can only have one temp space request";
    req_temp_cnt[nid] = cnt;
  }

  uint32_t num_color = static_cast<uint32_t>(common::GetExecNumMatchColor());
  std::vector<uint32_t> req_temp_color;
  // use graph coloring to find node that won't run in parallel
  num_color = graph::ColorNodeGroup(graph_, topo_order_, req_temp_cnt,
                                    num_color, &req_temp_color);

  // cached resources temp space
  std::map<Context, std::map<uint32_t, Resource> > cached_temp;
  total_allocated_temp_ = 0;

  // Resource allocation
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    const std::vector<ResourceRequest>& reqs = GetResource(nid);
    auto& requested = op_nodes_[nid].op_ctx.requested;
    requested.clear();
    // Get the resource of temporal space.
    for (const ResourceRequest& req : reqs) {
      const Context &ctx = op_nodes_[nid].ctx;
      if (req.type == ResourceRequest::kTempSpace) {
        uint32_t color = req_temp_color[nid];
        // try to reuse graph in same color
        std::map<uint32_t, Resource> &cmap = cached_temp[ctx];
        if (cmap.count(color) != 0) {
          requested.push_back(cmap.at(color));
        } else {
          Resource r = ResourceManager::Get()->Request(ctx, req);
          requested.push_back(r);
          cmap[color] = r;
          ++total_allocated_temp_;
        }
      } else if (req.type == ResourceRequest::kRandom) {
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
      } else {
        LOG(FATAL) << "resource type not yet supported";
      }
    }
  }
}

void GraphExecutor::InitOpNodes() {
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    OpNode& op_node = op_nodes_[nid];
    if (graph_.nodes[nid].is_forward()) {
      op_node.op.reset(graph_.nodes[nid].op->CreateOperator(op_node.ctx));
    } else {
      CHECK(graph_.nodes[nid].is_backward());
      op_node.op.reset(new BackwardOpWrapper(
          graph_.nodes[graph_.nodes[nid].backward_source_id].op.get(),
          op_nodes_[graph_.nodes[nid].backward_source_id].op));
    }
    bool allow_cache = true;
    for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
      DataEntryInfo& info = op_nodes_[e.source_id].outputs[e.index];
      if (info.type == kTobeBindByExternal) allow_cache = false;
    }
    for (DataEntryInfo& info : op_node.outputs) {
      if (info.type == kTobeBindByExternal) allow_cache = false;
    }
    if (allow_cache && op_node.op->exec_type() != Operator::kCrossDeviceCopy) {
      op_node.cached_exec = GetOpExecEntry(nid);
      op_node.cached_opr = Engine::Get()->NewOperator(
          op_node.cached_exec.exec_fun,
          op_node.cached_exec.use_vars,
          op_node.cached_exec.mutate_vars,
          FnProperty::kNormal);
    }
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  for (size_t i = topo_start; i < topo_end; ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    // special handle cross device copy op
    if (opnode.op->exec_type() == Operator::kCrossDeviceCopy) {
      CHECK_EQ(graph_.nodes[nid].inputs.size(), 1);
      CHECK_EQ(opnode.outputs.size(), 1);
      auto in = graph_.nodes[nid].inputs[0];
      CopyFromTo(op_nodes_[in.source_id].outputs[in.index].data,
                 &(opnode.outputs[0].data));
      continue;
    }
    opnode.op_ctx.is_train = is_train;
    if (opnode.cached_opr != nullptr) {
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx);
    } else {
      auto exec = GetOpExecEntry(nid);
      Engine::Get()->PushAsync(
          exec.exec_fun,
          opnode.ctx,
          exec.use_vars,
          exec.mutate_vars,
          FnProperty::kNormal);
    }
    if (monitor_callback_) {
      std::vector<std::string> output_names;
      if (graph_.nodes[nid].is_forward()) {
        output_names = graph_.nodes[nid].op->ListOutputs();
      } else {
        int source_id = graph_.nodes[nid].backward_source_id;
        output_names = graph_.nodes[source_id].op->ListArguments();
      }
      for (index_t i = 0; i < opnode.outputs.size(); ++i) {
        NDArray out_data = opnode.outputs[i].data;
        std::string name = graph_.nodes[nid].name + "_" + output_names[i];
        NDArray *cpy = new NDArray(out_data);
        this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
      }
    }
  }
}

void GraphExecutor::Print(std::ostream &os) const {
  os << "num_forward_nodes=" << num_forward_nodes_ << '\n';
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    os << "Op " << i << ":" << graph_.nodes[nid].name << " ctx=";
    Context ctx = op_nodes_[nid].ctx;
    os << (ctx.dev_mask() == cpu::kDevMask? "cpu" : "gpu");
    os << '(' << ctx.dev_id << ")\n";
    for (size_t j = 0; j < op_nodes_[nid].outputs.size(); ++j) {
      const DataEntryInfo &info = op_nodes_[nid].outputs[j];
      os << "\toutput[" << j << "]: shape=" << info.shape;
      if (info.storage_id != GraphStorageAllocator::kBadStorageID) {
        os << ", storage_id=" << info.storage_id;
      }
      if (info.inplace_op_id != -1) {
        os << ", inplace_consumer=" << graph_.nodes[info.inplace_op_id].name;
      }
      os << '\n';
    }
    for (size_t j = 0; j < op_nodes_[nid].op_ctx.requested.size(); ++j) {
      const Resource& resource = op_nodes_[nid].op_ctx.requested[j];
      os << "\tresource[" << j << "]: ";
      if (resource.req.type == ResourceRequest::kTempSpace) {
        os << "type=TempSpace, id=" << resource.id;
      } else if (resource.req.type == ResourceRequest::kRandom) {
        os << "type=RandomNumber";
      }
      os << '\n';
    }
  }
  os << "Total " << (total_allocated_bytes_ >> 20UL) <<" MB allocated\n";
  os << "Total " << total_allocated_temp_ <<" TempSpace resource requested\n";
}

void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray> &head_grads) {
  if (head_grads.size() != 0) {
    // TODO(bing, min): consider pass a map for backward
    CHECK_EQ(head_grad_nodes_.size(), head_grads.size());
    for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
      uint32_t nid = head_grad_nodes_[i];
      CHECK(graph_.nodes[nid].is_variable());
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      CHECK_EQ(info.type, kTobeBindByExternal);
      info.data = head_grads[i];
      CHECK(op_nodes_[nid].ctx == head_grads[i].ctx())
          << "Head Gradient context do not match the context of output op";
    }
  } else {
    // check all the head_grad_nodes need to have zero ref_count
    // loss function do not need out_grad
    for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
      uint32_t nid = head_grad_nodes_[i];
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      CHECK_EQ(info.ref_count, 0)
          << "Because the last operator is not Loss function, "
          << "head_gradient is required in calling backward.";
    }
  }
  RunOps(true, num_forward_nodes_, topo_order_.size());
}

Executor *Executor::Bind(Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states) {
  GraphExecutor *exec = new GraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_args, arg_grad_store, grad_req_type, aux_states);
  return exec;
}
}  // namespace mxnet
//===== EXPANDED: ../src/symbol/graph_executor.cc =====

//===== EXPANDIND: ../src/symbol/static_graph.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.cc
 * \brief static graph of mxnet
 */

namespace mxnet {

std::vector<uint32_t> StaticGraph::PostDFSOrder(const std::vector<uint32_t>& head_nodes,
                                                const std::unordered_set<uint32_t>& banned) const {
  std::vector<uint32_t> ret;
  std::unordered_set<uint32_t> visited;
  ret.reserve(nodes.size() / 2);
  std::vector<std::pair<uint32_t, uint32_t> > stack;
  // heads
  for (auto head : head_nodes) {
    if (visited.count(head) != 0) continue;
    stack.push_back(std::make_pair(head, 0));
    CHECK_EQ(banned.count(head), 0);
    // bugfix
    visited.insert(head);
    while (!stack.empty()) {
      std::pair<uint32_t, uint32_t>& back = stack.back();
      const Node& n = nodes[back.first];
      if (back.second == n.inputs.size() + (n.is_backward() ? 1 : 0)) {
        ret.push_back(back.first);
        visited.insert(back.first);
        stack.pop_back();
      } else {
        uint32_t input;
        if (back.second == n.inputs.size() && n.is_backward()) {
          input = n.backward_source_id;
          back.second++;
        } else {
          input = n.inputs[back.second++].source_id;
        }
        if (visited.count(input) == 0 && banned.count(input) == 0) {
          stack.push_back(std::make_pair(input, 0));
        }
      }
    }
  }
  return ret;
}

std::vector<uint32_t> StaticGraph::TopoSort() const {
  // out degree
  std::vector<int> out_degree(nodes.size(), 0);
  for (const Node& n : nodes) {
    for (const DataEntry& e : n.inputs) {
      ++out_degree[e.source_id];
    }
    if (n.is_backward()) {
      ++out_degree[n.backward_source_id];
    }
  }
  std::vector<uint32_t> head_nodes;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (out_degree[i] == 0) {
      head_nodes.push_back(static_cast<uint32_t>(i));
    }
  }
  return PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
}

bool StaticGraph::InferNodeShapes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<TShape> > *node_out_shapes,
                                  std::vector<std::vector<TShape> > *node_aux_shapes) const {
  for (uint32_t nid : topo_order) {
    const Node& node = nodes[nid];
    if (node.is_forward()) {
      std::vector<TShape> in_shape;
      for (const DataEntry& e : node.inputs) {
        in_shape.push_back((*node_out_shapes)[e.source_id][e.index]);
      }
      try {
        if (!node.op->InferShape(&in_shape,
                                 &(*node_out_shapes)[nid],
                                 &(*node_aux_shapes)[nid])) return false;
      } catch (const op::InferShapeError &err) {
        // error handling
        const std::string &op_name = node.name;
        std::string arg_name = node.op->ListArguments()[err.index];
        std::ostringstream os;
        os << "InferShape Error in "
           << op_name << "\'s" << ' ' << arg_name << " argument\n";
        auto &source = nodes[node.inputs[err.index].source_id];
        if (source.is_variable()) {
          os << "Corresponding keyword of symbol: " << source.name << '\n' << err.msg;
        }
        throw dmlc::Error(os.str());
      }
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        (*node_out_shapes)[e.source_id][e.index] = in_shape[i];
      }
    } else if (nodes[nid].is_backward()) {
      // simply use shapes from forward pass to assign backward shape
      const Node& forward = nodes[node.backward_source_id];
      CHECK(forward.is_forward());
      std::vector<TShape>& in_grad_shapes = (*node_out_shapes)[nid];
      CHECK(in_grad_shapes.size() == forward.inputs.size());
      // assign the input shape to output gradients
      for (size_t i = 0; i < forward.inputs.size(); ++i) {
        const DataEntry &e = forward.inputs[i];
        try {
          SHAPE_ASSIGN_CHECK(in_grad_shapes, i, (*node_out_shapes)[e.source_id][e.index]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = forward.name;
          std::string arg_name = forward.op->ListArguments()[e.index];
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << ' ' << arg_name << " gradient argument\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
      // consistent check for input shapes
      auto& out_data_shapes = (*node_out_shapes)[node.backward_source_id];
      // use BackwardInputs to select entries corresponding to node.inputs
      auto in_shape = forward.op->BackwardInputs(
          out_data_shapes, in_grad_shapes, out_data_shapes);
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        try {
          SHAPE_ASSIGN_CHECK((*node_out_shapes)[e.source_id], e.index, in_shape[i]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = nodes[e.source_id].name;
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << " gradient values\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }

      // set for auxilary states shape.
      auto& source_aux_shapes = (*node_aux_shapes)[node.backward_source_id];
      for (size_t i = 0; i < source_aux_shapes.size(); ++i) {
        try {
          (*node_aux_shapes)[nid].push_back(source_aux_shapes[i]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = nodes[nid].name;
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << " aux states\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
    }
  }
  // TODO(bing) assign shape for head gradient
  return true;
}

bool StaticGraph::InferNodeTypes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<int> > *node_out_types,
                                  std::vector<std::vector<int> > *node_aux_types) const {
  for (uint32_t nid : topo_order) {
    const Node& node = nodes[nid];
    if (node.is_forward()) {
      std::vector<int> in_type;
      for (const DataEntry& e : node.inputs) {
        in_type.push_back((*node_out_types)[e.source_id][e.index]);
      }
      try {
        if (!node.op->InferType(&in_type,
                                 &(*node_out_types)[nid],
                                 &(*node_aux_types)[nid])) return false;
      } catch (const op::InferTypeError &err) {
        // error handling
        const std::string &op_name = node.name;
        std::string arg_name = node.op->ListArguments()[err.index];
        std::ostringstream os;
        os << "InferType Error in "
           << op_name << "\'s" << ' ' << arg_name << " argument\n";
        auto &source = nodes[node.inputs[err.index].source_id];
        if (source.is_variable()) {
          os << "Corresponding keyword of symbol: " << source.name << '\n' << err.msg;
        }
        throw dmlc::Error(os.str());
      }
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        (*node_out_types)[e.source_id][e.index] = in_type[i];
      }
    } else if (nodes[nid].is_backward()) {
      // simply use types from forward pass to assign backward type
      const Node& forward = nodes[node.backward_source_id];
      CHECK(forward.is_forward());
      std::vector<int>& in_grad_types = (*node_out_types)[nid];
      CHECK(in_grad_types.size() == forward.inputs.size());
      // assign the input type to output gradients
      for (size_t i = 0; i < forward.inputs.size(); ++i) {
        const DataEntry &e = forward.inputs[i];
        try {
          TYPE_ASSIGN_CHECK(in_grad_types, i, (*node_out_types)[e.source_id][e.index]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = forward.name;
          std::string arg_name = forward.op->ListArguments()[e.index];
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << ' ' << arg_name << " gradient argument\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
      // consistent check for input types
      auto& out_data_types = (*node_out_types)[node.backward_source_id];
      // use BackwardInputs to select entries corresponding to node.inputs
      auto in_type = forward.op->BackwardInputs(
          out_data_types, in_grad_types, out_data_types);
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        try {
          TYPE_ASSIGN_CHECK((*node_out_types)[e.source_id], e.index, in_type[i]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = nodes[e.source_id].name;
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << " gradient values\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }

      // set for auxilary states type.
      auto& source_aux_types = (*node_aux_types)[node.backward_source_id];
      for (size_t i = 0; i < source_aux_types.size(); ++i) {
        try {
          (*node_aux_types)[nid].push_back(source_aux_types[i]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = nodes[nid].name;
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << " aux states\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
    }
  }
  // TODO(bing) assign type for head gradient
  return true;
}

bool StaticGraph::InferShape(std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape,
                             std::vector<TShape> *aux_shape) const {
  std::vector<std::vector<TShape> > node_out_shapes(nodes.size());
  std::vector<std::vector<TShape> > node_aux_shapes(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].is_forward()) {
      nout = nodes[i].op->NumOutputs();
    } else if (nodes[i].is_backward()) {
      nout = static_cast<int>(nodes[nodes[i].backward_source_id].inputs.size());
    }
    node_out_shapes[i].resize(nout);
  }
  CHECK(in_shape->size() == arg_nodes.size())
        << "Wrong number of inputs to infer shape";
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    node_out_shapes[arg_nodes[i]][0] = (*in_shape)[i];
  }
  if (!InferNodeShapes(this->TopoSort(),
                       &node_out_shapes,
                       &node_aux_shapes)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_shape)[i] = node_out_shapes[arg_nodes[i]][0];
  }
  out_shape->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_shape)[i] = node_out_shapes[e.source_id][e.index];
  }

  // set back auxiliary nodes.
  aux_shape->clear();
  std::vector<uint32_t> head_nodes;
  for (const auto& head : heads) {
    head_nodes.push_back(head.source_id);
  }
  std::vector<uint32_t> fwd_nodes = PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
  uint32_t counter = 0;
  for (uint32_t nid : fwd_nodes) {
    // backward consistentcy check.
    CHECK(nid == counter++);
    if (node_aux_shapes[nid].size() > 0) {
      for (auto const &shape : node_aux_shapes[nid]) {
        aux_shape->push_back(shape);
      }
    }
  }
  return true;
}

bool StaticGraph::InferType(std::vector<int> *in_type,
                             std::vector<int> *out_type,
                             std::vector<int> *aux_type) const {
  std::vector<std::vector<int> > node_out_types(nodes.size());
  std::vector<std::vector<int> > node_aux_types(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].is_forward()) {
      nout = nodes[i].op->NumOutputs();
    } else if (nodes[i].is_backward()) {
      nout = static_cast<int>(nodes[nodes[i].backward_source_id].inputs.size());
    }
    node_out_types[i].resize(nout, -1);
  }
  CHECK(in_type->size() == arg_nodes.size())
        << "Wrong number of inputs to infer type";
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    node_out_types[arg_nodes[i]][0] = (*in_type)[i];
  }
  if (!InferNodeTypes(this->TopoSort(),
                       &node_out_types,
                       &node_aux_types)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_type)[i] = node_out_types[arg_nodes[i]][0];
  }
  out_type->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_type)[i] = node_out_types[e.source_id][e.index];
  }

  // set back auxiliary nodes.
  aux_type->clear();
  std::vector<uint32_t> head_nodes;
  for (const auto& head : heads) {
    head_nodes.push_back(head.source_id);
  }
  std::vector<uint32_t> fwd_nodes = PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
  uint32_t counter = 0;
  for (uint32_t nid : fwd_nodes) {
    // backward consistentcy check.
    CHECK(nid == counter++);
    if (node_aux_types[nid].size() > 0) {
      for (auto const &type : node_aux_types[nid]) {
        aux_type->push_back(type);
      }
    }
  }
  return true;
}

StaticGraph::Node StaticGraph::CreateSumNode(
    const std::vector<DataEntry> &grad_source) {
  // find multiple gradients, need aggregate
  std::ostringstream os_size;
  Node agg_node;
  agg_node.op.reset(OperatorProperty::Create("ElementWiseSum"));
  os_size << grad_source.size();
  agg_node.op->Init({{"num_args", os_size.str()}});
  agg_node.inputs = grad_source;
  return agg_node;
}

StaticGraph::Node StaticGraph::CreateCopyNode(const DataEntry &source) {
  // find multiple gradients, need aggregate
  Node copy_node;
  copy_node.op.reset(OperatorProperty::Create("_CrossDeviceCopy"));
  copy_node.inputs = {source};
  return copy_node;
}

void StaticGraph::MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                                   std::vector<DataEntry>* arg_grads,
                                   std::map<uint32_t, uint32_t>* out_mirror_map) {
  // get topo order of nodes, before new nodes are added
  std::vector<uint32_t> topo_order = TopoSort();

  // build a mirror map, experimental
  std::map<uint32_t, uint32_t>& mirror_map = *out_mirror_map;
  mirror_map.clear();
  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  int mirror_step = dmlc::GetEnv("MXNET_BACKWARD_MIRROR_STEP", 100);
  int counter = 0;
  int *pcounter = &counter;

  auto need_mirror = [this, do_mirror, pcounter, mirror_step](uint32_t nid) {
    if (do_mirror == 0) return false;
    if (!nodes[nid].is_forward()) return false;
    std::string type = nodes[nid].op->TypeString();
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Dropout") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    ++pcounter[0];
    if (pcounter[0] % mirror_step == 0) return false;
    return true;
  };

  for (uint32_t nid : topo_order) {
    if (need_mirror(nid)) {
      uint32_t dup_node_id = static_cast<uint32_t>(nodes.size());
      Node node(nodes[nid]);
      node.name += "_mirror";
      for (DataEntry& e : node.inputs) {
        e.source_id = mirror_map.at(e.source_id);
      }
      nodes.push_back(std::move(node));
      mirror_map[nid] = dup_node_id;
    } else {
      mirror_map[nid] = nid;
    }
  }

  // normal gradient
  arg_grads->clear();
  head_grad_nodes->clear();
  // map out_data entry to out_grad
  std::map<DataEntry, std::vector<DataEntry> > grad_map;
  // allocate head gradient nodes
  for (DataEntry head : heads) {
    Node node;
    std::ostringstream os;
    os << nodes[head.source_id].name << '_' << head.index << "_grad";
    // TODO(bing): add index to name
    node.name = os.str();
    // node id
    uint32_t nid = static_cast<uint32_t>(nodes.size());
    nodes.push_back(std::move(node));
    // create a variable node for gradient input
    DataEntry igrad(nid, 0);
    head_grad_nodes->push_back(nid);
    // update gradient map
    auto it = grad_map.find(head);
    if (it == grad_map.end()) {
      grad_map[head] = {igrad};
    } else {
      it->second.push_back(igrad);
    }
  }
  // do backward pass traverse
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    uint32_t nid = *it;
    uint32_t mirror_nid = mirror_map[nid];
    // skip variables
    if (nodes[nid].is_variable()) continue;
    CHECK(nodes[nid].is_forward()) << "Do not support Backward of Backward";
    // get out_grad and out_data entry
    std::vector<DataEntry> out_grad, out_data;
    // nvisible is out_grad.size()
    int nvisible = nodes[nid].op->NumVisibleOutputs();
    // ntotal is out_data.size()
    int ntotal = nodes[nid].op->NumOutputs();
    // check all outpus
    for (int i = 0; i < ntotal; ++i) {
      DataEntry odata(mirror_nid, static_cast<uint32_t>(i));
      DataEntry okey(nid, static_cast<uint32_t>(i));
      out_data.push_back(odata);
      if (i >= nvisible) continue;
      // get out_grad
      auto it = grad_map.find(okey);
      CHECK(it != grad_map.end()) << "bad graph";
      std::vector<DataEntry> &gnodes = it->second;
      if (gnodes.size() == 1) {
        out_grad.push_back(gnodes[0]);
      } else {
        std::ostringstream os_name;
        Node agg_node = StaticGraph::CreateSumNode(gnodes);
        os_name << nodes[nid].name << '_' << i << "_out_grad_agg";
        agg_node.name = os_name.str();
        uint32_t agg_node_id = static_cast<uint32_t>(nodes.size());
        nodes.push_back(std::move(agg_node));
        out_grad.push_back(DataEntry(agg_node_id, 0));
      }
    }
    // Create a gradient backward node
    Node grad_node;
    // Point to the corresponding source
    grad_node.backward_source_id = mirror_nid;

    std::vector<DataEntry> source_inputs;
    for (const DataEntry& e : nodes[nid].inputs) {
      source_inputs.push_back(DataEntry(mirror_map[e.source_id], e.index));
    }
    // select out the dependent inputs
    grad_node.inputs = nodes[mirror_nid].op->BackwardInputs(
        out_grad, source_inputs, out_data);

    grad_node.name = nodes[mirror_nid].name + "_backward";
    uint32_t grad_node_id = static_cast<uint32_t>(nodes.size());
    nodes.push_back(std::move(grad_node));
    // update gradient map
    for (size_t i = 0; i < nodes[nid].inputs.size(); ++i) {
      DataEntry idata = nodes[nid].inputs[i];
      DataEntry igrad(grad_node_id, static_cast<uint32_t>(i));
      auto it = grad_map.find(idata);
      if (it == grad_map.end()) {
        grad_map[idata] = {igrad};
      } else {
        it->second.push_back(igrad);
      }
    }
  }
  // create return values of arg_grads
  arg_grads->resize(arg_nodes.size());
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    DataEntry odata(arg_nodes[i], 0);
    auto it = grad_map.find(odata);
    CHECK(it != grad_map.end()) << "bad graph";
    if (it->second.size() == 1) {
      arg_grads->at(i) = it->second[0];
    } else {
      std::ostringstream os_name;
      Node agg_node = StaticGraph::CreateSumNode(it->second);
      os_name << nodes[arg_nodes[i]].name << "_grad_agg";
      agg_node.name = os_name.str();
      uint32_t agg_node_id = static_cast<uint32_t>(nodes.size());
      nodes.push_back(std::move(agg_node));
      arg_grads->at(i) = DataEntry(agg_node_id, 0);
    }
  }
}

void StaticGraph::Node::Save(dmlc::JSONWriter *writer) const {
  writer->BeginObject();
  if (op.get() != nullptr) {
    writer->WriteObjectKeyValue("op", op->TypeString());
    std::map<std::string, std::string> param = op->GetParams();
    writer->WriteObjectKeyValue("param", param);
  } else {
    std::map<std::string, std::string> empty_param;
    std::string json_null = "null";
    writer->WriteObjectKeyValue("op", json_null);
    writer->WriteObjectKeyValue("param", empty_param);
  }
  writer->WriteObjectKeyValue("name", name);
  writer->WriteObjectKeyValue("inputs", inputs);
  writer->WriteObjectKeyValue("backward_source_id", backward_source_id);
  if (attr.size() != 0) writer->WriteObjectKeyValue("attr", attr);
  writer->EndObject();
}

void StaticGraph::Node::Load(dmlc::JSONReader *reader) {
  attr.clear();
  dmlc::JSONObjectReadHelper helper;
  std::string op_type_str;
  std::map<std::string, std::string> param;
  helper.DeclareField("op", &op_type_str);
  helper.DeclareField("param", &param);
  helper.DeclareField("name", &name);
  helper.DeclareField("inputs", &inputs);
  helper.DeclareField("backward_source_id", &backward_source_id);
  helper.DeclareOptionalField("attr", &attr);
  helper.ReadAllFields(reader);

  if (op_type_str != "null") {
    op.reset(OperatorProperty::Create(op_type_str.c_str()));
    std::vector<std::pair<std::string, std::string> > vec(param.begin(), param.end());
    op->Init(vec);
  } else {
    op.reset(nullptr);
  }
}

void StaticGraph::Save(dmlc::JSONWriter *writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("nodes", nodes);
  writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
  writer->WriteObjectKeyValue("heads", heads);
  writer->EndObject();
}

void StaticGraph::Load(dmlc::JSONReader *reader) {
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("nodes", &nodes);
  helper.DeclareField("arg_nodes", &arg_nodes);
  helper.DeclareField("heads", &heads);
  helper.ReadAllFields(reader);
}
}  // namespace mxnet
//===== EXPANDED: ../src/symbol/static_graph.cc =====

//===== EXPANDIND: ../src/symbol/symbol.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief symbol of mxnet
 */

namespace mxnet {
/*!
 * \brief Node is represents node of an operator in the symbolic graph.
 *
 * It stores connection to the inputs to function represented by OperatorProperty
 * NOTE on data structure: there are three types of node:
 * - Normal node: contains all the necessary elements of a graph.
 * - OperatorProperty: the inputs_ is empty, represents an OperatorProperty that has not been applied.
 * - Variable: the sym_ is nullptr, represents an named Variable of tensors that can be composed.
 */
struct Symbol::Node {
  /*! \brief Operator of this node */
  std::unique_ptr<OperatorProperty> op;
  /*! \brief name of the node */
  std::string name;
  /*! \brief inputs to this node */
  std::vector<DataEntry> inputs;
  /*! \brief source node of the current node */
  std::shared_ptr<Symbol::Node> backward_source_node;
  /*!
   * \brief additional attributes about the node,
   *  Use pointer to save space, as attr can be accessed in a slow way,
   *  not every node will have attributes.
   */
  std::unique_ptr<std::map<std::string, std::string> > attr;
  /*!
    *\brief constructor
    *\param op the OperatorProperty to construct the Node
    *\param name the name of the symbol
   */
  explicit Node(OperatorProperty *op,
                const std::string& name)
      : op(op), name(name) {}
  /*!
    *\brief copy constructor constructor
   */
  explicit Node(const Node& other)
      : name(other.name) {
    if (other.op != nullptr) {
      op.reset(other.op->Copy());
    }
    if (other.attr.get() != nullptr) {
      attr.reset(new std::map<std::string, std::string>(*(other.attr)));
    }
  }
  /*! \return Whether the symbol is atomic */
  inline bool is_atomic() const {
    return inputs.size() == 0 && op != nullptr;
  }
  /*! \return Whether it is unit variable */
  inline bool is_variable() const {
    return op == nullptr && !backward_source_node;
  }
  /*! \return Whether it is backward op */
  inline bool is_backward() const {
    return backward_source_node.get() != nullptr;
  }
};

/*! \return whwther the symbol is atomic */
inline bool Symbol::is_atomic() const {
  return heads_[0].source->is_atomic();
}
// implementation of template functions
template<typename FVisit>
inline void Symbol::DFSVisit(FVisit fvisit) const {
  std::vector<std::pair<const std::shared_ptr<Node>*, uint32_t> > stack;
  std::unordered_set<Node*> visited;
  // put the head into the graph
  for (auto &head : heads_) {
    Node* ptr = head.source.get();
    if (visited.count(ptr) == 0) {
      stack.push_back(std::make_pair(&head.source, 0));
      visited.insert(ptr);
    }
    while (!stack.empty()) {
      std::pair<const std::shared_ptr<Node> *, uint32_t>& back = stack.back();
      if (back.second == back.first->get()->inputs.size()) {
        fvisit(*(back.first));
        stack.pop_back();
      } else {
        std::vector<Symbol::DataEntry>& inputs = back.first->get()->inputs;
        Symbol::DataEntry& input = inputs.at(back.second++);
        Node* ptr = input.source.get();
        if (visited.count(ptr) == 0) {
          stack.push_back(std::make_pair(&input.source, 0));
          visited.insert(ptr);
        }
      }
    }
  }
}

// helper function to handle keyword argument mismatch
// throw approperiate messages
inline void KeywordArgumentMismatch(const char *source,
                                    const std::vector<std::string> &user_args,
                                    const std::vector<std::string> &args) {
  std::unordered_set<std::string> keys(args.begin(), args.end());
  std::ostringstream head, msg;
  msg << "\nCandidate arguments:\n";
  for (size_t i = 0; i < args.size(); ++i) {
    msg << "\t[" << i << ']' << args[i] << '\n';
  }

  for (const auto& key : user_args) {
    if (keys.count(key) == 0) {
      LOG(FATAL) << source
                 << "Keyword argument name " << key << " not found."
                 << msg.str();
    }
  }
}

int Symbol::FindDuplicateArgs(std::unordered_map<std::string, int> *out) const {
  out->clear();
  int max_dup = 1;
  this->DFSVisit([out, &max_dup](const std::shared_ptr<Node> &node) {
      if (node->is_variable()) {
        auto iter = out->find(node->name);
        if (iter == out->end()) {
          (*out)[node->name] = 1;
        } else {
          ++iter->second;
          max_dup = std::max(max_dup, iter->second);
        }
      }
    });
  return max_dup;
}

// public functions
Symbol Symbol::Copy() const {
  std::unordered_map<Node*, std::shared_ptr<Node> > old_new;
  // use DFSVisit to copy all the nodes
  this->DFSVisit([&old_new](const std::shared_ptr<Node> &node) {
      old_new[node.get()] =  std::make_shared<Node>(*node);
    });
  // connect nodes of new graph
  for (const auto &kv : old_new) {
    for (const DataEntry& n : kv.first->inputs) {
      Node *ptr = n.source.get();
      kv.second->inputs.push_back(DataEntry(old_new[ptr], n.index));
    }
  }
  // set the head
  Symbol s;
  for (auto &head : heads_) {
    s.heads_.push_back(DataEntry(old_new[head.source.get()], head.index));
  }
  return s;
}

void Symbol::Print(std::ostream &os) const {
  if (this->is_atomic()) {
    os << "AtomicFunction "<< " Type:" << heads_[0].source->op->TypeString() << '\n'
       << "Inputs:";
    std::vector<std::string> args = this->ListArguments();
    for (size_t i = 0; i < args.size(); ++i) {
      os << "\targ[" << i << "]=" << args[i] << "\n";
    }
  } else {
    // use DFSVisit to copy all the nodes
    os << "Outputs:\n";
    for (size_t i = 0; i < heads_.size(); ++i) {
      os << "\toutput[" << i << "]=" << heads_[i].source->name
         << '(' << heads_[i].index << ")\n";
    }
    this->DFSVisit([&os](const std::shared_ptr<Node> &node) {
        if (node->is_variable()) {
          os << "Variable:" << node->name << '\n';
        } else {
          std::string type_string;
          if (!node->backward_source_node) {
            type_string = node->op->TypeString();
          } else {
            type_string = node->backward_source_node->op->TypeString();
          }
          os << "Name: " << node->name << " Type:" << type_string << '\n'
             << "Inputs:\n";
          for (size_t i = 0; i < node->inputs.size(); ++i) {
            os << "\targ[" << i << "]=" << node->inputs[i].source->name
               << '(' << node->inputs[i].index << ")\n";
          }
        }
      });
  }
}

std::vector<std::string> Symbol::ListArguments() const {
  std::vector<std::string> ret;
  if (this->is_atomic()) {
    return heads_[0].source->op->ListArguments();
  } else {
    this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
        if (node->is_variable()) {
          ret.push_back(node->name);
        }
      });
    return ret;
  }
}

std::vector<std::string> Symbol::ListOutputs() const {
  std::vector<std::string> ret;
  for (auto &head : heads_) {
    if (head.source->is_variable()) {
      ret.push_back(head.source->name);
    } else {
      auto &hname = head.source->name;
      std::string rname;
      if (head.source->is_backward()) {
        rname = head.source->backward_source_node->op->ListArguments()[head.index];
      } else {
        rname = head.source->op->ListOutputs()[head.index];
      }
      if (hname.length() == 0) {
        ret.push_back(std::move(rname));
      } else {
        ret.push_back(hname + '_' + rname);
      }
    }
  }
  return ret;
}

std::vector<std::string> Symbol::ListAuxiliaryStates() const {
  std::vector<std::string> ret;
  if (this->is_atomic()) {
    return heads_[0].source->op->ListAuxiliaryStates();
  } else {
    this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
        if (node->op != nullptr) {
          auto aux_args = node->op->ListAuxiliaryStates();
          if (aux_args.size() > 0) {
            auto &hname = node->name;
            for (auto const &aux : aux_args) {
              ret.push_back(hname + '_' + aux);
            }
          }
        }
      });
    return ret;
  }
}

Symbol Symbol::operator[] (size_t index) const {
  size_t nreturn = NumOutputs();
  CHECK_LT(index, nreturn) << "Symbol only accept nonnegative index";
  if (nreturn == 1) {
    return *this;
  } else {
    Symbol s;
    s.heads_.push_back(heads_[index]);
    return s;
  }
}

Symbol Symbol::GetInternals() const {
  Symbol ret;
  this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
      Node* n = node.get();
      uint32_t nout;
      if (n->is_variable()) {
        nout = 1;
      } else if (n->is_backward()) {
        nout = static_cast<uint32_t>(n->backward_source_node->inputs.size());
      } else {
        nout = n->op->NumVisibleOutputs();
      }
      for (uint32_t i = 0; i < nout; ++i) {
        ret.heads_.push_back(DataEntry(node, i));
      }
    });
  return ret;
}

// create a default variable name
inline std::string DefaultVarName(const std::string &op_name,
                                  const std::string &arg_name) {
  if (op_name.length() == 0) {
    return arg_name;
  } else {
    return op_name + '_' + arg_name;
  }
}

void Symbol::Compose(const std::vector<Symbol>& args,
                     const std::string& name) {
  // CHECK_EQ(NumOutputs(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_EQ(args[i].NumOutputs(), 1)
        << "Argument " << i << " is a tuple with " <<  args[i].NumOutputs()
        << " elements, scalar is required";
  }
  // positional arguments requires all arguments for now.
  // TODO(bing) consider partial assignments
  if (this->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
    CHECK_LE(args.size(), req_args.size())
        << "Incorrect number of arguments, requires " << req_args.size()
        << ", provided " << args.size();
    heads_[0].source->inputs.resize(req_args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      heads_[0].source->inputs[i] = args[i].heads_[0];
    }
    for (size_t i = args.size(); i < req_args.size(); ++i) {
      heads_[0].source->inputs[i] = DataEntry(
          std::make_shared<Node>(nullptr, DefaultVarName(name, req_args[i])), 0);
      // also copy attribute of operator over to automatically created variable
      if (heads_[0].source->attr.get() != nullptr) {
        heads_[0].source->inputs[i].source->attr.reset(
            new std::map<std::string, std::string>(*(heads_[0].source->attr)));
      }
    }
  } else {
    // find all the place holders
    size_t arg_counter = 0;
    std::unordered_map<Node*, const DataEntry*> replace_map;
    std::vector<std::pair<DataEntry*, const DataEntry*> > replace_plan;
    // replace map stores the existing replacement plan for arguments node
    this->DFSVisit([&arg_counter, &replace_map, &replace_plan, &args]
                   (const std::shared_ptr<Node> &node) {
        // visit all the childs, find possible replacement
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          DataEntry *e = &(node->inputs[i]);
          if (e->source->is_variable()) {
            const DataEntry *target = nullptr;
            auto iter = replace_map.find(e->source.get());
            if (iter == replace_map.end()) {
              if (arg_counter < args.size()) {
                target = &(args[arg_counter].heads_[0]);
                replace_map[e->source.get()] = target;
              }
              ++arg_counter;
            } else {
              target = iter->second;
            }
            replace_plan.push_back(std::make_pair(e, target));
          }
        }
      });
    CHECK_EQ(args.size(), arg_counter)
        << "Incorrect number of arguments, requires " << arg_counter
        << ", provided " << args.size();
    // now run the replacement
    for (const auto& kv : replace_plan) {
      *(kv.first) = *(kv.second);
    }
  }
}

void Symbol::Compose(const std::unordered_map<std::string, Symbol>& kwargs,
                     const std::string& name) {
  // CHECK_EQ(NumOutputs(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (const auto& kv : kwargs) {
    CHECK_EQ(kv.second.NumOutputs(), 1)
        << "Keyword Argument " << kv.first << " is a tuple, scalar is required";
  }
  size_t nmatched = 0;
  if (this->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
    heads_[0].source->inputs.resize(req_args.size());
    for (size_t i = 0; i < req_args.size(); ++i) {
      auto iter = kwargs.find(req_args[i]);
      if (iter != kwargs.end()) {
        heads_[0].source->inputs[i] = iter->second.heads_[0];
        ++nmatched;
      } else {
        heads_[0].source->inputs[i] = DataEntry(
            std::make_shared<Node>(nullptr, DefaultVarName(name, req_args[i])), 0);
        // also copy attribute of operator over to automatically created variable
        if (heads_[0].source->attr.get() != nullptr) {
          heads_[0].source->inputs[i].source->attr.reset(
              new std::map<std::string, std::string>(*(heads_[0].source->attr)));
        }
      }
    }
    // if things goes wrong recover the old state
    if (nmatched != kwargs.size()) {
      heads_[0].source->inputs.clear();
    }
  } else {
    // find all the arguments positions
    std::unordered_map<std::string, int> dup_args;
    int max_dup = this->FindDuplicateArgs(&dup_args);
    if (max_dup > 1) {
      for (const auto& kv : dup_args) {
        CHECK_EQ(kv.second, 1)
            << " Argument name=\"" << kv.first << "\" occured in "
            << kv.second << " places in the Symbol, "
            << "Keyword argument call is not supported because this duplication.";
      }
    }
    CHECK_EQ(max_dup, 1);
    std::vector<std::pair<DataEntry*, const DataEntry*> > replace_plan;
    std::unordered_set<Node *> visited;
    // replace map stores the existing replacement plan for arguments node
    this->DFSVisit([&nmatched, &visited, &kwargs, &replace_plan]
                   (const std::shared_ptr<Node> &node) {
        // visit all the childs, find possible replacement
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          DataEntry *e = &(node->inputs[i]);
          if (e->source->is_variable()) {
            const DataEntry *target = nullptr;
            auto iter = kwargs.find(e->source->name);
            if (iter != kwargs.end()) {
              target = &(iter->second.heads_[0]);
              // count how many arguments have been matched.
              if (visited.count(e->source.get()) == 0) {
                visited.insert(e->source.get());
                ++nmatched;
              }
              replace_plan.push_back(std::make_pair(e, target));
            }
          }
        }
      });
    if (nmatched == kwargs.size()) {
      for (const auto& kv : replace_plan) {
        *(kv.first) = *(kv.second);
      }
    }
  }
  if (nmatched != kwargs.size()) {
    std::vector<std::string> keys(kwargs.size());
    std::transform(kwargs.begin(), kwargs.end(), keys.begin(),
                   [](decltype(*kwargs.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.Compose", keys, ListArguments());
  }
}

void Symbol::SetAttr(const std::string &key, const std::string& value) {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.SetAttr only works for non-grouped symbol";
  }
  if (node->attr.get() == nullptr) {
    node->attr.reset(new std::map<std::string, std::string>());
  }
  (*node->attr)[key] = value;
}

bool Symbol::GetAttr(const std::string& key, std::string* out) {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.GetAttr only works for non-grouped symbol";
  }
  if (node->attr.get() == nullptr) return false;
  auto it = node->attr->find(key);
  if (it == node->attr->end()) return false;
  *out = it->second;
  return true;
}

Symbol Symbol::operator () (const std::vector<Symbol>& args,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(args, name);
  return s;
}

Symbol Symbol::operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(kwargs, name);
  return s;
}

Symbol Symbol::Grad(const std::vector<std::string>& wrt) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  uint32_t num_nodes = g.nodes.size();
  std::vector<uint32_t> head_grad_nodes;
  std::vector<StaticGraph::DataEntry> arg_grads;
  // mirror is need to be disabled here.
  std::map<uint32_t, uint32_t> mirror;
  g.MakeBackwardPass(&head_grad_nodes, &arg_grads, &mirror);

  std::vector<std::shared_ptr<Node> > shared_node;
  this->DFSVisit([&shared_node](const std::shared_ptr<Node> &n) {
      shared_node.push_back(n);
    });
  for (std::vector<StaticGraph::Node>::const_iterator it = g.nodes.begin() + num_nodes;
       it != g.nodes.end(); ++it) {
    auto sym_node = std::make_shared<Symbol::Node>(nullptr, it->name);
    if (it->backward_source_id != -1) {
      sym_node->backward_source_node = shared_node[it->backward_source_id];
    }
    shared_node.push_back(sym_node);
    for (auto e : it->inputs) {
      Symbol::DataEntry entry(shared_node[e.source_id], e.index);
      sym_node->inputs.push_back(std::move(entry));
    }
  }
  // make arg lookup dict
  auto arg_list = ListArguments();
  std::unordered_map<std::string, uint32_t> arg_index;
  for (uint32_t i = 0; i < arg_list.size(); ++i) {
    arg_index[arg_list[i]] = i;
  }
  // generate the heads
  Symbol ret;
  for (const std::string& name : wrt) {
    if (arg_index.find(name) != arg_index.end()) {
      uint32_t index = arg_index[name];
      Symbol::DataEntry entry(shared_node[arg_grads[index].source_id], arg_grads[index].index);
      ret.heads_.push_back(entry);
    } else {
      KeywordArgumentMismatch("Symbol.Grad ", wrt, arg_list);
    }
  }
  return ret;
}

bool Symbol::InferShape(std::vector<TShape> *arg_shapes,
                        std::vector<TShape> *out_shapes,
                        std::vector<TShape> *aux_shapes) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  return g.InferShape(arg_shapes, out_shapes, aux_shapes);
}

bool Symbol::InferShape(const std::unordered_map<std::string, TShape>& known_arg_shapes,
                        std::vector<TShape> *arg_shapes,
                        std::vector<TShape> *out_shapes,
                        std::vector<TShape> *aux_shapes) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  arg_shapes->clear();
  arg_shapes->resize(g.arg_nodes.size(), TShape());
  size_t nmatched = 0;
  for (size_t i = 0; i < g.arg_nodes.size(); ++i) {
    const std::string& name = g.nodes[g.arg_nodes[i]].name;
    auto it = known_arg_shapes.find(name);
    if (it != known_arg_shapes.end()) {
      arg_shapes->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_shapes.size()) {
    std::vector<std::string> keys(known_arg_shapes.size());
    std::transform(known_arg_shapes.begin(), known_arg_shapes.end(), keys.begin(),
                   [](decltype(*known_arg_shapes.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.InferShape", keys, ListArguments());
  }
  return g.InferShape(arg_shapes, out_shapes, aux_shapes);
}

bool Symbol::InferType(std::vector<int> *arg_types,
                        std::vector<int> *out_types,
                        std::vector<int> *aux_types) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  return g.InferType(arg_types, out_types, aux_types);
}

bool Symbol::InferType(const std::unordered_map<std::string, int>& known_arg_types,
                        std::vector<int> *arg_types,
                        std::vector<int> *out_types,
                        std::vector<int> *aux_types) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  arg_types->clear();
  arg_types->resize(g.arg_nodes.size(), -1);
  size_t nmatched = 0;
  for (size_t i = 0; i < g.arg_nodes.size(); ++i) {
    const std::string& name = g.nodes[g.arg_nodes[i]].name;
    auto it = known_arg_types.find(name);
    if (it != known_arg_types.end()) {
      arg_types->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_types.size()) {
    std::vector<std::string> keys(known_arg_types.size());
    std::transform(known_arg_types.begin(), known_arg_types.end(), keys.begin(),
                   [](decltype(*known_arg_types.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.InferType", keys, ListArguments());
  }
  return g.InferType(arg_types, out_types, aux_types);
}


void Symbol::Save(dmlc::JSONWriter *writer) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  g.Save(writer);
}

void Symbol::Load(dmlc::JSONReader *reader) {
  StaticGraph g;
  g.Load(reader);
  this->FromStaticGraph(g);
}

Symbol Symbol::Create(OperatorProperty *op)  {
  // use special representation for atomic symbol
  auto node = std::make_shared<Node>(op, "");
  size_t nret = op->NumVisibleOutputs();
  Symbol s;
  for (uint32_t i = 0; i < nret; ++i) {
    s.heads_.push_back(DataEntry(node, i));
  }
  return s;
}

Symbol Symbol::CreateGroup(const std::vector<Symbol> &symbols) {
  Symbol ret;
  for (const auto &s : symbols) {
    ret.heads_.insert(ret.heads_.end(), s.heads_.begin(), s.heads_.end());
  }
  return ret;
}

Symbol Symbol::CreateVariable(const std::string &name) {
  Symbol s;
  s.heads_.push_back(DataEntry(std::make_shared<Node>(nullptr, name), 0));
  return s;
}

void Symbol::ToStaticGraph(StaticGraph *out_graph) const {
  std::vector<Node*> node_order;
  std::unordered_map<Node*, uint32_t> node_index;
  auto &arg_nodes = out_graph->arg_nodes;
  arg_nodes.clear();

  this->DFSVisit([&node_order, &node_index, &arg_nodes](const std::shared_ptr<Node> &n) {
      uint32_t nid = static_cast<uint32_t>(node_index.size());
      node_index[n.get()] = nid;
      if (n->is_variable()) {
        arg_nodes.push_back(nid);
      }
      node_order.push_back(n.get());
    });
  // setup nodes
  out_graph->nodes.resize(node_index.size());
  for (uint32_t nid = 0; nid < node_order.size(); ++nid) {
    if (node_order[nid]->op != nullptr) {
      out_graph->nodes[nid].op.reset(node_order[nid]->op->Copy());
    } else {
      out_graph->nodes[nid].op.reset(nullptr);
    }
    // backward source
    if (node_order[nid]->backward_source_node) {
      out_graph->nodes[nid].backward_source_id =
          node_index[node_order[nid]->backward_source_node.get()];
    } else {
      out_graph->nodes[nid].backward_source_id = -1;
    }
    if (node_order[nid]->attr.get() != nullptr) {
      out_graph->nodes[nid].attr = *(node_order[nid]->attr);
    }
    out_graph->nodes[nid].name = node_order[nid]->name;
    auto &inputs = out_graph->nodes[nid].inputs;
    inputs.clear();
    for (const DataEntry &src : node_order[nid]->inputs) {
      StaticGraph::DataEntry e;
      e.index = src.index;
      e.source_id = node_index[src.source.get()];
      inputs.push_back(e);
    }
  }
  // setup heads
  out_graph->heads.clear();
  for (auto &head : heads_) {
    StaticGraph::DataEntry e;
    e.source_id = node_index[head.source.get()];
    e.index = head.index;
    out_graph->heads.push_back(e);
  }
}

void Symbol::FromStaticGraph(const StaticGraph &graph) {
  std::unordered_map<uint32_t, std::shared_ptr<Node> > nodes;
  std::vector<uint32_t> topo_order = graph.TopoSort();
  // copy ver nodes in topo order
  for (uint32_t nid : topo_order) {
    auto &gnode = graph.nodes[nid];
    auto sym_node = std::make_shared<Symbol::Node>(nullptr, gnode.name);
    if (gnode.op.get() != nullptr) {
      sym_node->op.reset(gnode.op->Copy());
    }
    if (gnode.backward_source_id != -1) {
      sym_node->backward_source_node = nodes.at(gnode.backward_source_id);
    }
    if (gnode.attr.size() != 0) {
      sym_node->attr.reset(new std::map<std::string, std::string>(gnode.attr));
    }
    for (const StaticGraph::DataEntry& e : gnode.inputs) {
      Symbol::DataEntry entry(nodes.at(e.source_id), e.index);
      sym_node->inputs.push_back(std::move(entry));
    }
    nodes[nid] = sym_node;
  }
  // generate the heads
  heads_.clear();
  for (const StaticGraph::DataEntry& e : graph.heads) {
    Symbol::DataEntry entry(nodes.at(e.source_id), e.index);
    heads_.push_back(std::move(entry));
  }
}
}  // namespace mxnet
//===== EXPANDED: ../src/symbol/symbol.cc =====

//===== EXPANDIND: ../src/operator/operator.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief operator module of mxnet
 */

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::OperatorPropertyReg);
}  // namespace dmlc

namespace mxnet {
// implementation of all factory functions
OperatorProperty *OperatorProperty::Create(const char* type_name) {
  auto *creator = dmlc::Registry<OperatorPropertyReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Operator " << type_name << " in registry";
  }
  return creator->body();
}
}  // namespace mxnet
//===== EXPANDED: ../src/operator/operator.cc =====

//===== EXPANDIND: ../src/operator/activation.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
//===== EXPANDIND: ../src/operator/activation-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_


namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp>
class ActivationOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[activation::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[activation::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[activation::kOut], F<ForwardOp>(data));
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> m_out_grad = out_grad[activation::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_out_data = out_data[activation::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_in_grad = in_grad[activation::kData].FlatTo2D<xpu, real_t>(s);
    Assign(m_in_grad, req[activation::kData], F<BackwardOp>(m_out_data) * m_out_grad);
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

  virtual ExecType exec_type() const {
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    return kAsync;
  }
};  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ActivationParam type);

#if DMLC_USE_CXX11
class ActivationProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Activation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[activation::kOut], out_data[activation::kOut], in_data[activation::kData]};
#else
    return {out_grad[activation::kOut], out_data[activation::kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[activation::kOut], in_grad[activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[activation::kData], out_data[activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_
//===== EXPANDED: ../src/operator/activation-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ActivationParam param) {
  switch (param.act_type) {
    case activation::kReLU:
      return new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case activation::kSigmoid:
      return new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case activation::kTanh:
      return new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    case activation::kSoftReLU:
      return new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>();
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe("Apply activation function to input."
          "Softmax Activation is only available with CUDNN on GPU"
          "and will be computed at each location across channel if input is 4D.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/activation.cc =====

//===== EXPANDIND: ../src/operator/batch_norm.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/batch_norm-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_


namespace mxnet {
namespace op {

namespace batchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta};
enum BatchNormOpOutputs {kOut, kMean, kVar, kOutNoAffine};
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace batchnorm

struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
  }
};

template<typename xpu>
class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(BatchNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    const size_t expected_out = this->param_.fix_gamma ? 3 : 4;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), expected_out);
      CHECK_EQ(req.size(), expected_out);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[batchnorm::kData].shape_[1]) /
                         static_cast<real_t>(in_data[batchnorm::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out, out_no_affine;
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                               in_data[batchnorm::kData].shape_[1], 1, 1);
      data = in_data[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[batchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (ctx.is_train && !param_.fix_gamma) {
        out_no_affine = out_data[batchnorm::kOutNoAffine].get_with_shape<xpu, 4, real_t>(dshape,
                                                                                         s);
      }
    } else {
      data = in_data[batchnorm::kData].get<xpu, 4, real_t>(s);
      out = out_data[batchnorm::kOut].get<xpu, 4, real_t>(s);
      if (ctx.is_train && !param_.fix_gamma) {
        out_no_affine = out_data[batchnorm::kOutNoAffine].get<xpu, 4, real_t>(s);
      }
    }
    Tensor<xpu, 1> slope = in_data[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[batchnorm::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, real_t>(s);
    // cal
    if (ctx.is_train) {
      Tensor<xpu, 1> mean = out_data[batchnorm::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[batchnorm::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[batchnorm::kMean] == kNullOp || req[batchnorm::kMean] == kWriteTo);
      CHECK(req[batchnorm::kVar] == kNullOp || req[batchnorm::kVar] == kWriteTo);
      // The first three steps must be enforced.
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
          data - broadcast<1>(mean, data.shape_)));
      if (param_.fix_gamma) {
        Assign(out, req[batchnorm::kOut], (data - broadcast<1>(mean, data.shape_)) /
               F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
               broadcast<1>(bias, out.shape_));
      } else {
        CHECK(req[batchnorm::kOutNoAffine] == kNullOp || req[batchnorm::kOutNoAffine] == kWriteTo);
        out_no_affine = (data - broadcast<1>(mean, data.shape_)) /
                       F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_));
        Assign(out, req[batchnorm::kOut], out_no_affine * broadcast<1>(slope, out.shape_) +
              broadcast<1>(bias, out.shape_));
      }
    } else {
      Assign(out, req[batchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    const size_t expected_out = param_.fix_gamma ? 3 : 4;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), expected_out);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    Tensor<xpu, 4> out_no_affine;
    const real_t scale = static_cast<real_t>(out_grad[batchnorm::kOut].shape_[1]) /
                         static_cast<real_t>(out_grad[batchnorm::kOut].shape_.Size());
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
                               out_grad[batchnorm::kOut].shape_[1], 1, 1);
      data = in_data[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[batchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (!param_.fix_gamma) {
        out_no_affine = out_data[batchnorm::kOutNoAffine].get_with_shape<xpu, 4, real_t>(dshape,
                                                                                         s);
      }
    } else {
      data = in_data[batchnorm::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[batchnorm::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[batchnorm::kData].get<xpu, 4, real_t>(s);
      if (!param_.fix_gamma) {
        out_no_affine = out_data[batchnorm::kOutNoAffine].get<xpu, 4, real_t>(s);
      }
    }

    Tensor<xpu, 1> mean = out_data[batchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[batchnorm::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[batchnorm::kBeta].get<xpu, 1, real_t>(s);
    // get requested temp space
    Tensor<xpu, 2> workspace = ctx.requested[batchnorm::kTempSpace].get_space<xpu>(
        mshadow::Shape2(3, mean.shape_[0]), s);
    Tensor<xpu, 1> gmean = workspace[0];
    Tensor<xpu, 1> gvar = workspace[1];
    Tensor<xpu, 1> tmp = workspace[2];
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
    moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
    // cal
    gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
                                (data - broadcast<1>(mean, data.shape_)) *
                                -0.5f *
                                F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
                                                     -1.5f));
    gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
    gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
    tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
    tmp *= gvar;
    gmean += tmp;
    // assign
    if (!param_.fix_gamma) {
      Assign(gslope, req[batchnorm::kGamma], sumall_except_dim<1>(grad * out_no_affine));
      Assign(grad_in, req[batchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
           broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
           broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                 data.shape_)) +
           broadcast<1>(gmean, data.shape_) * scale);
    } else {
      Assign(grad_in, req[batchnorm::kData], grad *
           broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
           broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                 data.shape_)) +
           broadcast<1>(gmean, data.shape_) * scale);
    }
    Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
  }

 private:
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(BatchNormParam param);


#if DMLC_USE_CXX11
class BatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));
    if (!param_.fix_gamma) {
      out_shape->push_back(dshape);
    }
    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.fix_gamma) {
      return {out_grad[batchnorm::kOut],
              out_data[batchnorm::kMean],
              out_data[batchnorm::kVar],
              in_data[batchnorm::kData],
              in_data[batchnorm::kGamma],
              in_data[batchnorm::kBeta]
            };
    } else {
      return {out_grad[batchnorm::kOut],
              out_data[batchnorm::kOutNoAffine],
              out_data[batchnorm::kMean],
              out_data[batchnorm::kVar],
              in_data[batchnorm::kData],
              in_data[batchnorm::kGamma],
              in_data[batchnorm::kBeta]
            };
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[batchnorm::kOut], in_grad[batchnorm::kData]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return param_.fix_gamma ? 3 : 4;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.fix_gamma) {
      return {"output", "mean", "var"};
    } else {
      return {"output", "mean", "var", "output_no_affine"};
    }
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BatchNormParam param_;
};  // class BatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_
//===== EXPANDED: ../src/operator/batch_norm-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNormParam param) {
  return new BatchNormOp<cpu>(param);
}

Operator *BatchNormProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(BatchNormParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/batch_norm.cc =====

//===== EXPANDIND: ../src/operator/block_grad.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file block_grad.cc
 * \brief
 * \author Bing Xu
*/
//===== EXPANDIND: ../src/operator/block_grad-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file block_grad-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BLOCK_GRAD_INL_H_
#define MXNET_OPERATOR_BLOCK_GRAD_INL_H_

namespace mxnet {
namespace op {

namespace blockgrad {
enum BlockGradientOpInputs {kData};
enum BlockGradientOpOutputs {kOut};
}  // namespace blockgrad

template<typename xpu>
class BlockGradientOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[blockgrad::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[blockgrad::kOut].FlatTo2D<xpu, real_t>(s);
    out = F<mshadow_op::identity>(data);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad = in_grad[blockgrad::kData].FlatTo2D<xpu, real_t>(s);
    grad = 0.f;
  }
};  // class BlockGradientOp

template<typename xpu>
Operator *CreateOp();

#if DMLC_USE_CXX11
class BlockGradientProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(blockgrad::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    return new BlockGradientProp();
  }

  std::string TypeString() const override {
    return "BlockGrad";
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const override {
    return {{in_data[blockgrad::kData], out_data[blockgrad::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;
};  // class BlockGradientProperty

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_BLOCK_GRAD_INL_H_
//===== EXPANDED: ../src/operator/block_grad-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>() {
  return new BlockGradientOp<cpu>();
}

Operator *BlockGradientProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp);
}

MXNET_REGISTER_OP_PROPERTY(BlockGrad, BlockGradientProp)
.describe("Get output from a symbol and pass 0 gradient back")
.add_argument("data", "Symbol", "Input data.");

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/block_grad.cc =====

//===== EXPANDIND: ../src/operator/concat.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/concat-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file concat-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONCAT_INL_H_
#define MXNET_OPERATOR_CONCAT_INL_H_
//===== EXPANDIND: ../src/operator/channel_op_common.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file channel_op_common.h
 * \brief common function used for concat and split channel
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#define MXNET_OPERATOR_CHANNEL_OP_COMMON_H_

namespace mxnet {
namespace op {

template<typename xpu, int dim>
inline void Concatenate(const std::vector<mshadow::Tensor<xpu, dim> > &input,
                        mshadow::Tensor<xpu, dim> *output, const int dimension) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  mshadow::Tensor<xpu, dim> out = *output;
  size_t size = input.size();

  index_t begin = 0;
  switch (dimension) {
    case 0: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(0);
          slice<0>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 1: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(1);
          slice<1>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 2: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(2);
          slice<2>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 3: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(3);
          slice<3>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
  }
}



template<typename xpu, int dim>
void Split(const mshadow::Tensor<xpu, dim> &input,
           std::vector<mshadow::Tensor<xpu, dim> > *output,
           const int dimension,
           std::vector<bool> mask = std::vector<bool>(31, true)) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  std::vector<mshadow::Tensor<xpu, dim> > out = *output;
  size_t size = out.size();
  index_t begin = 0;
  switch (dimension) {
    case 0: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(0);
        if (mask[i]) out[i] = slice<0>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 1: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(1);
        if (mask[i]) out[i] = slice<1>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 2: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(2);
        if (mask[i]) out[i] = slice<2>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 3: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(3);
        if (mask[i]) out[i] = slice<3>(input, begin, end);
        begin = end;
      }
      break;
    }
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
//===== EXPANDED: ../src/operator/channel_op_common.h =====


namespace mxnet {
namespace op {

namespace concat_enum {
enum ConcatOpInputs {kData0, kData1, kData2, kData3, kData4};
enum ConcatOpOutputs {kOut};
}  // namespace concat_enum

struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  int num_args;
  int dim;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(dim).set_range(0,  4).set_default(1)
    .describe("the dimension to be concated.");
  }
};  // struct ConcatParam

template<typename xpu>
class ConcatOp : public Operator {
 public:
  explicit ConcatOp(ConcatParam param)
    : size_(param.num_args), dimension_(param.dim) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req[concat_enum::kOut], kWriteTo);
    CHECK_LT(dimension_, in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > data(size_);
    Tensor<xpu, 4> out;
    if (in_data[concat_enum::kData0].ndim() < 4) {
      uint32_t dim = 0;
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape;
        if (in_data[concat_enum::kData0].ndim() == 2)
          dshape = Shape4(in_data[i].shape_[0], in_data[i].shape_[1], 1, 1);
        else
          dshape = Shape4(in_data[i].shape_[0], in_data[i].shape_[1], in_data[i].shape_[2], 1);
        data[i] = in_data[i].get_with_shape<xpu, 4, real_t>(dshape, s);
        dim += in_data[i].shape_[dimension_];
      }
      Shape<4> dshape_out;
      int a, b, c;
      a = (dimension_ == 0) ? dim : in_data[concat_enum::kData0].shape_[0];
      b = (dimension_ == 1) ? dim : in_data[concat_enum::kData0].shape_[1];
      int dim2 = (in_data[concat_enum::kData0].ndim() == 2) ? 1
                        : in_data[concat_enum::kData0].shape_[2];
      c = (dimension_ == 2) ? dim : dim2;
      dshape_out = Shape4(a, b, c, 1);
      out = out_data[concat_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape_out, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        data[i] = in_data[i].get<xpu, 4, real_t>(s);
      }
      out = out_data[concat_enum::kOut].get<xpu, 4, real_t>(s);
    }
    Concatenate(data, &out, dimension_);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > grad_in(size_);
    Tensor<xpu, 4> grad;
    std::vector<bool> mask(size_, true);
    if (out_grad[concat_enum::kOut].ndim() < 4) {
      uint32_t dim = 0;
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape;
        if (out_grad[concat_enum::kOut].ndim() == 2)
          dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], 1, 1);
        else
          dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], in_grad[i].shape_[2], 1);
        dim += in_grad[i].shape_[dimension_];
        if (req[i] == kNullOp) {
          // Input doesn't need a gradient, don't propagate any
          mask[i] = false;
          // set the dimension so that Split knows how much to advance
          grad_in[i].shape_[dimension_] = dshape[dimension_];
          continue;
        }
        grad_in[i] = in_grad[i].get_with_shape<xpu, 4, real_t>(dshape, s);
        CHECK_EQ(req[i], kWriteTo);
      }
      Shape<4> dshape_out;
      int a, b, c;
      a = (dimension_ == 0) ? dim : in_grad[concat_enum::kData0].shape_[0];
      b = (dimension_ == 1) ? dim : in_grad[concat_enum::kData0].shape_[1];
      int dim2 = (out_grad[concat_enum::kOut].ndim() == 2) ? 1
                      : in_grad[concat_enum::kData0].shape_[2];
      c = (dimension_ == 2) ? dim : dim2;
      dshape_out = Shape4(a, b, c, 1);
      grad = out_grad[concat_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape_out, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        if (req[i] == kNullOp) {
          // Input doesn't need a gradient, don't propagate any
          mask[i] = false;
          // set the dimension so that Split knows how much to advance
          grad_in[i].shape_[dimension_] = in_grad[i].shape_[dimension_];
          continue;
        }
        grad_in[i] = in_grad[i].get<xpu, 4, real_t>(s);
        CHECK_EQ(req[i], kWriteTo);
      }
      grad = out_grad[concat_enum::kOut].get<xpu, 4, real_t>(s);
    }
    Split(grad, &grad_in, dimension_, mask);
  }

 private:
  int size_;
  int dimension_;
};  // class ConcatOp

template<typename xpu>
Operator *CreateOp(ConcatParam param);

#if DMLC_USE_CXX11
class ConcatProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    TShape dshape = in_shape->at(concat_enum::kData0);
    if (dshape.ndim() == 0) return false;
    CHECK_GT(dshape.ndim(), 1);
    CHECK_LT(static_cast<index_t>(param_.dim), dshape.ndim())
        <<"the dimension to be concated is not in the range of input's dimension";
    for (int i = 1; i < param_.num_args; ++i) {
      const TShape &tmp = in_shape->at(i);
      if (tmp.ndim() == 0) return false;
      for (index_t j = 0; j < dshape.ndim(); ++j) {
        if (j == static_cast<index_t>(param_.dim)) {
          dshape[param_.dim] += tmp[param_.dim];
        } else {
          CHECK_EQ(dshape[j], tmp[j])
              << "Incorrect shape[" << i << "]: "
              << tmp << ". "
              << "(first input shape: "
              << dshape << ")";
        }
      }
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ConcatProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Concat";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ConcatParam param_;
};  // class ConcatProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONCAT_INL_H_
//===== EXPANDED: ../src/operator/concat-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param) {
  return new ConcatOp<cpu>(param);
}

Operator* ConcatProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.describe("Perform an feature concat on channel dim (dim 1) over all the inputs.")
.add_argument("data", "Symbol[]", "List of tensors to concatenate")
.add_arguments(ConcatParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/concat.cc =====

//===== EXPANDIND: ../src/operator/convolution.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/convolution-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONVOLUTION_INL_H_



namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs {kData, kWeight, kBias};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
}

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("convolution stride: (y, x)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
    .describe("convolution dilate: (y, x)");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
    .describe("pad for convolution: (y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of groups partition. "
              "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
              "apply convolution and concat instead to achieve the same need.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(128, 4096)
    .describe("Tmp workspace for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class ConvolutionOp : public Operator {
 public:
  explicit ConvolutionOp(ConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[conv::kData].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
    Tensor<xpu, 4> out = out_data[conv::kOut].get<xpu, 4, real_t>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<xpu>(
        Shape1(this->InitTemp(data.shape_, out.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step),
                                        param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid,
                                       gstride * (gid + 1));
        temp_dst[gid] = dot(wmat[gid], tmpc);
      }
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[conv::kBias].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[conv::kData].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
    Tensor<xpu, 4> grad = out_grad[conv::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[conv::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> gwmat = in_grad[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<xpu>(
              Shape1(this->InitTemp(data.shape_, grad.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2> tmp_gwmat = gwmat[gid];
          Assign(tmp_gwmat, req[conv::kWeight], dot(temp_dst[gid], tmpc.T()));
        } else {
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }
      if (req[conv::kData] == kWriteTo || req[conv::kData] == kWriteInplace) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          tmpc = dot(wmat[gid].T(), temp_dst[gid]);
        }
        if (param_.pad[0] == 0 && param_.pad[1] == 0) {
          gdata.Slice(i, i + step) = pack_col2patch(temp_col,
                                     data.Slice(i, i + step).shape_,
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.dilate[0]);
        } else {
          Shape<4> pshape = data.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad[0];
          pshape[3] += 2 * param_.pad[1];
          gdata.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                          pshape,
                                          param_.kernel[0],
                                          param_.kernel[1],
                                          param_.stride[0],
                                          param_.dilate[0]),
                                          gdata[i][0].shape_);
        }
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[conv::kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[conv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     param_.num_filter / param_.num_group,
                                     oshape[2] * oshape[3]);
    // param_.workspace is in elements of sizeof(real_t)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
          static_cast<index_t>(param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
          ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(real_t) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(real_t) << " Bytes";
    return required_size;
  }

  ConvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(ConvolutionParam param);

#if DMLC_USE_CXX11
class ConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[conv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       conv::kWeight,
                       Shape4(param_.num_filter, dshape[1], param_.kernel[0], param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GE(param_.dilate.Size(), 0) \
        << "incorrect dilate size: " << param_.dilate;
    CHECK(ksize_x <= dshape[3] && ksize_y <= dshape[2])
        << "kernel size exceed input";
    (*out_shape)[conv::kOut][1] = param_.num_filter;
    (*out_shape)[conv::kOut][2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] == 1 ? ksize_y : ksize_y * param_.dilate[0] - 1)) / param_.stride[0] + 1;
    (*out_shape)[conv::kOut][3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] == 1 ? ksize_x : ksize_x * param_.dilate[1] - 1)) / param_.stride[1] + 1;
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Convolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData], in_data[conv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
//===== EXPANDED: ../src/operator/convolution-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConvolutionParam param) {
  return new ConvolutionOp<cpu>(param);
}

Operator* ConvolutionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/convolution.cc =====

//===== EXPANDIND: ../src/operator/dropout.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/dropout-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_DROPOUT_INL_H_
#define MXNET_OPERATOR_DROPOUT_INL_H_

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
}  // namespace dropout

namespace mxnet {
namespace op {

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out at training time");
  }
};  // struct DropoutParam

template<typename xpu>
class DropoutOp : public Operator {
 public:
  explicit DropoutOp(DropoutParam param) {
    this->pkeep_ = 1.0f - param.p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[dropout::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[dropout::kOut].FlatTo2D<xpu, real_t>(s);
    if (ctx.is_train) {
      Tensor<xpu, 2> mask = out_data[dropout::kMask].FlatTo2D<xpu, real_t>(s);
      Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
      mask = F<mshadow_op::threshold>(prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_);
      Assign(out, req[dropout::kOut], data * mask);
    } else {
      Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad = out_grad[dropout::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> mask = out_data[dropout::kMask].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> gdata = in_grad[dropout::kData].FlatTo2D<xpu, real_t>(s);
    Assign(gdata, req[dropout::kData], grad * mask);
  }

 private:
  real_t pkeep_;
};  // class DropoutOp


template<typename xpu>
Operator *CreateOp(DropoutParam param);

#if DMLC_USE_CXX11
class DropoutProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DropoutProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Dropout";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout::kOut], out_data[dropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout::kOut], in_grad[dropout::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[dropout::kData], out_data[dropout::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  DropoutParam param_;
};  // class DropoutProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_INL_H_

//===== EXPANDED: ../src/operator/dropout-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param) {
  return new DropoutOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe("Apply dropout to input")
.add_argument("data", "Symbol", "Input data to dropout.")
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


//===== EXPANDED: ../src/operator/dropout.cc =====

//===== EXPANDIND: ../src/operator/elementwise_binary_op.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op.cc
 * \brief elementwise binary operator
*/
//===== EXPANDIND: ../src/operator/elementwise_binary_op-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op-inl.h
 * \brief Elementwise binary operation, plus, minus, mul, div
*/
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_


namespace mxnet {
namespace op {

namespace elembinary {
enum ElementWiseBinaryOpInputs {kLhs, kRhs};
enum ElementWiseBinaryOpOutputs {kOut};
enum ElementWiseBinaryOpType {kPlus, kMinus, kMul, kDiv, kPower, kMaximum, kMinimum};
enum ElementWiseBinaryOpResource { kTempSpace };
}  // elembinary

template<typename Op>
inline elembinary::ElementWiseBinaryOpType GetOpType();

template<typename Op>
inline const char* GetOpTypeString();

template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::plus>() {
  return elembinary::kPlus;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::minus>() {
  return elembinary::kMinus;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::mul>() {
  return elembinary::kMul;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::div>() {
  return elembinary::kDiv;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::power>() {
  return elembinary::kPower;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::maximum>() {
  return elembinary::kMaximum;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::minimum>() {
  return elembinary::kMinimum;
}

template<>
inline const char* GetOpTypeString<mshadow::op::plus>() {
  return "_Plus";
}
template<>
inline const char* GetOpTypeString<mshadow::op::minus>() {
  return "_Minus";
}

template<>
inline const char* GetOpTypeString<mshadow::op::mul>() {
  return "_Mul";
}

template<>
inline const char* GetOpTypeString<mshadow::op::div>() {
  return "_Div";
}

template<>
inline const char* GetOpTypeString<mshadow_op::power>() {
  return "_Power";
}

template<>
inline const char* GetOpTypeString<mshadow_op::maximum>() {
  return "_Maximum";
}

template<>
inline const char* GetOpTypeString<mshadow_op::minimum>() {
  return "_Minimum";
}

template<typename xpu, typename ForwardOp>
class ElementWiseBinaryOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> lhs = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[elembinary::kOut], F<ForwardOp>(lhs, rhs));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    CHECK_EQ(req.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> m_out_grad = out_grad[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> lhs_grad = in_grad[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs_grad = in_grad[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
    switch (GetOpType<ForwardOp>()) {
    case elembinary::kPlus: {
      Assign(lhs_grad, req[elembinary::kLhs], F<mshadow_op::identity>(m_out_grad));
      Assign(rhs_grad, req[elembinary::kRhs], F<mshadow_op::identity>(m_out_grad));
      break;
    }
    case elembinary::kMinus: {
      Assign(lhs_grad, req[elembinary::kLhs], F<mshadow_op::identity>(m_out_grad));
      Assign(rhs_grad, req[elembinary::kRhs], F<mshadow_op::negation>(m_out_grad));
      break;
    }
    case elembinary::kMul: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs], lhs_data * m_out_grad);
      Assign(lhs_grad, req[elembinary::kLhs], rhs_data * m_out_grad);
      break;
    }
    case elembinary::kDiv: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs],
             F<mshadow_op::negation>(m_out_grad * lhs_data) / F<mshadow_op::square>(rhs_data));
      Assign(lhs_grad, req[elembinary::kLhs], m_out_grad / rhs_data);
      break;
    }
    case elembinary::kPower: {
      Tensor<xpu, 2> base_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> exponent_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> m_out_data = out_data[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs],
             F<mshadow_op::log>(base_data) * m_out_data * m_out_grad);
      Assign(lhs_grad, req[elembinary::kLhs],
             exponent_data * F<mshadow_op::power>(base_data, exponent_data - 1) * m_out_grad);
      break;
    }
    case elembinary::kMaximum: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Assign(lhs_grad, req[elembinary::kLhs],
             m_out_grad * F<mshadow_op::maximum_grad>(lhs_data, rhs_data));
      Assign(rhs_grad, req[elembinary::kRhs],
             m_out_grad * F<mshadow_op::minimum_grad>(lhs_data, rhs_data));
      break;
    }
    case elembinary::kMinimum: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Assign(lhs_grad, req[elembinary::kLhs],
             m_out_grad * F<mshadow_op::minimum_grad>(lhs_data, rhs_data));
      Assign(rhs_grad, req[elembinary::kRhs],
             m_out_grad * F<mshadow_op::maximum_grad>(lhs_data, rhs_data));
      break;
    }
    }
  }
};  // class ElementWiseBinaryOp


template<typename xpu>
inline Operator* CreateElementWiseBinaryOp_(elembinary::ElementWiseBinaryOpType type) {
  switch (type) {
  case elembinary::kPlus:
    return new ElementWiseBinaryOp<xpu, mshadow::op::plus>();
  case elembinary::kMinus:
    return new ElementWiseBinaryOp<xpu, mshadow::op::minus>();
  case elembinary::kMul:
    return new ElementWiseBinaryOp<xpu, mshadow::op::mul>();
  case elembinary::kDiv:
    return new ElementWiseBinaryOp<xpu, mshadow::op::div>();
  case elembinary::kPower:
    return new ElementWiseBinaryOp<xpu, mshadow_op::power>();
  case elembinary::kMaximum:
    return new ElementWiseBinaryOp<xpu, mshadow_op::maximum>();
  case elembinary::kMinimum:
    return new ElementWiseBinaryOp<xpu, mshadow_op::minimum>();
  }
  LOG(FATAL) << "uknown op type";
  return NULL;
}

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateElementWiseBinaryOp(elembinary::ElementWiseBinaryOpType type);

#if DMLC_USE_CXX11
template<typename ForwardOp>
class ElementWiseBinaryOpProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    CHECK_EQ(kwargs.size(), 0)
        << TypeString() << " do not take any additional keyword arguments besides lhs and rhs";
  }
  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[lhs, rhs]";
    if (in_shape->at(elembinary::kLhs).ndim() != 0) {
      SHAPE_ASSIGN_CHECK(*in_shape, elembinary::kRhs, in_shape->at(elembinary::kLhs));
    } else if (in_shape->at(elembinary::kRhs).ndim() != 0) {
      in_shape->at(elembinary::kLhs) = in_shape->at(elembinary::kRhs);
    } else {
      return false;
    }
    const TShape &dshape = in_shape->at(elembinary::kLhs);
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  std::vector<std::string> ListArguments() const override {
    return {"lhs", "rhs"};
  }

  OperatorProperty* Copy() const override {
    return new ElementWiseBinaryOpProp<ForwardOp>();
  }

  std::string TypeString() const override {
    return GetOpTypeString<ForwardOp>();
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    switch (GetOpType<ForwardOp>()) {
    case elembinary::kPlus:
    case elembinary::kMinus:
      return {out_grad[elembinary::kOut]};
    case elembinary::kMul:
    case elembinary::kDiv:
    case elembinary::kMaximum:
    case elembinary::kMinimum:
      return {out_grad[elembinary::kOut], in_data[elembinary::kLhs], in_data[elembinary::kRhs]};
    case elembinary::kPower:
      return {out_grad[elembinary::kOut], in_data[elembinary::kLhs], in_data[elembinary::kRhs],
              out_data[elembinary::kOut]};
    }
    LOG(FATAL) << "not reached";
    return {};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    switch (GetOpType<ForwardOp>()) {
    case elembinary::kPlus:
    case elembinary::kMinus:
    case elembinary::kMaximum:
    case elembinary::kMinimum:
      return {};
    case elembinary::kMul:
    case elembinary::kDiv:
    case elembinary::kPower:
      return {{out_grad[elembinary::kOut], in_grad[elembinary::kLhs]}};
    }
    LOG(FATAL) << "not reached";
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[elembinary::kLhs], out_data[elembinary::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
//===== EXPANDED: ../src/operator/elementwise_binary_op-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator* CreateElementWiseBinaryOp<cpu>(elembinary::ElementWiseBinaryOpType type) {
  return CreateElementWiseBinaryOp_<cpu>(type);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
template<typename ForwardOp>
Operator* ElementWiseBinaryOpProp<ForwardOp>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateElementWiseBinaryOp, GetOpType<ForwardOp>());
}

MXNET_REGISTER_OP_PROPERTY(_Plus, ElementWiseBinaryOpProp<mshadow::op::plus>)
.describe("Perform an elementwise plus.");
MXNET_REGISTER_OP_PROPERTY(_Minus, ElementWiseBinaryOpProp<mshadow::op::minus>)
.describe("Perform an elementwise minus.");
MXNET_REGISTER_OP_PROPERTY(_Mul, ElementWiseBinaryOpProp<mshadow::op::mul>)
.describe("Perform an elementwise mul.");
MXNET_REGISTER_OP_PROPERTY(_Div, ElementWiseBinaryOpProp<mshadow::op::div>)
.describe("Perform an elementwise div.");
MXNET_REGISTER_OP_PROPERTY(_Power, ElementWiseBinaryOpProp<mshadow_op::power>)
.describe("Perform an elementwise power.");
MXNET_REGISTER_OP_PROPERTY(_Maximum, ElementWiseBinaryOpProp<mshadow_op::maximum>)
.describe("Perform an elementwise power.");
MXNET_REGISTER_OP_PROPERTY(_Minimum, ElementWiseBinaryOpProp<mshadow_op::minimum>)
.describe("Perform an elementwise power.");

}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/elementwise_binary_op.cc =====

//===== EXPANDIND: ../src/operator/elementwise_sum.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cc
 * \brief elementwise sum operator
*/
//===== EXPANDIND: ../src/operator/elementwise_sum-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file elemementwise_sum-inl.h
 * \brief elementwise sum
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_


namespace mxnet {
namespace op {

namespace elemsum {
enum ElementWiseSumOpInputs {kData0, kData1, kData2, kData3};
enum ElementWiseSumOpOutputs {kOut};
}  // namespace elemsum

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be sumed.");
  }
};

template<typename xpu>
class ElementWiseSumOp : public Operator {
 public:
  explicit ElementWiseSumOp(ElementWiseSumParam param)
    : size_(param.num_args) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    if (req[elemsum::kOut] == kNullOp) return;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[elemsum::kOut].FlatTo2D<xpu, real_t>(s);
    switch (size_) {
      case 2: {
        Tensor<xpu, 2> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1);
        break;
      }
      case 3: {
        Tensor<xpu, 2> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[elemsum::kData2].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1 + in_2);
        break;
      }
      case 4: {
        Tensor<xpu, 2> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[elemsum::kData2].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_3 = in_data[elemsum::kData3].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1 + in_2 + in_3);
        break;
      }
      default: {
        Tensor<xpu, 2> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[elemsum::kOut], F<mshadow_op::identity>(in_0));
        for (int i = 1; i < size_; ++i) {
          out += in_data[i].FlatTo2D<xpu, real_t>(s);
        }
        break;
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> ograd = out_grad[elemsum::kOut].FlatTo2D<xpu, real_t>(s);
    for (int i = 0; i < size_; ++i) {
      if (req[i] == kNullOp || req[i] == kWriteInplace) continue;
      Tensor<xpu, 2> igrad = in_grad[i].FlatTo2D<xpu, real_t>(s);
      Assign(igrad, req[i], F<mshadow_op::identity>(ograd));
    }
  }
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("size_", size_);
    writer->EndObject();
  }
  inline void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("size_", &size_);
    helper.ReadAllFields(reader);
  }

 private:
  int size_;
};  // class ElementWiseSumOp

template<typename xpu>
Operator* CreateOp(ElementWiseSumParam param);

#if DMLC_USE_CXX11
class ElementWiseSumProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }
  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    int sidx = -1;
    for (int i = 0; i < param_.num_args; ++i) {
      if (in_shape->at(i).ndim() != 0) {
        sidx = i;
        break;
      }
    }
    if (sidx == -1) return false;
    for (int i = 0; i < param_.num_args; ++i) {
      if (i != sidx) {
        SHAPE_ASSIGN_CHECK(*in_shape, i, in_shape->at(sidx));
      }
    }
    out_shape->clear();
    out_shape->push_back(in_shape->at(sidx));
    return true;
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ElementWiseSumProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "ElementWiseSum";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[0], in_grad[0]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[0], out_data[0]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ElementWiseSumParam param_;
};  // class ElementWiseSumProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_
//===== EXPANDED: ../src/operator/elementwise_sum-inl.h =====

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ElementWiseSumParam param) {
  return new ElementWiseSumOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ElementWiseSumProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

MXNET_REGISTER_OP_PROPERTY(ElementWiseSum, ElementWiseSumProp)
.describe("Perform an elementwise sum over all the inputs.")
.add_arguments(ElementWiseSumParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/elementwise_sum.cc =====

//===== EXPANDIND: ../src/operator/fully_connected.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
//===== EXPANDIND: ../src/operator/fully_connected-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_



namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) change to only set lower bound
    // add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_range(1, 100000)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    Tensor<xpu, 2> data = in_data[fullc::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[fullc::kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[fullc::kOut].FlatTo2D<xpu, real_t>(s);
    out = dot(data, wmat.T());
    if (!param_.no_bias) {
      Tensor<xpu, 1> bias = in_data[fullc::kBias].get<xpu, 1, real_t>(s);
      out += repmat(bias, data.size(0));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[fullc::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[fullc::kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> grad = out_grad[fullc::kOut].FlatTo2D<xpu, real_t>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2> gwmat = in_grad[fullc::kWeight].get<xpu, 2, real_t>(s);
    Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[fullc::kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }
    // gradient of data
    Tensor<xpu, 2> gdata = in_grad[fullc::kData].FlatTo2D<xpu, real_t>(s);
    Assign(gdata, req[fullc::kData], dot(grad, wmat));
  }

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[fullc::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = 0;
    mshadow::Shape<2> ishape = dshape.FlatTo2D();
    num_input = ishape[1];
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.num_hidden));
    return true;
  }

  OperatorProperty* Copy() const override {
    FullyConnectedProp* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullc::kOut], in_data[fullc::kData], in_data[fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullc::kData], in_grad[fullc::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
//===== EXPANDED: ../src/operator/fully_connected-inl.h =====

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyConnectedParam param) {
  return new FullyConnectedOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* FullyConnectedProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedProp)
.describe("Apply matrix multiplication to input then add a bias.")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/fully_connected.cc =====

//===== EXPANDIND: ../src/operator/leaky_relu.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/leaky_relu-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LEAKY_RELU_INL_H_
#define MXNET_OPERATOR_LEAKY_RELU_INL_H_


namespace mxnet {
namespace op {

namespace leakyrelu {
enum LeakyReLUOpInputs {kData, kGamma};
enum LeakyReLUOpOutputs {kOut, kMask};
enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU, kELU};
enum LeakyReLUOpResource {kRandom};
}  // namespace leakyrelu

struct LeakyReLUParam : public dmlc::Parameter<LeakyReLUParam> {
  // use int for enumeration
  int act_type;
  float slope;
  float lower_bound;
  float upper_bound;
  DMLC_DECLARE_PARAMETER(LeakyReLUParam) {
    DMLC_DECLARE_FIELD(act_type).set_default(leakyrelu::kLeakyReLU)
    .add_enum("rrelu", leakyrelu::kRReLU)
    .add_enum("leaky", leakyrelu::kLeakyReLU)
    .add_enum("prelu", leakyrelu::kPReLU)
    .add_enum("elu", leakyrelu::kELU)
    .describe("Activation function to be applied.");
    DMLC_DECLARE_FIELD(slope).set_default(0.25f)
    .describe("Init slope for the activation. (For leaky and elu only)");
    DMLC_DECLARE_FIELD(lower_bound).set_default(0.125f)
    .describe("Lower bound of random slope. (For rrelu only)");
    DMLC_DECLARE_FIELD(upper_bound).set_default(0.334f)
    .describe("Upper bound of random slope. (For rrelu only)");
  }
};

struct prelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 0.0f : a;
  }
};

template<typename xpu>
class LeakyReLUOp : public Operator {
 public:
  explicit LeakyReLUOp(LeakyReLUParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    Tensor<xpu, 4> mask;
    Tensor<xpu, 1> weight;
    if (in_data[leakyrelu::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[leakyrelu::kData].shape_[0],
                               in_data[leakyrelu::kData].shape_[1], 1, 1);
      data = in_data[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
    } else {
      data = in_data[leakyrelu::kData].get<xpu, 4, real_t>(s);
      out = out_data[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get<xpu, 4, real_t>(s);
      }
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, param_.slope));
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        Assign(out, req[leakyrelu::kOut],
               F<mshadow_op::xelu>(data, broadcast<1>(weight, out.shape_)));
        break;
      }
      case leakyrelu::kRReLU: {
        if (ctx.is_train) {
          Random<xpu>* prnd = ctx.requested[leakyrelu::kRandom].get_random<xpu, real_t>(s);
          mask = prnd->uniform(mask.shape_);
          mask = mask * (param_.upper_bound - param_.lower_bound) + param_.lower_bound;
          Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, mask));
        } else {
          const float slope = (param_.lower_bound + param_.upper_bound) / 2.0f;
          Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, slope));
        }
        break;
      }
      case leakyrelu::kELU: {
        Assign(out, req[leakyrelu::kOut], F<mshadow_op::elu>(data, param_.slope));
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> output;
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> gdata;
    Tensor<xpu, 4> grad;
    Tensor<xpu, 4> mask;
    Tensor<xpu, 1> weight;
    Tensor<xpu, 1> grad_weight;
    if (out_grad[leakyrelu::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[leakyrelu::kOut].shape_[0],
                               out_grad[leakyrelu::kOut].shape_[1], 1, 1);
      grad = out_grad[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      gdata = in_grad[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      output = out_data[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
      if (param_.act_type == leakyrelu::kPReLU) {
        data = in_data[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
    } else {
      grad = out_grad[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      gdata = in_grad[leakyrelu::kData].get<xpu, 4, real_t>(s);
      output = out_data[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get<xpu, 4, real_t>(s);
      }
      if (param_.act_type == leakyrelu::kPReLU) {
        data = in_data[leakyrelu::kData].get<xpu, 4, real_t>(s);
      }
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, param_.slope) * grad);
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        grad_weight = in_grad[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        grad_weight = sumall_except_dim<1>(F<prelu_grad>(data) * grad);
        gdata = F<mshadow_op::xelu_grad>(output, broadcast<1>(weight, data.shape_)) * grad;
        break;
      }
      case leakyrelu::kRReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, mask) * grad);
        break;
      }
      case leakyrelu::kELU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::elu_grad>(output, param_.slope) * grad);
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

 private:
  LeakyReLUParam param_;
};  // class LeakyReLUOp

template<typename xpu>
Operator* CreateOp(LeakyReLUParam type);

#if DMLC_USE_CXX11
class LeakyReLUProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.act_type == leakyrelu::kPReLU) {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, gamma]";
    } else {
      CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    }
    const TShape &dshape = in_shape->at(leakyrelu::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.act_type == leakyrelu::kPReLU) {
      in_shape->at(leakyrelu::kGamma) = TShape(Shape1(dshape[1]));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    if (param_.act_type == leakyrelu::kRReLU) {
      out_shape->push_back(dshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LeakyReLUProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LeakyReLU";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {out_grad[leakyrelu::kOut],
              out_data[leakyrelu::kOut],
              in_data[leakyrelu::kData],
              in_data[leakyrelu::kGamma]};
    } else if (param_.act_type == leakyrelu::kRReLU) {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kMask], out_data[leakyrelu::kOut]};
    } else {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kData]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[leakyrelu::kOut], in_grad[leakyrelu::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {};
    } else {
      return {{in_data[leakyrelu::kData], out_data[leakyrelu::kOut]}};
    }
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {"data", "gamma"};
    } else {
      return {"data"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {"output", "mask"};
    } else {
      return {"output"};
    }
  }

  int NumOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {ResourceRequest::kRandom};
    } else {
      return std::vector<ResourceRequest>();
    }
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  LeakyReLUParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_LEAKY_RELU_INL_H_

//===== EXPANDED: ../src/operator/leaky_relu-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(LeakyReLUParam param) {
  return new LeakyReLUOp<cpu>(param);
}

Operator *LeakyReLUProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(LeakyReLUParam);

MXNET_REGISTER_OP_PROPERTY(LeakyReLU, LeakyReLUProp)
.describe("Apply activation function to input.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(LeakyReLUParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/leaky_relu.cc =====

//===== EXPANDIND: ../src/operator/lrn.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/lrn-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LRN_INL_H_
#define MXNET_OPERATOR_LRN_INL_H_

namespace mxnet {
namespace op {

namespace lrn_enum {
enum LRNInputs {kData};
enum LRNOutputs {kOut, kTmpNorm};
}  // namespace lrn_enum

struct LRNParam : public dmlc::Parameter<LRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(LRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("value of the alpha variance scaling parameter in the normalization formula");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("value of the beta power parameter in the normalization formula");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("value of the k parameter in normalization formula");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct LRNParam

template<typename xpu>
class LocalResponseNormOp : public Operator {
 public:
  explicit LocalResponseNormOp(LRNParam param) {
    param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(xxx): Test with gradient chceker
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    // CHECK_EQ(req.size(), 2);
    CHECK_EQ(param_.nsize % 2, 1) << "LRN only supports odd values for local_size";
    const real_t salpha = param_.alpha / param_.nsize;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[lrn_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[lrn_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp_norm = out_data[lrn_enum::kTmpNorm].get<xpu, 4, real_t>(s);
    tmp_norm = chpool<red::sum>(F<mshadow_op::square>(data) , param_.nsize) * salpha + param_.knorm;
    Assign(out, req[lrn_enum::kOut], data *  F<mshadow_op::power>(tmp_norm, -param_.beta));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    const real_t salpha = param_.alpha / param_.nsize;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[lrn_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp_norm = out_data[lrn_enum::kTmpNorm].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[lrn_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_in = in_grad[lrn_enum::kData].get<xpu, 4, real_t>(s);
    grad_in = grad * F<mshadow_op::power>(tmp_norm, -param_.beta);
    grad_in += (- 2.0f * param_.beta * salpha) *
               chpool<red::sum>(grad * data *
                                F<mshadow_op::power>(tmp_norm, -param_.beta - 1.0f),
                                param_.nsize)  * data;
  }

 private:
  LRNParam param_;
};  // class LocalResponseNormOp

template<typename xpu>
Operator *CreateOp(LRNParam param);

#if DMLC_USE_CXX11
class LocalResponseNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
#if MXNET_USE_CUDNN != 1
    out_shape->push_back(dshape);
#endif
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LocalResponseNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LRN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[lrn_enum::kOut], in_data[lrn_enum::kData], out_data[lrn_enum::kOut]};
#else
    return {out_grad[lrn_enum::kOut], in_data[lrn_enum::kData], out_data[lrn_enum::kTmpNorm]};
#endif
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{out_grad[lrn_enum::kOut], in_grad[lrn_enum::kData]}};
#endif
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return MXNET_USE_CUDNN == 1 ? 1 : 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
#if MXNET_USE_CUDNN == 1
    return {"output"};
#else
    return {"output", "tmp_norm"};
#endif
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  LRNParam param_;
};  // LocalResponseNormProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LRN_INL_H_

//===== EXPANDED: ../src/operator/lrn-inl.h =====

#if MXNET_USE_CUDNN == 1
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(LRNParam param) {
  return new LocalResponseNormOp<cpu>(param);
}

Operator* LocalResponseNormProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(LRNParam);

MXNET_REGISTER_OP_PROPERTY(LRN, LocalResponseNormProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_arguments(LRNParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/lrn.cc =====

//===== EXPANDIND: ../src/operator/pooling.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/
//===== EXPANDIND: ../src/operator/pooling-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_POOLING_INL_H_
#define MXNET_OPERATOR_POOLING_INL_H_


namespace mxnet {
namespace op {

namespace pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};
}  // namespace pool_enum

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    // TODO(bing) change to only set lower bound
    DMLC_DECLARE_FIELD(kernel)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("pooling kernel size: (y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .set_expect_ndim(2).enforce_nonzero()
    .describe("stride: for pooling (y, x)");

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .set_expect_ndim(2)
    .describe("pad for pooling: (y, x)");
  }
};

template<typename xpu, typename Reducer>
class PoolingOp : public Operator {
 public:
  explicit PoolingOp(PoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[pool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[pool_enum::kOut].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    // TODO(bing): dual stride in mshadow
    CHECK_EQ(param_.stride[0], param_.stride[1])
        << "Only same stride is supported now";
    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) {
      Assign(out,
             req[pool_enum::kOut],
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.kernel[0],
                           param_.kernel[1],
                           param_.stride[0]));
    } else if (param_.pool_type == pool_enum::kAvgPooling) {
      Assign(out,
             req[pool_enum::kOut],
             (1.0f / (param_.kernel[0] * param_.kernel[1])) * \
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.kernel[0],
                           param_.kernel[1],
                           param_.stride[0]));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    // TODO(bing): remove pad (0,0)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[pool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[pool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> output_data = out_data[pool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[pool_enum::kData].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]);

    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) {
      Assign(input_grad, req[pool_enum::kData],
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
    } else if (param_.pool_type == pool_enum::kAvgPooling) {
      Assign(input_grad, req[pool_enum::kData],
             (1.0f / param_.kernel[0] / param_.kernel[1]) *\
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
    }
  }

 private:
  PoolingParam param_;
};  // class PoolingOp

template<typename xpu>
Operator* CreateOp(PoolingParam param);


#if DMLC_USE_CXX11
class PoolingProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
                               "Pooling: Input data should be 4D in (batch, channel, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    oshape[2] = std::min(dshape[2] + 2 * param_.pad[0] - param_.kernel[0] + param_.stride[0] - 1,
                         dshape[2] + 2 * param_.pad[0] - 1) / param_.stride[0] + 1;
    oshape[3] = std::min(dshape[3] + 2 * param_.pad[1] - param_.kernel[1] + param_.stride[1] - 1,
                         dshape[3] + 2 * param_.pad[1] - 1) / param_.stride[1] + 1;
    CHECK(oshape[2] > 0 && oshape[3] > 0) << "Pooling: kernel size exceed input";
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingProp *prop_sym = new PoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Pooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData], out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[pool_enum::kData], in_grad[pool_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  PoolingParam param_;
};  // class PoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_

//===== EXPANDED: ../src/operator/pooling-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(PoolingParam param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new PoolingOp<cpu, mshadow::red::maximum>(param);
    case pool_enum::kAvgPooling:
      return new PoolingOp<cpu, mshadow::red::sum>(param);
    case pool_enum::kSumPooling:
      return new PoolingOp<cpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

Operator* PoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/pooling.cc =====

//===== EXPANDIND: ../src/operator/regression_output.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_output.cc
 * \brief regression output operator
*/
//===== EXPANDIND: ../src/operator/regression_output-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_ouput-inl.h
 * \brief Regression output operator.
 */
#ifndef MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
#define MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_


namespace mxnet {
namespace op {

namespace reg_enum {
enum RegressionOutputOpInputs {kData, kLabel};
enum RegressionOutputOutputs {kOut};
enum RegressionOutputType {kLinear, kLogistic, kMAE};
}  // reg_enum

struct RegressionOutputParam : public dmlc::Parameter<RegressionOutputParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(RegressionOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
  };
};

// Special Operator to output regression value in forward
// And get gradient in calculation.
template<typename xpu, typename ForwardOp, typename BackwardOp>
class RegressionOutputOp : public Operator {
 public:
  explicit RegressionOutputOp(RegressionOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "RegressionOutputOp Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "RegressionOutputOp Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[reg_enum::kOut], F<ForwardOp>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    real_t num_output =
      in_data[reg_enum::kLabel].Size()/in_data[reg_enum::kLabel].shape_[0];
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> label = in_data[reg_enum::kLabel]
      .get_with_shape<xpu, 2, real_t>(out.shape_, s);
    Assign(grad, req[reg_enum::kData], param_.grad_scale/num_output*
      F<BackwardOp>(out, reshape(label, grad.shape_)));
  }

 private:
  RegressionOutputParam param_;
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateRegressionOutputOp(reg_enum::RegressionOutputType type,
                                   RegressionOutputParam param);

#if DMLC_USE_CXX11
template<reg_enum::RegressionOutputType type>
class RegressionOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    auto &lshape = (*in_shape)[1];
    if (lshape.ndim() == 0) {
      // special treatment for 1D output, to allow 1D label by default.
      // Think about change convention later
      if (dshape.ndim() == 2 && dshape[1] == 1) {
        lshape = Shape1(dshape[0]);
      } else {
        lshape = dshape;
      }
    } else if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
      std::ostringstream os;
      os << "Shape inconsistent, Provided " <<  '='<< lshape << ','
         << " inferred shape=" << dshape;
      throw ::mxnet::op::InferShapeError(os.str(), 1);
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RegressionOutputProp<type>();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    switch (type) {
      case reg_enum::kLinear: return "LinearRegressionOutput";
      case reg_enum::kLogistic: return "LogisticRegressionOutput";
      case reg_enum::kMAE: return "MAERegressionOutput";
      default: LOG(FATAL) << "unknown type"; return "";
    }
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[reg_enum::kLabel], out_data[reg_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[reg_enum::kOut], in_grad[reg_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[reg_enum::kData], out_data[reg_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  RegressionOutputParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
//===== EXPANDED: ../src/operator/regression_output-inl.h =====


namespace mxnet {
namespace op {

template<>
Operator *CreateRegressionOutputOp<cpu>(reg_enum::RegressionOutputType type,
                                        RegressionOutputParam param) {
  switch (type) {
    case reg_enum::kLinear:
      return new RegressionOutputOp<cpu, mshadow::op::identity, mshadow::op::minus>(param);
    case reg_enum::kLogistic:
      return new RegressionOutputOp<cpu, mshadow_op::sigmoid, mshadow::op::minus>(param);
    case reg_enum::kMAE:
      return new RegressionOutputOp<cpu, mshadow::op::identity, mshadow_op::minus_sign>(param);
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return nullptr;
}

// DO_BIND_DISPATCH comes from operator_common.h
template<reg_enum::RegressionOutputType type>
Operator *RegressionOutputProp<type>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateRegressionOutputOp, type, param_);
}

DMLC_REGISTER_PARAMETER(RegressionOutputParam);

MXNET_REGISTER_OP_PROPERTY(LinearRegressionOutput, RegressionOutputProp<reg_enum::kLinear>)
.describe("Use linear regression for final output, this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(MAERegressionOutput, RegressionOutputProp<reg_enum::kMAE>)
.describe("Use mean absolute error regression for final output, "
          "this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(LogisticRegressionOutput, RegressionOutputProp<reg_enum::kLogistic>)
.describe("Use Logistic regression for final output, this is used on final output of a net.\n"
          "Logistic regression is suitable for binary classification "
          "or probability prediction tasks.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/regression_output.cc =====

//===== EXPANDIND: ../src/operator/reshape.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/reshape-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file reshape-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RESHAPE_INL_H_
#define MXNET_OPERATOR_RESHAPE_INL_H_


namespace mxnet {
namespace op {

namespace reshape_enum {
enum ReshapeOpInputs {kData};
enum ReshapeOpOutputs {kOut};
}  // namespace reshape_enum

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape target_shape;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(target_shape)
    .describe("Target new shape. One and only one dim can be 0, "
              "in which case it will be infered from the rest of dims");
  }
};

template<typename xpu>
class ReshapeOp : public Operator {
 public:
  explicit ReshapeOp(ReshapeParam param) {}  // Do nothing

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (req[reshape_enum::kOut] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[reshape_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[reshape_enum::kOut].FlatTo2D<xpu, real_t>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (data.dptr_ == out.dptr_) return;
    CHECK_EQ(data.shape_.Size(), out.shape_.Size());
    Assign(out, req[reshape_enum::kOut], reshape(data, out.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req.size(), 1);
    if (req[reshape_enum::kData] == kNullOp) return;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad_in = in_grad[reshape_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad_out = out_grad[reshape_enum::kData].FlatTo2D<xpu, real_t>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (grad_out.dptr_ == grad_in.dptr_) return;
    CHECK_EQ(grad_out.shape_.Size(), grad_in.shape_.Size());
    Assign(grad_in, req[reshape_enum::kData], reshape(grad_out, grad_in.shape_));
  }
};  // class ReshapeOp

template<typename xpu>
Operator* CreateOp(ReshapeParam);

#if DMLC_USE_CXX11
class ReshapeProp : public OperatorProperty {
 public:
  ReshapeProp() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(reshape_enum::kData);
    if (dshape.ndim() == 0) return false;
    TShape oshape = param_.target_shape;
    int neg_count = 0;
    index_t neg_idx = 0;
    for (index_t i = 0; i < oshape.ndim(); ++i) {
      if (oshape[i] == 0) {
        neg_count++;
        neg_idx = i;
      }
    }
    if (neg_count == 1) {
      oshape[neg_idx] = 1;
      oshape[neg_idx] = dshape.Size()/oshape.Size();
    }
    CHECK(oshape.Size() == dshape.Size())
        << "Target shape size is different to source. "
        << "Target: " << param_.target_shape.Size()
        << "\nSource: " << dshape.Size();
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ReshapeProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Reshape";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[reshape_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[reshape_enum::kData], out_data[reshape_enum::kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[reshape_enum::kOut], in_grad[reshape_enum::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  ReshapeParam param_;
};  // class ReshapeProp

class FlattenProp : public ReshapeProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    // need to use this on osx
    return std::map<std::string, std::string>();
  }

  std::string TypeString() const override {
    return "Flatten";
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(reshape_enum::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    uint32_t target_dim = 1;
    for (uint32_t i = 1; i < dshape.ndim(); ++i) {
      target_dim *= dshape[i];
    }
    out_shape->push_back(mshadow::Shape2(dshape[0], target_dim));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FlattenProp();
    return ptr;
  }
};  // class FlattenProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RESHAPE_INL_H_
//===== EXPANDED: ../src/operator/reshape-inl.h =====



namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ReshapeParam param) {
  return new ReshapeOp<cpu>(param);
}

Operator* ReshapeProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.describe("Reshape input to target shape")
.add_argument("data", "Symbol", "Input data to  reshape.")
.add_arguments(ReshapeParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.describe("Flatten input")
.add_argument("data", "Symbol", "Input data to  flatten.");
}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/reshape.cc =====

//===== EXPANDIND: ../src/operator/slice_channel.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel.cc
 * \brief
 * \author Bing Xu
*/

//===== EXPANDIND: ../src/operator/slice_channel-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
#define MXNET_OPERATOR_SLICE_CHANNEL_INL_H_


namespace mxnet {
namespace op {

namespace slice_enum {
enum SliceChannelOpInputs {kData};
enum SliceChannelOpOutputs {kOut0, kOut1, kOut2, kOut3, kOut4};
}  // namespace slice_enum

struct SliceChannelParam : public dmlc::Parameter<SliceChannelParam> {
  int num_outputs;
  DMLC_DECLARE_PARAMETER(SliceChannelParam) {
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("Number of outputs to be sliced.");
  }
};  // struct SliceChannelParam

template<typename xpu>
class SliceChannelOp : public Operator {
 public:
  explicit SliceChannelOp(SliceChannelParam param)
    : size_(param.num_outputs) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > outputs(size_);
    Tensor<xpu, 4> data;
    if (in_data[slice_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[slice_enum::kData].shape_[0],
                               in_data[slice_enum::kData].shape_[1], 1, 1);
      data = in_data[slice_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      Shape<4> slice_shape = dshape;
      slice_shape[1] = dshape[1] / size_;
      for (int i = 0; i < size_; ++i) {
        outputs[i] = out_data[i].get_with_shape<xpu, 4, real_t>(slice_shape, s);
      }
    } else {
      data = in_data[slice_enum::kData].get<xpu, 4, real_t>(s);
      for (int i = 0; i < size_; ++i) {
        outputs[i] = out_data[i].get<xpu, 4, real_t>(s);
      }
    }
    Split(data, &outputs, 1);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), static_cast<size_t>(size_));
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > grad_out(size_);
    Tensor<xpu, 4> grad;
    if (out_grad[slice_enum::kOut0].ndim() == 2) {
      Shape<4> slice_shape = Shape4(out_grad[slice_enum::kOut0].shape_[0],
                                    out_grad[slice_enum::kOut0].shape_[1], 1, 1);
      for (int i = 0; i < size_; ++i) {
        grad_out[i] = out_grad[i].get_with_shape<xpu, 4, real_t>(slice_shape, s);
      }
      Shape<4> dshape = slice_shape;
      dshape[1] *= size_;
      grad = in_grad[slice_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        grad_out[i] = out_grad[i].get<xpu, 4, real_t>(s);
      }
      grad = in_grad[slice_enum::kData].get<xpu, 4, real_t>(s);
    }
    Concatenate(grad_out, &grad, 1);
  }

 private:
  int size_;
};  // class SliceChannelOp


template<typename xpu>
Operator *CreateOp(SliceChannelParam param);


#if DMLC_USE_CXX11
class SliceChannelProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_outputs; ++i) {
      ret.push_back(std::string("output") + static_cast<char>('0' + i));
    }
    return ret;
  }

  int NumOutputs() const override {
    return param_.num_outputs;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    TShape dshape = in_shape->at(slice_enum::kData);
    if (dshape.ndim() == 0) return false;
    CHECK_GT(dshape.ndim(), 1);
    CHECK_EQ(dshape[1] % param_.num_outputs, 0)
      << "Channel must be divided by the output number: "
      << dshape[1] << " / " << param_.num_outputs;
    dshape[1] /= param_.num_outputs;
    out_shape->clear();
    for (int i = 0; i < param_.num_outputs; ++i) {
      out_shape->push_back(dshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SliceChannelProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SliceChannel";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  SliceChannelParam param_;
};  // class SliceChannelProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
//===== EXPANDED: ../src/operator/slice_channel-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(SliceChannelParam param) {
  return new SliceChannelOp<cpu>(param);
}

Operator* SliceChannelProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SliceChannelParam);

MXNET_REGISTER_OP_PROPERTY(SliceChannel, SliceChannelProp)
.describe("Slice channel into many outputs with equally divided channel")
.add_arguments(SliceChannelParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/slice_channel.cc =====

//===== EXPANDIND: ../src/operator/softmax_output.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
//===== EXPANDIND: ../src/operator/softmax_output-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_


namespace mxnet {
namespace op {

namespace softmaxout_enum {
enum SoftmaxOutputOpInputs {kData, kLabel};
enum SoftmaxOutputOpOutputs {kOut};
}  // namespace softmaxout_enum

struct SoftmaxOutputParam : public dmlc::Parameter<SoftmaxOutputParam> {
  float grad_scale;
  float ignore_label;
  bool multi_output;
  bool use_ignore;
  DMLC_DECLARE_PARAMETER(SoftmaxOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the ignore_label will not work in backward, and this only"
      "be used when multi_output=true");
    DMLC_DECLARE_FIELD(multi_output).set_default(false)
    .describe("If set to true, for a (n,k,x_1,..,x_n) dimensional"
      "input tensor, softmax will generate n*x_1*...*x_n output, each"
      "has k classes");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contributor"
      "to the backward gradient");
  };
};

template<typename xpu>
class SoftmaxOutputOp : public Operator {
 public:
  explicit SoftmaxOutputOp(SoftmaxOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "SoftmaxOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "SoftmaxOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = in_data[softmaxout_enum::kData].size(0);
      int k = in_data[softmaxout_enum::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[softmaxout_enum::kData].Size()/n/k));
      Tensor<xpu, 3> data = in_data[softmaxout_enum::kData].get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3> out = out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, real_t>(s3, s);
      Softmax(out, data);
    } else {
      Tensor<xpu, 2> data = in_data[softmaxout_enum::kData].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, real_t>(s);
      Softmax(out, data);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = out_data[softmaxout_enum::kOut].size(0);
      int k = out_data[softmaxout_enum::kOut].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[softmaxout_enum::kOut].Size()/n/k));
      Tensor<xpu, 2> label = in_data[softmaxout_enum::kLabel].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 3> out = out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3> grad = in_grad[softmaxout_enum::kData].get_with_shape<xpu, 3, real_t>(s3, s);
      if (param_.use_ignore) {
          SoftmaxGrad(grad, out, label, static_cast<real_t>(param_.ignore_label));
      } else {
          SoftmaxGrad(grad, out, label);
      }
      grad *= param_.grad_scale/s3[2];
    } else {
      Tensor<xpu, 1> label = in_data[softmaxout_enum::kLabel].get<xpu, 1, real_t>(s);
      Tensor<xpu, 2> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> grad = in_grad[softmaxout_enum::kData].FlatTo2D<xpu, real_t>(s);
      SoftmaxGrad(grad, out, label);
      grad *= param_.grad_scale;
    }
  }

 private:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxOutputParam param);

#if DMLC_USE_CXX11
class SoftmaxOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    if (param_.multi_output) {
      SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel,
                         Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]));
    } else {
      SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel, Shape1(dshape[0]));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[softmaxout_enum::kLabel], out_data[softmaxout_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[softmaxout_enum::kOut], in_grad[softmaxout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmaxout_enum::kData], out_data[softmaxout_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputProp

class DeprecatedSoftmaxProp : public SoftmaxOutputProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    LOG(INFO) << "Softmax symbol is renamed to SoftmaxOutput. "
      << "This API will be deprecated in Dec, 2015";
    SoftmaxOutputProp::param_.Init(kwargs);
  }

  std::string TypeString() const override {
    return "Softmax";
  }
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
//===== EXPANDED: ../src/operator/softmax_output-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxOutputParam param) {
  return new SoftmaxOutputOp<cpu>(param);
}

Operator *SoftmaxOutputProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxOutput, SoftmaxOutputProp)
.describe("Perform a softmax transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("label", "Symbol", "Label data.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Softmax, DeprecatedSoftmaxProp)
.describe("DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

//===== EXPANDED: ../src/operator/softmax_output.cc =====

//===== EXPANDIND: ../src/operator/deconvolution.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution.cc
 * \brief
 * \author Wei Wu
*/

//===== EXPANDIND: ../src/operator/deconvolution-inl.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution-inl.h
 * \brief
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_DECONVOLUTION_INL_H_



namespace mxnet {
namespace op {

namespace deconv {
  enum DeconvolutionOpInputs {kData, kWeight, kBias};
  enum DeconvolutionOpOutputs {kOut};
  enum DeconvolutionOpResource {kTempSpace};
}

struct DeconvolutionParam : public dmlc::Parameter<DeconvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(DeconvolutionParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("deconvolution kernel size: (y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("deconvolution stride: (y, x)");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
    .describe("pad for deconvolution: (y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("deconvolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("number of groups partition");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(128, 4096)
    .describe("Tmp workspace for deconvolution (MB)");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
    .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class DeconvolutionOp : public Operator {
 public:
  explicit DeconvolutionOp(DeconvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[deconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[deconv::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[deconv::kOut].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[deconv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[deconv::kTempSpace].get_space<xpu>(
        Shape1(this->InitTemp(out.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(out.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    1, 1);  // Deconvolution only support dilate equals 1
      } else {
        temp_col = unpack_patch2col(pad(out.Slice(i, i + step),
                                        param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    1, 1);  // Deconvolution only support dilate equals 1
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid,
                                       gstride * (gid + 1));
        tmpc = dot(wmat[gid].T(), temp_dst[gid]);
      }
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        out.Slice(i, i + step) = pack_col2patch(temp_col,
                                   out.Slice(i, i + step).shape_,
                                   param_.kernel[0],
                                   param_.kernel[1],
                                   param_.stride[0],
                                   1);  // Deconvolution only support dilate equals 1
      } else {
        Shape<4> pshape = out.Slice(i, i + step).shape_;
        pshape[2] += 2 * param_.pad[0];
        pshape[3] += 2 * param_.pad[1];
        out.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                        pshape,
                                        param_.kernel[0],
                                        param_.kernel[1],
                                        param_.stride[0],
                                        1),  // Deconvolution only support dilate equals 1
                                        out[i][0].shape_);
      }
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[deconv::kBias].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[deconv::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad = out_grad[deconv::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[deconv::kData].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[deconv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
    Tensor<xpu, 3> gwmat = in_grad[deconv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[deconv::kTempSpace].get_space<xpu>(
              Shape1(this->InitTemp(grad.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(grad.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     1, 1);  // Deconvolution only support dilate equals 1
      } else {
        temp_col = unpack_patch2col(pad(grad.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     1, 1);  // Deconvolution only support dilate equals 1
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2> tmp_gwmat = gwmat[gid];
          Assign(tmp_gwmat, req[deconv::kWeight], dot(temp_dst[gid], tmpc.T()));
        } else {
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }
      if (req[deconv::kData] == kWriteTo || req[deconv::kData] == kWriteInplace) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          temp_dst[gid] = dot(wmat[gid], tmpc);
        }
        gdata.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                                    mshadow::Shape4(gdata.shape_[1],
                                                    step,
                                                    gdata.size(2),
                                                    gdata.size(3))));
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[deconv::kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[deconv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     oshape[1] / param_.num_group,
                                     oshape[2] * oshape[3]);
    // See convolution for workspace calculations
    nstep_ = std::max(
        std::min(
            static_cast<index_t>(param_.workspace / shape_colunit_.Size() + shape_dstunit_.Size()),
            ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(real_t) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(real_t);
    return required_size;
  }

  DeconvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class DeconvolutionOp

template<typename xpu>
Operator* CreateOp(DeconvolutionParam param);

#if DMLC_USE_CXX11
class DeconvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[deconv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       deconv::kWeight,
                       Shape4(dshape[1], param_.num_filter, param_.kernel[0], param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    (*out_shape)[deconv::kOut][1] = param_.num_filter;
    (*out_shape)[deconv::kOut][2] = param_.stride[0] * (dshape[2] - 1) +
        ksize_y - 2 * param_.pad[0];
    (*out_shape)[deconv::kOut][3] = param_.stride[1] * (dshape[3] - 1) +
        ksize_x - 2 * param_.pad[1];
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DeconvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Deconvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[deconv::kOut], in_data[deconv::kData], in_data[deconv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  DeconvolutionParam param_;
};  // class DeconvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DECONVOLUTION_INL_H_
//===== EXPANDED: ../src/operator/deconvolution-inl.h =====


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(DeconvolutionParam param) {
  return new DeconvolutionOp<cpu>(param);
}

Operator* DeconvolutionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(DeconvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Deconvolution, DeconvolutionProp)
.add_argument("data", "Symbol", "Input data to the DeconvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(DeconvolutionParam::__FIELDS__())
.describe("Apply deconvolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
//===== EXPANDED: ../src/operator/deconvolution.cc =====

//===== EXPANDIND: ../src/storage/storage.cc =====

/*!
 * Copyright (c) 2015 by Contributors
 */
//===== EXPANDIND: ../src/storage/storage_manager.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file storage_manager.h
 * \brief Storage manager.
 */
#ifndef MXNET_STORAGE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_STORAGE_MANAGER_H_


namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager interface.
 */
class StorageManager {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  virtual void* Alloc(size_t size) = 0;
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   * \param size Size of the storage.
   */
  virtual void Free(void* ptr, size_t size) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~StorageManager() = default;
};  // namespace StorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_STORAGE_MANAGER_H_
//===== EXPANDED: ../src/storage/storage_manager.h =====

//===== EXPANDIND: ../src/storage/naive_storage_manager.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file naive_storage_manager.h
 * \brief Naive storage manager.
 */
#ifndef MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_


namespace mxnet {
namespace storage {

/*!
 * \brief Naive storage manager.
 */
template <class DeviceStorage>
class NaiveStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  NaiveStorageManager() = default;
  /*!
   * \brief Default destructor.
   */
  ~NaiveStorageManager() = default;
  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(NaiveStorageManager);
};  // class NaiveStorageManager

template <class DeviceStorage>
void* NaiveStorageManager<DeviceStorage>::Alloc(size_t size) {
  return DeviceStorage::Alloc(size);
}

template <class DeviceStorage>
void NaiveStorageManager<DeviceStorage>::Free(void* ptr, size_t) {
  DeviceStorage::Free(ptr);
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
//===== EXPANDED: ../src/storage/naive_storage_manager.h =====

//===== EXPANDIND: ../src/storage/pooled_storage_manager.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_


namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager with a memory pool.
 */
template <class DeviceStorage, size_t kThreshold>
class PooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  PooledStorageManager() = default;
  /*!
   * \brief Default destructor.
   */
  ~PooledStorageManager() {
    ReleaseAll();
  }
  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t size) override;

 private:
  void ReleaseAll();
  // internal mutex
  std::mutex mutex_;
  // used memory
  size_t used_memory_ = 0;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(PooledStorageManager);
};  // class PooledStorageManager

template <class DeviceStorage, size_t kThreshold>
void* PooledStorageManager<DeviceStorage, kThreshold>::Alloc(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    if (kThreshold <= used_memory_) {
      ReleaseAll();
    }
    used_memory_ += size;
    return DeviceStorage::Alloc(size);
  } else {
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    return ret;
  }
}

template <class DeviceStorage, size_t kThreshold>
void PooledStorageManager<DeviceStorage, kThreshold>::Free(void* ptr,
                                                           size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(ptr);
}

template <class DeviceStorage, size_t kThreshold>
void PooledStorageManager<DeviceStorage, kThreshold>::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      DeviceStorage::Free(j);
      used_memory_ -= i.first;
    }
  }
  memory_pool_.clear();
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
//===== EXPANDED: ../src/storage/pooled_storage_manager.h =====

//===== EXPANDIND: ../src/storage/cpu_device_storage.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_device_storage.h
 * \brief CPU storage implementation.
 */
#ifndef MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_CPU_DEVICE_STORAGE_H_


namespace mxnet {
namespace storage {

/*!
 * \brief CPU storage implementation.
 */
class CPUDeviceStorage {
 public:
  /*!
   * \brief Aligned allocation on CPU.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);

 private:
  /*!
   * \brief Alignment of allocation.
   */
  static constexpr size_t alignment_ = 16;
};  // class CPUDeviceStorage

inline void* CPUDeviceStorage::Alloc(size_t size) {
#if _MSC_VER
  void* ptr;
  ptr = _aligned_malloc(size, alignment_);
  return CHECK_NOTNULL(ptr);
#else
  void* ptr;
  int ret = posix_memalign(&ptr, alignment_, size);
  CHECK_EQ(ret, 0) << "Allocation failed";
  return ptr;
#endif
}

inline void CPUDeviceStorage::Free(void* ptr) {
#if _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
//===== EXPANDED: ../src/storage/cpu_device_storage.h =====

//===== EXPANDIND: ../src/storage/gpu_device_storage.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

//===== EXPANDIND: ../src/common/cuda_utils.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file cuda_utils.h
 * \brief CUDA debugging utilities.
 */
#ifndef MXNET_COMMON_CUDA_UTILS_H_
#define MXNET_COMMON_CUDA_UTILS_H_


#if MXNET_USE_CUDA


namespace mxnet {
namespace common {
/*! \brief common utils for cuda */
namespace cuda {
/*!
 * \brief Get string representation of cuBLAS errors.
 * \param error The error.
 * \return String representation.
 */
inline const char* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    break;
  }
  return "Unknown cuBLAS status";
}

/*!
 * \brief Get string representation of cuRAND errors.
 * \param status The status.
 * \return String representation.
 */
inline const char* CurandGetErrorString(curandStatus_t status) {
  switch (status) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND status";
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    cudaError_t e = cudaGetLastError();                                      \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  }

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Protected cuBLAS call.
 * \param func Expression to call.
 *
 * It checks for cuBLAS errors after invocation of the expression.
 */
#define CUBLAS_CALL(func)                                       \
  {                                                             \
    cublasStatus_t e = (func);                                  \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS)                          \
        << "cuBLAS: " << common::cuda::CublasGetErrorString(e); \
  }

/*!
 * \brief Protected cuRAND call.
 * \param func Expression to call.
 *
 * It checks for cuRAND errors after invocation of the expression.
 */
#define CURAND_CALL(func)                                       \
  {                                                             \
    curandStatus_t e = (func);                                  \
    CHECK_EQ(e, CURAND_STATUS_SUCCESS)                          \
        << "cuRAND: " << common::cuda::CurandGetErrorString(e); \
  }

#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN


#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

#endif  // MXNET_USE_CUDNN
#endif  // MXNET_COMMON_CUDA_UTILS_H_
//===== EXPANDED: ../src/common/cuda_utils.h =====

#if MXNET_USE_CUDA
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace storage {

/*!
 * \brief GPU storage implementation.
 */
class GPUDeviceStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);
};  // class GPUDeviceStorage

inline void* GPUDeviceStorage::Alloc(size_t size) {
  void* ret = nullptr;
#if MXNET_USE_CUDA
  CUDA_CALL(cudaMalloc(&ret, size));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
  return ret;
}

inline void GPUDeviceStorage::Free(void* ptr) {
#if MXNET_USE_CUDA
  // throw special exception for caller to catch.
  cudaError_t err = cudaFree(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
  }
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
//===== EXPANDED: ../src/storage/gpu_device_storage.h =====

//===== EXPANDIND: ../src/storage/pinned_memory_storage.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_device_storage.h
 * \brief CPU storage with pinned memory
 */
#ifndef MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#define MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_


namespace mxnet {
namespace storage {

class PinnedMemoryStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);

  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);
};

inline void* PinnedMemoryStorage::Alloc(size_t size) {
  void* ret = nullptr;
#if MXNET_USE_CUDA
  // make the memory available across all devices
  CUDA_CALL(cudaHostAlloc(&ret, size, cudaHostAllocPortable));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
  return ret;
}

inline void PinnedMemoryStorage::Free(void* ptr) {
#if MXNET_USE_CUDA
  cudaError_t err = cudaFreeHost(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
  }
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
//===== EXPANDED: ../src/storage/pinned_memory_storage.h =====

//===== EXPANDIND: ../src/common/lazy_alloc_array.h =====

/*!
 * Copyright (c) 2015 by Contributors
 * \file lazy_alloc_array.h
 * \brief An array that lazily allocate elements as
 *   First time the cell get visited.
 */
#ifndef MXNET_COMMON_LAZY_ALLOC_ARRAY_H_
#define MXNET_COMMON_LAZY_ALLOC_ARRAY_H_


namespace mxnet {
namespace common {

template<typename TElem>
class LazyAllocArray {
 public:
  /*!
   * \brief Get element of corresponding index,
   *  if it is not created create by creator
   * \param index the array index position
   * \param creator a lambda function to create new element when needed.
   */
  template<typename FCreate>
  inline TElem* Get(int index, FCreate creator);
  /*!
   * \brief for each not null element of the array, call fvisit
   * \param fvisit a function of (size_t, TElem*)
   */
  template<typename FVisit>
  inline void ForEach(FVisit fvisit);
  /*! \brief clear all the allocated elements in array */
  inline void Clear();

 private:
  /*! \brief the initial size of the array */
  static constexpr std::size_t kInitSize = 16;
  /*! \brief mutex used during creation */
  std::mutex create_mutex_;
  /*! \brief internal data fir initial size */
  std::array<std::unique_ptr<TElem>, kInitSize> head_;
  /*! \brief overflow array of more elements */
  std::vector<std::unique_ptr<TElem> > more_;
};

// implementations
template<typename TElem>
template<typename FCreate>
inline TElem* LazyAllocArray<TElem>::Get(int index, FCreate creator) {
  CHECK_GE(index, 0);
  size_t idx = static_cast<size_t>(index);
  if (idx < kInitSize) {
    TElem *ptr = head_[idx].get();
    if (ptr != nullptr) {
      return ptr;
    } else {
      std::lock_guard<std::mutex> lock(create_mutex_);
      TElem *ptr = head_[idx].get();
      if (ptr != nullptr) return ptr;
      head_[idx].reset(ptr = creator());
      return ptr;
    }
  } else {
    std::lock_guard<std::mutex> lock(create_mutex_);
    idx -= kInitSize;
    if (more_.size() <= idx) more_.resize(idx + 1);
    TElem *ptr = more_[idx].get();
    if (ptr != nullptr) return ptr;
    more_[idx].reset(ptr = creator());
    return ptr;
  }
}

template<typename TElem>
inline void LazyAllocArray<TElem>::Clear() {
  std::lock_guard<std::mutex> lock(create_mutex_);
  for (size_t i = 0; i < head_.size(); ++i) {
    head_[i].reset(nullptr);
  }
  for (size_t i = 0; i < more_.size(); ++i) {
    more_[i].reset(nullptr);
  }
}

template<typename TElem>
template<typename FVisit>
inline void LazyAllocArray<TElem>::ForEach(FVisit fvisit) {
  std::lock_guard<std::mutex> lock(create_mutex_);
  for (size_t i = 0; i < head_.size(); ++i) {
    if (head_[i].get() != nullptr) {
      fvisit(i, head_[i].get());
    }
  }
  for (size_t i = 0; i < more_.size(); ++i) {
    if (more_[i].get() != nullptr) {
      fvisit(i + kInitSize, more_[i].get());
    }
  }
}
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_LAZY_ALLOC_ARRAY_H_
//===== EXPANDED: ../src/common/lazy_alloc_array.h =====


namespace mxnet {

// consider change storage as a pure abstract class
class StorageImpl : public Storage {
 public:
  Handle Alloc(size_t size, Context ctx) override;
  void Free(Handle handle) override;
  StorageImpl() {}
  virtual ~StorageImpl() = default;

 private:
  static constexpr size_t kPoolThreshold = 4096 * 1024 * 1024ul;
  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;
  static constexpr size_t kMaxNumberOfDeviceIDs = Context::kMaxDevID + 1;

  template <class DeviceStorage>
  using CurrentStorageManager =
      storage::PooledStorageManager<DeviceStorage, kPoolThreshold>;

  static void ActivateDevice(Context ctx) {
    switch (ctx.dev_type) {
      case Context::kCPU: break;
      case Context::kGPU:
      case Context::kCPUPinned:
#if MXNET_USE_CUDA
        CUDA_CALL(cudaSetDevice(ctx.dev_id));
#else  // MXNET_USE_CUDA
        LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
        break;
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }
  // internal storage managers
  std::array<common::LazyAllocArray<storage::StorageManager>,
             kMaxNumberOfDevices> storage_managers_;
};  // struct Storage::Impl

Storage::Handle StorageImpl::Alloc(size_t size, Context ctx) {
  // space already recycled, ignore request
  Handle hd;
  hd.ctx = ctx;
  hd.size = size;
  auto&& device = storage_managers_.at(ctx.dev_type);
  storage::StorageManager *manager = device.Get(
      ctx.dev_id, [ctx]() {
        storage::StorageManager *ptr = nullptr;
        switch (ctx.dev_type) {
          case Context::kCPU: {
            ptr = new CurrentStorageManager<storage::CPUDeviceStorage>();
            break;
          }
          case Context::kCPUPinned: {
            ptr = new CurrentStorageManager<storage::PinnedMemoryStorage>();
            break;
          }
          case Context::kGPU: {
            ptr = new CurrentStorageManager<storage::GPUDeviceStorage>();
            break;
          }
          default: LOG(FATAL) <<  "Unimplemented device " << ctx.dev_type;
        }
        return ptr;
      });
  this->ActivateDevice(ctx);
  hd.dptr = manager->Alloc(size);
  return hd;
}

void StorageImpl::Free(Storage::Handle handle) {
  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  storage::StorageManager *maneger = device.Get(
      ctx.dev_id, []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });
  this->ActivateDevice(ctx);
  maneger->Free(handle.dptr, handle.size);
}

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}
}  // namespace mxnet
//===== EXPANDED: ../src/storage/storage.cc =====

//===== EXPANDIND: ../src/common/tblob_op_registry.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file tblob_op_registry.cc
 * Implementation of tblob op registry
 */

namespace mxnet {
namespace common {
class TBlobUnaryOpProp;

class TBlobOpRegEntryImpl : public TBlobOpRegEntry {
 public:
  // functions
  TSelf& set_function(int dev_mask,
                      UnaryFunction funary,
                      bool inplace_in_out,
                      bool register_symbolic) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reg_counter_;
    if (funary_.size() <= static_cast<size_t>(dev_mask)) {
      funary_.resize(dev_mask + 1, nullptr);
    }
    if (funary_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_[dev_mask] = funary;
    inplace_in0_out_forward_ = inplace_in_out;
    if (reg_counter_ == 1) {
      this->RegisterUnary();
      register_symbolic_ = register_symbolic;
      if (register_symbolic) {
        this->RegisterUnarySymbolic();
      }
    }
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradType1 fgrad,
                      bool inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (funary_grad_t1_.size() <= static_cast<size_t>(dev_mask)) {
      funary_grad_t1_.resize(dev_mask + 1, nullptr);
    }
    if (funary_grad_t1_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device gradient function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_grad_t1_[dev_mask] = fgrad;
    inplace_out_in0_grad_ = inplace_out_in_grad;
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradType2 fgrad,
                      bool inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (funary_grad_t2_.size() <= static_cast<size_t>(dev_mask)) {
      funary_grad_t2_.resize(dev_mask + 1, nullptr);
    }
    if (funary_grad_t2_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device gradient function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_grad_t2_[dev_mask] = fgrad;
    inplace_out_in0_grad_ = inplace_out_in_grad;
    return *this;
  }

  TSelf& set_shape_infer(UnaryShapeInfer fshapeinfer) override {
    std::lock_guard<std::mutex> lock(mutex_);
    unary_infer_ = fshapeinfer;
    return *this;
  }

  TSelf& describe(const std::string &description) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reg_counter_ != 1) return *this;
    NDArrayReg().describe(description);
    if (register_symbolic_) {
      OpReg().describe(description);
    }
    return *this;
  }

 private:
  // make friend with unary op
  friend class TBlobUnaryOpProp;
  // internal mutex
  std::mutex mutex_;
  // registration counter
  int reg_counter_{0};
  bool register_symbolic_{true};
  // unary shape inferencer
  UnaryShapeInfer unary_infer_{nullptr};
  // unary functions on each device mask
  std::vector<UnaryFunction> funary_;
  // type 1 gradient function
  std::vector<UnaryGradType1> funary_grad_t1_;
  // type 2 gradient function
  std::vector<UnaryGradType2> funary_grad_t2_;
  // whether do inplace optimization of in 0 and output
  bool inplace_in0_out_forward_{true};
  // whether do inplace optimization of out_grad and in_grad0
  bool inplace_out_in0_grad_{false};
  // NDArray registry
  NDArrayFunctionReg *ndarray_reg_{nullptr};
  OperatorPropertyReg *op_reg_{nullptr};
  // internal function to register NDArray function.
  inline NDArrayFunctionReg &NDArrayReg() {
    if (ndarray_reg_ == nullptr) {
      NDArrayFunctionReg &reg =
          ::dmlc::Registry<NDArrayFunctionReg>::Get()->__REGISTER__(this->name);
      ndarray_reg_ = &reg;
    }
    return *ndarray_reg_;
  }
  // internal function to register NDArray function.
  inline OperatorPropertyReg &OpReg() {
    if (op_reg_ == nullptr) {
      OperatorPropertyReg &reg =
          ::dmlc::Registry<OperatorPropertyReg>::Get()->__REGISTER__(this->name);
      op_reg_ = &reg;
    }
    return *op_reg_;
  }
  // start registering all stuffs
  void RegisterUnary();
  void RegisterUnarySymbolic();
};

// Unary operator to invoke generic TBlob function.
struct TBlobUnaryOperator : public Operator {
  TBlobOpRegEntry::UnaryFunction forward;
  TBlobOpRegEntry::UnaryGradType1 backward1{nullptr};
  TBlobOpRegEntry::UnaryGradType2 backward2{nullptr};

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    TBlob out = out_data[0];
    (*forward)(in_data[0], &out, req[0], ctx.run_ctx);
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad,
                const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    arg::OutGrad ograd; ograd.data = out_grad[0];
    TBlob igrad = in_grad[0];
    if (backward1 != nullptr) {
      arg::OutValue out_value; out_value.data = out_data[0];
      (*backward1)(ograd, out_value, &igrad, req[0], ctx.run_ctx);
    } else if (backward2 != nullptr) {
      arg::Input0 in0; in0.data = in_data[0];
      (*backward2)(ograd, in0, &igrad, req[0], ctx.run_ctx);
    } else {
      LOG(FATAL) << "Backward is not supported";
    }
  }
};  // class UnaryOperator

class TBlobUnaryOpProp : public OperatorProperty {
 public:
  std::string name;
  TBlobOpRegEntryImpl* source;

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    if (source->unary_infer_ == nullptr) {
      out_shape->push_back(dshape);
    } else {
      out_shape->push_back((*(source->unary_infer_))(dshape));
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TBlobUnaryOpProp();
    ptr->source = source;
    ptr->name = name;
    return ptr;
  }

  std::string TypeString() const override {
    return name;
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (source->funary_grad_t1_.size() != 0) {
      return {out_grad[0], out_data[0]};
    } else if (source->funary_grad_t2_.size() != 0) {
      return {out_grad[0], in_data[0]};
    } else {
      LOG(FATAL) << "Backward of " << name << " is not decalred";
      return {};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    if (source->inplace_out_in0_grad_) {
      return {{out_grad[0], in_grad[0]}};
    } else {
      return {};
    }
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (source->inplace_in0_out_forward_) {
      return {{in_data[0], out_data[0]}};
    } else {
      return {};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    size_t dev_mask = ctx.dev_mask();
    TBlobUnaryOperator *op = new TBlobUnaryOperator();
    CHECK(dev_mask < source->funary_.size() && source->funary_[dev_mask] != nullptr);
    op->forward = source->funary_[dev_mask];
    if (dev_mask < source->funary_grad_t1_.size()) {
      op->backward1 = source->funary_grad_t1_[dev_mask];
    }
    if (dev_mask < source->funary_grad_t2_.size()) {
      op->backward2 = source->funary_grad_t2_[dev_mask];
    }
    return op;
  }
};

void TBlobOpRegEntryImpl::RegisterUnary() {
  CHECK_EQ(reg_counter_, 1);
  // The body to be registered
  auto body = [this] (NDArray **used_vars,
                      real_t *s,
                      NDArray **mutate_vars,
                      int num_params,
                      char **param_keys,
                      char **param_vals) {
    NDArray src = *used_vars[0];
    NDArray *out = mutate_vars[0];
    TShape dshape = src.shape();
    if (unary_infer_ != nullptr) dshape = unary_infer_(dshape);

    if (out->is_none()) {
      *out = NDArray(dshape, src.ctx(), true, src.dtype());
    } else {
      CHECK(out->ctx() == src.ctx()) << "target context mismatch";
      CHECK(out->dtype() == src.dtype()) << "target data type mismatch";
      CHECK(out->shape() == dshape) << "target shape mismatch "
      << out->shape() << " vs. " << dshape;
    }
    // important: callback must always capture by value
    NDArray ret = *out;
    // get the const variables
    std::vector<Engine::VarHandle> const_vars;
    if (src.var() != ret.var()) const_vars.push_back(src.var());
    // check if the function exist
    int dev_mask = src.ctx().dev_mask();
    if (static_cast<size_t>(dev_mask) >= funary_.size() ||
        funary_[dev_mask] == nullptr) {
      if (dev_mask == gpu::kDevMask) LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      LOG(FATAL) << "Function " << this->name << "not registered for device " << dev_mask;
    }
    // invoke the function
    UnaryFunction fun = funary_[dev_mask];
    Engine::Get()->PushSync([src, ret, fun, dev_mask](RunContext ctx) {
        ret.CheckAndAlloc();
        TBlob tmp = ret.data();
        (*fun)(src.data(), &tmp, kWriteTo, ctx);
#if MXNET_USE_CUDA
        if (dev_mask == gpu::kDevMask) {
          ctx.get_stream<gpu>()->Wait();
        }
#endif
      }, src.ctx(), const_vars, {ret.var()});
  };
  // register the function.
  NDArrayReg()
      .set_body(body)
      .set_num_use_vars(1)
      .set_num_mutate_vars(1)
      .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
      .add_argument("src", "NDArray", "Source input to the function");
}

void TBlobOpRegEntryImpl::RegisterUnarySymbolic() {
  // register the operator
  auto op_factory = [this]() {
    TBlobUnaryOpProp *prop = new TBlobUnaryOpProp();
    prop->name = this->name;
    prop->source = this;
    return prop;
  };
  OpReg()
      .set_body(op_factory)
      .add_argument("src", "Symbol", "Source symbolic input to the function");
}
TBlobOpRegEntry& TBlobOpRegistry::__REGISTER_OR_FIND__(const std::string &name) {
  if (fmap_.count(name) != 0) return *fmap_.at(name);
  TBlobOpRegEntry *e = new TBlobOpRegEntryImpl();
  e->name = name;
  fmap_[name] = e;
  return *e;
}

TBlobOpRegistry* TBlobOpRegistry::Get() {
  static TBlobOpRegistry inst;
  return &inst;
}

TBlobOpRegistry::~TBlobOpRegistry() {
  for (auto kv : fmap_) {
    delete kv.second;
  }
}
}  // namespace common
}  // namespace mxnet
//===== EXPANDED: ../src/common/tblob_op_registry.cc =====


//===== EXPANDIND: ../src/resource.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.cc
 * \brief Implementation of resource manager.
 */

namespace mxnet {
namespace resource {

// implements resource manager
class ResourceManagerImpl : public ResourceManager {
 public:
  ResourceManagerImpl() noexcept(false)
      : global_seed_(0) {
    cpu_temp_space_copy_ = dmlc::GetEnv("MXNET_CPU_TEMP_COPY", 16);
    gpu_temp_space_copy_ = dmlc::GetEnv("MXNET_GPU_TEMP_COPY", 4);
    engine_ref_ = Engine::_GetSharedRef();
    cpu_rand_.reset(new ResourceRandom<cpu>(
        Context::CPU(), global_seed_));
    cpu_space_.reset(new ResourceTempSpace<cpu>(
        Context::CPU(), cpu_temp_space_copy_));
  }
  ~ResourceManagerImpl() {
    // need explicit delete, before engine get killed
    cpu_rand_.reset(nullptr);
    cpu_space_.reset(nullptr);
#if MXNET_USE_CUDA
    gpu_rand_.Clear();
    gpu_space_.Clear();
#endif
    if (engine_ref_ != nullptr) {
      // release the reference to engine.
      engine_ref_ = nullptr;
    }
  }

  // request resources
  Resource Request(Context ctx, const ResourceRequest &req) override {
    if (ctx.dev_mask() == cpu::kDevMask) {
      switch (req.type) {
        case ResourceRequest::kRandom: return cpu_rand_->resource;
        case ResourceRequest::kTempSpace: return cpu_space_->GetNext();
        default: LOG(FATAL) << "Unknown supported type " << req.type;
      }
    } else {
      CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
#if MSHADOW_USE_CUDA
      switch (req.type) {
        case ResourceRequest::kRandom: {
          return gpu_rand_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceRandom<gpu>(ctx, global_seed_);
            })->resource;
        }
        case ResourceRequest::kTempSpace: {
          return gpu_space_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceTempSpace<gpu>(ctx, gpu_temp_space_copy_);
            })->GetNext();
        }
        default: LOG(FATAL) << "Unknown supported type " << req.type;
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    Resource ret;
    return ret;
  }

  void SeedRandom(uint32_t seed) override {
    global_seed_ = seed;
    cpu_rand_->Seed(global_seed_);
#if MXNET_USE_CUDA
    gpu_rand_.ForEach([seed](size_t i, ResourceRandom<gpu> *p) {
        p->Seed(seed);
      });
#endif
  }

 private:
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGPUs = 16;
  /*! \brief Random number magic number to seed different random numbers */
  static constexpr uint32_t kRandMagic = 127UL;
  // the random number resources
  template<typename xpu>
  struct ResourceRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointer to PRNG */
    mshadow::Random<xpu> *prnd;
    /*! \brief resource representation */
    Resource resource;
    /*! \brief constructor */
    explicit ResourceRandom(Context ctx, uint32_t global_seed)
        : ctx(ctx) {
      mshadow::SetDevice<xpu>(ctx.dev_id);
      resource.var = Engine::Get()->NewVariable();
      prnd = new mshadow::Random<xpu>(ctx.dev_id + global_seed * kRandMagic);
      resource.ptr_ = prnd;
      resource.req = ResourceRequest(ResourceRequest::kRandom);
    }
    ~ResourceRandom() {
      mshadow::Random<xpu> *r = prnd;
      Engine::Get()->DeleteVariable(
          [r](RunContext rctx) {
            MSHADOW_CATCH_ERROR(delete r);
          }, ctx, resource.var);
    }
    // set seed to a PRNG
    inline void Seed(uint32_t global_seed) {
      uint32_t seed = ctx.dev_id + global_seed * kRandMagic;
      mshadow::Random<xpu> *r = prnd;
      Engine::Get()->PushSync([r, seed](RunContext rctx) {
          r->set_stream(rctx.get_stream<xpu>());
          r->Seed(seed);
        }, ctx, {}, {resource.var});
    }
  };
  // temporal space resource.
  template<typename xpu>
  struct ResourceTempSpace {
    /*! \brief the context of the device */
    Context ctx;
    /*! \brief the underlying space */
    std::vector<mshadow::TensorContainer<xpu, 1, real_t>*> space;
    /*! \brief resource representation */
    std::vector<Resource> resource;
    /*! \brief current pointer to the round roubin alloator */
    std::atomic<size_t> curr_ptr;
    /*! \brief constructor */
    explicit ResourceTempSpace(Context ctx, size_t ncopy)
        : ctx(ctx), space(ncopy), resource(ncopy), curr_ptr(0) {
      mshadow::SetDevice<xpu>(ctx.dev_id);
      for (size_t i = 0; i < space.size(); ++i) {
        space[i] = new mshadow::TensorContainer<xpu, 1, real_t>();
        resource[i].var = Engine::Get()->NewVariable();
        resource[i].id = static_cast<int32_t>(i);
        resource[i].ptr_ = space[i];
        resource[i].req = ResourceRequest(ResourceRequest::kTempSpace);
      }
    }
    ~ResourceTempSpace() {
      for (size_t i = 0; i < space.size(); ++i) {
        mshadow::TensorContainer<xpu, 1, real_t>* r = space[i];
        Engine::Get()->DeleteVariable(
            [r](RunContext rctx){
              MSHADOW_CATCH_ERROR(r->Release());
              delete r;
            }, ctx, resource[i].var);
      }
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr = ++curr_ptr;
      // reset ptr to avoid undefined behavior during overflow
      // usually this won't happen
      if (ptr > kMaxDigit) {
        curr_ptr.store((ptr + 1) % space.size());
      }
      return resource[ptr % space.size()];
    }
  };
  /*! \brief number of copies in CPU temp space */
  int cpu_temp_space_copy_;
  /*! \brief number of copies in GPU temp space */
  int gpu_temp_space_copy_;
  /*! \brief Reference to the engine */
  std::shared_ptr<Engine> engine_ref_;
  /*! \brief internal seed to the random number generator */
  uint32_t global_seed_;
  /*! \brief CPU random number resources */
  std::unique_ptr<ResourceRandom<cpu> > cpu_rand_;
  /*! \brief CPU temp space resources */
  std::unique_ptr<ResourceTempSpace<cpu> > cpu_space_;
#if MXNET_USE_CUDA
  /*! \brief random number generator for GPU */
  common::LazyAllocArray<ResourceRandom<gpu> > gpu_rand_;
  /*! \brief temp space for GPU */
  common::LazyAllocArray<ResourceTempSpace<gpu> > gpu_space_;
#endif
};
}  // namespace resource

ResourceManager* ResourceManager::Get() {
  static resource::ResourceManagerImpl inst;
  return &inst;
}
}  // namespace mxnet
//===== EXPANDED: ../src/resource.cc =====


//===== EXPANDIND: ../src/c_api/c_predict_api.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_predict_api.cc
 * \brief C predict API of mxnet
 */
//===== EXPANDIND: ../dmlc-core/include/dmlc/memory_io.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file memory_io.h
 * \brief defines binary serialization class to serialize things into/from memory region.
 */
#ifndef DMLC_MEMORY_IO_H_
#define DMLC_MEMORY_IO_H_


namespace dmlc {
/*!
 * \brief A Stream that operates on fixed region of memory
 *  This class allows us to read/write from/to a fixed memory region.
 */
struct MemoryFixedSizeStream : public SeekStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the head pointer of the memory region.
   * \param buffer_size the size of the memorybuffer
   */
  MemoryFixedSizeStream(void *p_buffer, size_t buffer_size)
      : p_buffer_(reinterpret_cast<char*>(p_buffer)),
        buffer_size_(buffer_size) {
    curr_ptr_ = 0;
  }
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ + size <= buffer_size_);
    size_t nread = std::min(buffer_size_ - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, p_buffer_ + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    CHECK(curr_ptr_ + size <=  buffer_size_);
    std::memcpy(p_buffer_ + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  char *p_buffer_;
  /*! \brief current pointer */
  size_t buffer_size_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryFixedSizeStream

/*!
 * \brief A in memory stream that is backed by std::string.
 *  This class allows us to read/write from/to a std::string.
 */
struct MemoryStringStream : public dmlc::SeekStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the pointer to the string.
   */
  explicit MemoryStringStream(std::string *p_buffer)
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ <= p_buffer_->length());
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    std::memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryStringStream
}  // namespace dmlc
#endif  // DMLC_MEMORY_IO_H_
//===== EXPANDED: ../dmlc-core/include/dmlc/memory_io.h =====

//===== EXPANDIND: ../include/mxnet/c_predict_api.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_predict_api.h
 * \brief C predict API of mxnet, contains a minimum API to run prediction.
 *  This file is self-contained, and do not dependent on any other files.
 */
#ifndef MXNET_C_PREDICT_API_H_
#define MXNET_C_PREDICT_API_H_

#ifdef __cplusplus
#define MXNET_EXTERN_C extern "C"
#else
#define MXNET_EXTERN_C
#endif

#ifdef _WIN32
#ifdef MXNET_EXPORTS
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllexport)
#else
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllimport)
#endif
#else
#define MXNET_DLL MXNET_EXTERN_C
#endif

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief handle to Predictor */
typedef void *PredictorHandle;
/*! \brief handle to NDArray list */
typedef void *NDListHandle;

/*!
 * \brief Get the last error happeneed.
 * \return The last error happened at the predictor.
 */
MXNET_DLL const char* MXGetLastError();

/*!
 * \brief create a predictor
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flatted data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreate(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           mx_uint num_input_nodes,
                           const char** input_keys,
                           const mx_uint* input_shape_indptr,
                           const mx_uint* input_shape_data,
                           PredictorHandle* out);

/*!
 * \brief create a predictor wich customized outputs
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flatted data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param num_output_nodes Number of output nodes to the net,
 * \param output_keys The name of output argument.
 *    For example {"global_pool"}
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */

MXNET_DLL int MXPredCreatePartialOut(const char* symbol_json_str,
                                     const void* param_bytes,
                                     int param_size,
                                     int dev_type, int dev_id,
                                     mx_uint num_input_nodes,
                                     const char** input_keys,
                                     const mx_uint* input_shape_indptr,
                                     const mx_uint* input_shape_data,
                                     mx_uint num_output_nodes,
                                     const char** output_keys,
                                     PredictorHandle* out);
/*!
 * \brief Get the shape of output node.
 *  The returned shape_data and shape_ndim is only valid before next call to MXPred function.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param shape_data Used to hold pointer to the shape data
 * \param shape_ndim Used to hold shape dimension.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetOutputShape(PredictorHandle handle,
                                   mx_uint index,
                                   mx_uint** shape_data,
                                   mx_uint* shape_ndim);
/*!
 * \brief Set the input data of predictor.
 * \param handle The predictor handle.
 * \param key The name of input node to set.
 *     For feedforward net, this is "data".
 * \param data The pointer to the data to be set, with the shape specified in MXPredCreate.
 * \param size The size of data array, used for safety check.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredSetInput(PredictorHandle handle,
                             const char* key,
                             const mx_float* data,
                             mx_uint size);
/*!
 * \brief Run a forward pass to get the output.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredForward(PredictorHandle handle);
/*!
 * \brief Run a interactive forward pass to get the output.
 *  This is helpful for displaying progress of prediction which can be slow.
 *  User must call PartialForward from step=0, keep increasing it until step_left=0.
 * \code
 * int step_left = 1;
 * for (int step = 0; step_left != 0; ++step) {
 *    MXPredPartialForward(handle, step, &step_left);
 *    printf("Current progress [%d/%d]\n", step, step + step_left + 1);
 * }
 * \endcode
 * \param handle The handle of the predictor.
 * \param step The current step to run forward on.
 * \param step_left The number of steps left
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredPartialForward(PredictorHandle handle, int step, int* step_left);
/*!
 * \brief Get the output value of prediction.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param data User allocated data to hold the output.
 * \param size The size of data array, used for safe checking.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetOutput(PredictorHandle handle,
                              mx_uint index,
                              mx_float* data,
                              mx_uint size);
/*!
 * \brief Free a predictor handle.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredFree(PredictorHandle handle);
/*!
 * \brief Create a NDArray List by loading from ndarray file.
 *     This can be used to load mean image file.
 * \param nd_file_bytes The byte contents of nd file to be loaded.
 * \param nd_file_size The size of the nd file to be loaded.
 * \param out The out put NDListHandle
 * \param out_length Length of the list.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListCreate(const char* nd_file_bytes,
                             int nd_file_size,
                             NDListHandle *out,
                             mx_uint* out_length);
/*!
 * \brief Get an element from list
 * \param handle The handle to the NDArray
 * \param index The index in the list
 * \param out_key The output key of the item
 * \param out_data The data region of the item
 * \param out_shape The shape of the item.
 * \param out_ndim The number of dimension in the shape.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListGet(NDListHandle handle,
                          mx_uint index,
                          const char** out_key,
                          const mx_float** out_data,
                          const mx_uint** out_shape,
                          mx_uint* out_ndim);
/*!
 * \brief Free a predictor handle.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListFree(NDListHandle handle);

#endif  // MXNET_C_PREDICT_API_H_
//===== EXPANDED: ../include/mxnet/c_predict_api.h =====

//===== EXPANDIND: ../src/c_api/c_api_error.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef MXNET_C_API_C_API_ERROR_H_
#define MXNET_C_API_C_API_ERROR_H_


/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return MXAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &_except_) { Finalize; return MXAPIHandleException(_except_); } return 0; // NOLINT(*)

/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void MXAPISetLastError(const char* msg);
/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int MXAPIHandleException(const dmlc::Error &e) {
  MXAPISetLastError(e.what());
  return -1;
}
#endif  // MXNET_C_API_C_API_ERROR_H_
//===== EXPANDED: ../src/c_api/c_api_error.h =====


using namespace mxnet;

// predictor interface
struct MXAPIPredictor {
  // output arrays
  std::vector<NDArray> out_arrays;
  // argument arrays
  std::vector<NDArray> arg_arrays;
  // output shapes
  std::vector<TShape> out_shapes;
  // key to arguments
  std::unordered_map<std::string, size_t> key2arg;
  // executor
  std::unique_ptr<Executor> exec;
};

struct MXAPINDList {
  std::vector<std::string> keys;
  std::vector<TShape> shapes;
  std::vector<size_t> indptr;
  std::vector<mx_float> data;
};

int MXPredCreate(const char* symbol_json_str,
                 const void* param_bytes,
                 int param_size,
                 int dev_type, int dev_id,
                 mx_uint num_input_nodes,
                 const char** input_keys,
                 const mx_uint* input_shape_indptr,
                 const mx_uint* input_shape_data,
                PredictorHandle* out) {
  return MXPredCreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      NULL,
      out);
}

int MXPredCreatePartialOut(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           mx_uint num_input_nodes,
                           const char** input_keys,
                           const mx_uint* input_shape_indptr,
                           const mx_uint* input_shape_data,
                           mx_uint num_output_nodes,
                           const char** output_keys,
                           PredictorHandle* out) {
  MXAPIPredictor* ret = new MXAPIPredictor();
  API_BEGIN();
  Symbol sym;
  // load in the symbol.
  {
    std::string json = symbol_json_str;
    std::istringstream is(json);
    dmlc::JSONReader reader(&is);
    sym.Load(&reader);
  }
  // looks likely to output the internal results
  if (num_output_nodes != 0) {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputs();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (mx_uint i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = Symbol::CreateGroup(out_syms);
  }

  // load the parameters
  std::unordered_map<std::string, NDArray> arg_params, aux_params;
  {
    std::unordered_set<std::string> arg_names, aux_names;
    std::vector<std::string> arg_names_vec = sym.ListArguments();
    std::vector<std::string> aux_names_vec = sym.ListAuxiliaryStates();
    for (size_t i = 0; i < arg_names_vec.size(); ++i) {
      arg_names.insert(arg_names_vec[i]);
    }
    for (size_t i = 0; i < aux_names_vec.size(); ++i) {
      aux_names.insert(aux_names_vec[i]);
    }
    std::vector<NDArray> data;
    std::vector<std::string> names;
    dmlc::MemoryFixedSizeStream fi((void*)param_bytes, param_size);  // NOLINT(*)
    NDArray::Load(&fi, &data, &names);
    CHECK_EQ(names.size(), data.size())
        << "Invalid param file format";
    for (size_t i = 0; i < names.size(); ++i) {
      if (!strncmp(names[i].c_str(), "aux:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (aux_names.count(name) != 0) {
          aux_params[name] = data[i];
        }
      }
      if (!strncmp(names[i].c_str(), "arg:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (arg_names.count(name) != 0) {
          arg_params[name] = data[i];
        }
      }
    }
  }

  // shape inference and bind
  std::unordered_map<std::string, TShape> known_shape;
  for (mx_uint i = 0; i < num_input_nodes; ++i) {
    known_shape[std::string(input_keys[i])] =
        TShape(input_shape_data + input_shape_indptr[i],
               input_shape_data + input_shape_indptr[i + 1]);
  }
  std::vector<TShape> arg_shapes;
  std::vector<std::string> arg_names = sym.ListArguments();
  std::vector<std::string> aux_names = sym.ListAuxiliaryStates();
  std::vector<TShape> out_shapes(sym.ListOutputs().size());
  std::vector<TShape> aux_shapes(aux_names.size());
  for (size_t i = 0; i < arg_names.size(); ++i) {
    std::string key = arg_names[i];
    ret->key2arg[key] = i;
    if (known_shape.count(key) != 0) {
      arg_shapes.push_back(known_shape[key]);
    } else {
      arg_shapes.push_back(TShape());
    }
  }
  CHECK(sym.InferShape(&arg_shapes, &out_shapes, &aux_shapes))
      << "The shape information of is not enough to get the shapes";
  ret->out_shapes = out_shapes;
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);

  std::vector<NDArray> arg_arrays, aux_arrays;
  for (size_t i = 0; i < arg_shapes.size(); ++i) {
    NDArray nd = NDArray(arg_shapes[i], ctx);
    if (arg_params.count(arg_names[i]) != 0) {
      CopyFromTo(arg_params[arg_names[i]], &nd);
    }
    arg_arrays.push_back(nd);
  }
  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    NDArray nd = NDArray(aux_shapes[i], ctx);
    if (aux_params.count(aux_names[i]) != 0) {
      CopyFromTo(aux_params[aux_names[i]], &nd);
    }
    aux_arrays.push_back(nd);
  }
  ret->arg_arrays = arg_arrays;
  // bind
  {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store(arg_arrays.size());
    std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
    ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                   arg_arrays,
                                   grad_store, grad_req,
                                   aux_arrays));
    ret->out_arrays = ret->exec->outputs();
  }
  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}

int MXPredGetOutputShape(PredictorHandle handle,
                         mx_uint out_index,
                         mx_uint** shape_data,
                         mx_uint* shape_ndim) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(out_index, p->out_arrays.size())
      << "Index exceed number of outputs";
  *shape_data = p->out_shapes[out_index].data();
  *shape_ndim = p->out_shapes[out_index].ndim();
  API_END();
}

int MXPredSetInput(PredictorHandle handle,
                   const char* key,
                   const mx_float* data,
                   mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    LOG(FATAL) << "cannot find input key " << key;
  }
  NDArray& nd = p->arg_arrays[it->second];
  nd.SyncCopyFromCPU(data, size);
  API_END();
}

int MXPredForward(PredictorHandle handle) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->Forward(false);
  API_END();
}

int MXPredPartialForward(PredictorHandle handle, int step, int* step_left) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->PartialForward(false, step, step_left);
  API_END();
}

int MXPredGetOutput(PredictorHandle handle,
                    mx_uint index,
                    mx_float* data,
                    mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->out_arrays.size())
      << "Output index out of range";
  const NDArray& nd = p->out_arrays[index];
  nd.SyncCopyToCPU(data, size);
  API_END();
}

int MXPredFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPIPredictor*>(handle);
  API_END();
}

int MXNDListCreate(const char* nd_file_bytes,
                   int nd_file_size,
                   NDListHandle *out,
                   mx_uint* out_length) {
  MXAPINDList* ret = new MXAPINDList();
  API_BEGIN();
  std::vector<NDArray> arrays;
  dmlc::MemoryFixedSizeStream fi((void*)nd_file_bytes, nd_file_size);  // NOLINT(*)
  NDArray::Load(&fi,
                &(arrays),
                &(ret->keys));
  if (ret->keys.size() == 0) {
    ret->keys.resize(arrays.size());
  }
  ret->indptr.push_back(0);
  for (size_t i = 0; i < arrays.size(); ++i) {
    TShape shape = arrays[i].shape();
    size_t begin = ret->data.size();
    size_t size = shape.Size();
    ret->shapes.push_back(shape);
    ret->data.resize(begin + size);
    arrays[i].SyncCopyToCPU(dmlc::BeginPtr(ret->data) + begin, size);
    ret->indptr.push_back(begin + size);
  }
  *out = ret;
  *out_length = static_cast<mx_uint>(arrays.size());
  API_END();
}

int MXNDListGet(NDListHandle handle,
                mx_uint index,
                const char** out_key,
                const mx_float** out_data,
                const mx_uint** out_shape,
                mx_uint* out_ndim) {
  MXAPINDList* p = static_cast<MXAPINDList*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->shapes.size())
      << "Index out of range";
  *out_key = p->keys[index].c_str();
  *out_data = dmlc::BeginPtr(p->data) + p->indptr[index];
  *out_shape = p->shapes[index].data();
  *out_ndim = p->shapes[index].ndim();
  API_END();
}

int MXNDListFree(NDListHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPINDList*>(handle);
  API_END();
}
//===== EXPANDED: ../src/c_api/c_predict_api.cc =====

//===== EXPANDIND: ../src/c_api/c_api_error.cc =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
//===== EXPANDIND: ../src/common/thread_local.h =====

/*!
 *  Copyright (c) 2015 by Contributors
 * \file thread_local.h
 * \brief Common utility for thread local storage.
 */
#ifndef MXNET_COMMON_THREAD_LOCAL_H_
#define MXNET_COMMON_THREAD_LOCAL_H_


namespace mxnet {
namespace common {
    
#define MX_TREAD_LOCAL __declspec(thread)
#ifndef MX_TREAD_LOCAL
#message("Warning: Threadlocal is not enabled");
#endif

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
template<typename T>
class ThreadLocalStore {
 public:
  /*! \return get a thread local singleton */
  static T* Get() {
    static MX_TREAD_LOCAL T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }

 private:
  /*! \brief constructor */
  ThreadLocalStore() {}
  /*! \brief destructor */
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static ThreadLocalStore<T> *Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T *str) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
  }
  /*! \brief internal mutex */
  std::mutex mutex_;
  /*!\brief internal data */
  std::vector<T*> data_;
};
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_THREAD_LOCAL_H_
//===== EXPANDED: ../src/common/thread_local.h =====


struct ErrorEntry {
  std::string last_error;
};

typedef mxnet::common::ThreadLocalStore<ErrorEntry> MXAPIErrorStore;

const char *MXGetLastError() {
  return MXAPIErrorStore::Get()->last_error.c_str();
}

void MXAPISetLastError(const char* msg) {
  MXAPIErrorStore::Get()->last_error = msg;
}
//===== EXPANDED: ../src/c_api/c_api_error.cc =====

//===== EXPANDED: mxnet_predict0.cc =====


