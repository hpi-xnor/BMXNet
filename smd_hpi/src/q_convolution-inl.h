/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized CONV operator
 * \author HPI-DeepLearning
*/
#ifndef MXNET_OPERATOR_Q_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_Q_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../src/operator/operator_common.h"
#include "../../src/operator/mshadow_op.h"
#include "./q_helper.h"
#include "./xnor_cpu.h"

#include <csignal>

namespace mxnet {
namespace op {

namespace q_conv {
enum QConvolutionOpInputs {kData, kWeight, kBias};
enum QConvolutionOpOutputs {kOut};
enum QConvolutionOpResource {kTempSpace};
enum QConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct QConvolutionParam : public dmlc::Parameter<QConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  // mf quantization and binarization variables
  uint32_t act_bit;
  bool scaling_factor;
  bool binarized_weights_only;
  DMLC_DECLARE_PARAMETER(QConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("convolution stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for convolution: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions. Equivalent to slicing input into num_group\n    "
              "partitions, apply convolution on each, then concatenate the results");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum tmp workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", q_conv::kOff)
    .add_enum("limited_workspace", q_conv::kLimited)
    .add_enum("fastest", q_conv::kFastest)
    .set_default(dmlc::optional<int>())
    .describe("Whether to pick convolution algo by running performance test.\n    "
              "Leads to higher startup time but may give faster speed. Options are:\n    "
              "\'off\': no tuning\n    "
              "\'limited_workspace\': run test and pick the fastest algorithm "
              "that doesn't exceed workspace limit.\n    "
              "\'fastest\': pick the fastest algorithm and ignore workspace limit.\n    "
              "If set to None (default), behavior is determined by environment\n    "
              "variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,\n    "
              "1 for limited workspace (default), 2 for fastest.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCHW for 2d and NCDHW for 3d.");
    DMLC_DECLARE_FIELD(act_bit).set_default(32).set_range(1, 32)
            .describe("Number of bits to quantize weights to.");
    DMLC_DECLARE_FIELD(scaling_factor).set_default(false)
            .describe("Enable alpha and beta scaling factors.");
    DMLC_DECLARE_FIELD(binarized_weights_only).set_default(false)
            .describe("Path to binarized weights. set automatically by model converter.");
  }
};

template<typename xpu, typename DType>
class QConvolutionOp : public Operator {
 public:
  explicit QConvolutionOp(QConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCHW and NCDHW layout";
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    //std::raise(SIGINT);
    CHECK_EQ(req[q_conv::kOut], kWriteTo);
    CHECK(param_.binarized_weights_only ? !ctx.is_train : true);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    }
    Tensor<xpu, 4, DType> data = in_data[q_conv::kData].get<xpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
        in_data[q_conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 4, DType> out = out_data[q_conv::kOut].get<xpu, 4, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // xnor related check
    CHECK_EQ(data.shape_[1] % 32, 0) << "input channel currently have to be multiple of 32 but are: " << data.shape_[1];

    // for training we apply quantization function on weights.
    if(ctx.is_train){
      // mf quantize weights
      Tensor<xpu, 1, DType> w1d = in_data[q_conv::kWeight].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> abs = ctx.requested[q_conv::kTempSpace].get_space_typed<xpu, 1, DType>(w1d.shape_, w1d.stream_);
      helper::quantize(w1d, abs, this->param_.act_bit);
      // /mf quantize weights
    }

    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[q_conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(data.shape_, out.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                               workspace.dptr_ + temp_col.shape_.Size(),
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
        mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
                                       gstride * (gid + 1));
        //=================================================================//
        // For the training in order to make the training easier and faster, 
        // we binarize the input and weights of Qconv layer to +1 and -1, 
        // still apply the standard dot() operator to generate the 
        // gemm result. But for 1-bit prediction we then apply xnor+_popc
        // to generate the same result as the dot() function.
        // this means for prediction phase and 1-bit, the QConvolutionForward(...)
        // should give the exactly same result as the sign( dot() ) method.
        if(!ctx.is_train && this->param_.act_bit == 1){
          CHECK(gid == 0) << "groups not yet supported for pre-binarized weights";
          //xnor based convolution
          QConvolutionForward(wmat_shape[1], // m
                              wmat_shape[2], // n
                              tmpc.size(1), // k
                              data,
                              wmat[gid],
                              tmpc,
                              temp_dst[gid],
                              param_);
        }else{
          temp_dst[gid] = dot(wmat[gid], tmpc);
        }                  
      }

      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[q_conv::kBias].get<xpu, 1, DType>(s);
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
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    }
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[q_conv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[q_conv::kData].get<xpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
        in_data[q_conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 4, DType> grad = out_grad[q_conv::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[q_conv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> gwmat =
        in_grad[q_conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[q_conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(data.shape_, grad.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                               workspace.dptr_ + temp_col.shape_.Size(),
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
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
          Assign(tmp_gwmat, req[q_conv::kWeight], dot(temp_dst[gid], tmpc.T()));
        } else {
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }

      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        tmpc = dot(wmat[gid].T(), temp_dst[gid]);
      }
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        Assign(gdata.Slice(i, i + step), req[q_conv::kData],
               pack_col2patch(temp_col,
                              data.Slice(i, i + step).shape_,
                              param_.kernel[0],
                              param_.kernel[1],
                              param_.stride[0],
                              param_.stride[1],
                              param_.dilate[0],
                              param_.dilate[1]));
      } else {
        Shape<4> pshape = data.Slice(i, i + step).shape_;
        pshape[2] += 2 * param_.pad[0];
        pshape[3] += 2 * param_.pad[1];
        Assign(gdata.Slice(i, i + step), req[q_conv::kData],
               crop(pack_col2patch(temp_col,
                                   pshape,
                                   param_.kernel[0],
                                   param_.kernel[1],
                                   param_.stride[0],
                                   param_.stride[1],
                                   param_.dilate[0],
                                   param_.dilate[1]),
                    gdata[i][0].shape_));
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[q_conv::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[q_conv::kBias], sumall_except_dim<1>(grad));
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
    // param_.workspace is in elements of sizeof(DType)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
            static_cast<index_t>(
                param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
            ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(DType) << " Bytes";
    return required_size;
  }

  QConvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class QConvolutionOp

template<typename xpu>
Operator* CreateOp(QConvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class QConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ((int)param_.kernel.ndim(), 3) << param_.kernel.ndim() << "D convolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      LOG(WARNING) << "convolution with bias untested //mf";
      CHECK_EQ((int)in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ((int)in_shape->size(), 2) << "Input:[data, weight]";
    }
    // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[q_conv::kData];
    if (dshp.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ((int)dshp.ndim(), 4) \
          << "Input data should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                               param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      //std::raise(SIGINT);
      SHAPE_ASSIGN_CHECK(*in_shape, q_conv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, q_conv::kBias, Shape1(param_.num_filter));
      }
      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      CHECK_EQ(dshape[1] % param_.num_group, 0) \
          << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0) \
          << "output num_filter must divide group size";
      CHECK_GT((int)param_.kernel.Size(), 0) \
          << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0) \
          << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0) \
          << "incorrect dilate size: " << param_.dilate;
      CHECK(ksize_y <= dshape[2] + 2 * param_.pad[0]
            && ksize_x <= dshape[3] + 2 * param_.pad[1])
          << "kernel size exceed input";
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
          (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
          (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      return true;
    } else {
      LOG(FATAL) << "Unknown convolution type (only 2d binary convolution supported)";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE((int)in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    CHECK_EQ(sizeof(mxnet::op::xnor_cpu::BINARY_WORD), mshadow::mshadow_sizeof(dtype))
      << "We store our binarized weights in mxnets data structures,"
      << "so we rely on sizeof(BINARY_WORD) == sizeof(DType)";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "QConvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[q_conv::kOut], in_data[q_conv::kData], in_data[q_conv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QConvolutionParam param_;
};  // class QConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_q_CONVOLUTION_INL_H_
