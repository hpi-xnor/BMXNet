/*!
 * Copyright (c) 2016 by Contributors
 * \file q_activation-inl.h
 * \brief Quantized Activation operator
 * \author HPI-DeepLearning
*/
#ifndef MXNET_OPERATOR_Q_ACTIVATION_INL_H_
#define MXNET_OPERATOR_Q_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "../../src/operator/operator_common.h"
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace q_activation {
  enum QActivationOpInputs {kData};
  enum QActivationOpOutputs {kOut};
}  // lowbit_activation

struct QActivationParam : public dmlc::Parameter<QActivationParam> {
  unsigned int act_bit;
  bool backward_only;
  DMLC_DECLARE_PARAMETER(QActivationParam) {
    DMLC_DECLARE_FIELD(act_bit).set_default(1).set_range(1, 32)
    .describe("Quantized activation function.");
    DMLC_DECLARE_FIELD(backward_only).set_default(false)
        .describe("If set 'backward_only' to true, then the quantized activation process"
          "in forward pass will not be performed in this layer, the input data will" 
          "be just copied to output. This setting is created for the combi-use with" 
          "QConv-and QFully-layers, since the quantized activation for input data will" 
          "be done in the forward pass of those two layers.");
  }
};

/**
 * \brief This is the implementation of quantized activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class QActivationOp : public Operator {
 public:
  explicit QActivationOp(QActivationParam param) {
    this->act_bit_ = param.act_bit;
    this->backward_only = param.backward_only;
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
    Tensor<xpu, 2, DType> data = in_data[q_activation::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[q_activation::kOut].FlatTo2D<xpu, DType>(s);
    if(!backward_only){
      if(act_bit_ == 1){
        Assign(out, req[q_activation::kOut], F<mshadow_op::det_sign>(data));
      }else{
        Assign(out, req[q_activation::kOut], F<mshadow_op::quantize>(
              F<mshadow_op::maximum>(F<mshadow_op::minimum>(data, scalar(DType(1))), scalar(DType(0))), //clip to [0, 1]
              scalar(DType(act_bit_))));
      }
    }else
      Assign(out, req[q_activation::kOut],  F<mshadow_op::identity>(data));
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
    Tensor<xpu, 2, DType> m_out_grad = out_grad[q_activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_data = in_data[q_activation::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad[q_activation::kData].FlatTo2D<xpu, DType>(s);
    if(act_bit_ == 1){
      Assign(m_in_grad, req[q_activation::kData], F<mshadow_op::det_sign_grad>(m_in_data) * m_out_grad);
    }else{
      Assign(m_in_grad, req[q_activation::kData], F<mshadow_op::quantize_grad>(m_in_data) * m_out_grad);
    }
  }
  private:
    int act_bit_;
    bool backward_only;
};  // class QActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(QActivationParam param, int dtype);

#if DMLC_USE_CXX11
class QActivationProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(q_activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
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
    auto ptr = new QActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "QActivation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[q_activation::kOut], out_data[q_activation::kOut], in_data[q_activation::kData]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[q_activation::kOut], in_grad[q_activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[q_activation::kData], out_data[q_activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_Q_ACTIVATION_INL_H_
