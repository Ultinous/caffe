#pragma once

#include "focal_loss_base.hpp"
#include "kl_divergence_loss_layer.hpp"

namespace caffe
{

template<typename Dtype>
using FKLDivergenceLossLayerBase =
    FocalLossBase
    <
      Dtype,
      KLDivergence,
      SquaredDiff
    >;

template <typename Dtype>
class FKLDivergenceLossLayer : public FKLDivergenceLossLayerBase<Dtype>
{
  using base = FKLDivergenceLossLayerBase<Dtype>;
public:
  using base::base;

  virtual inline const char* type() const { return "FKLDivergenceLoss"; }
};

} // namespace caffe
