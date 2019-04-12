#pragma once

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <caffe/layer.hpp>
#include <caffe/blob.hpp>
#include <caffe/util/rng.hpp>

namespace caffe
{
namespace ultinous
{
template<typename Dtype>
class CPMTargetLayer : public Layer<Dtype>
{
public:
  explicit CPMTargetLayer(LayerParameter const &param)
    : Layer<Dtype>(param)
    , m_param(param.cpm_target_param())
  {}

  void LayerSetUp(vector<Blob<Dtype>*> const &bottom, vector<Blob<Dtype>*> const &top) override;

  void Reshape(vector<Blob<Dtype>*> const &bottom, vector<Blob<Dtype>*> const &top) override;

  inline const char* type() const override { return "CPMTargetLayer"; }

  inline int ExactNumTopBlobs() const override { return 1; }
  
  inline int MinBottomBlobs() const override { return 2; }
  inline int MaxBottomBlobs() const override { return 3; }

protected:
  void Forward_cpu(vector<Blob<Dtype> *> const &bottom, vector<Blob<Dtype> *> const &top) override;

  void Backward_cpu(vector<Blob<Dtype> *> const &top, vector<bool> const &propagate_down, vector<Blob<Dtype> *> const &bottom) override {}

private:
  CPMTargetParam m_param;
  Dtype m_positive_threshold;
  std::uint32_t m_target_pixel_count;
  Dtype m_foreground_fraction;
  Dtype m_hnm_threshold;
};

} // namespace ultinous
} // namespace caffe
