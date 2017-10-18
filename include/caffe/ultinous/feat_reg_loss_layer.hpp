#pragma once

#include <vector>
#include <deque>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
namespace ultinous {


template <typename Dtype>
class FeatRegLossLayer : public LossLayer<Dtype> {
 public:
  explicit FeatRegLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FeatRegLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
//  virtual inline int MinBottomBlobs() const { return 2; }
//  virtual inline int MaxBottomBlobs() const { return 4; }
  /*virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }*/
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> m_squares, m_ones, m_errors;
  
  std::vector<Dtype> m_batch_norms;
  std::deque<Dtype> m_norms;
  
  Dtype m_min_norm, m_max_norm;
};

}  // namespace ultinous
}  // namespace caffe

