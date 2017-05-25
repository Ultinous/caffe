#pragma once

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe
{
namespace ultinous
{

template <typename Dtype>
class CascadeTargetLayer : public Layer<Dtype>
{
public:
  explicit CascadeTargetLayer(const LayerParameter &param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "CascadeTargetLayer"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

protected:
  int m_minibatch_size;
  Dtype m_fg_fraction;
  Dtype m_fg_treshold;
  Dtype m_bg_treshold_hi;
  Dtype m_bg_treshold_lo;

  Blob<Dtype> m_maximum;
  Blob<int> m_keep_inds;

};

}//ultinous
}//caffe