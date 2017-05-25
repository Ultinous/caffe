#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe
{

namespace ultinous {

template<typename Dtype>
class ProposalTargetLayer : public Layer<Dtype> {
public:
  explicit ProposalTargetLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "ProposalTargetLayer"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);


protected:
  int m_num_classes;
  int m_minibatch_size;
  Dtype m_fg_fraction;
  Dtype m_fg_treshold;
  Dtype m_bg_treshold_lo;
  Dtype m_bg_treshold_hi;
  bool m_bbox_normalize_targets_precomputed;
  std::vector<Dtype> m_bbox_inside_weights /* = { 1, 1, 1, 1 }*/;
  std::vector<Dtype> m_bbox_normalize_means/* = { 0, 0, 0, 0 }*/;
  std::vector<Dtype> m_bbox_normalize_stds /*=  {0.1, 0.1, 0.2, 0.2 }*/;

  Blob<Dtype> m_maximum_temp;
  Blob<int> m_keep_inds;

};

}//ultinous

}//caffe

#endif

