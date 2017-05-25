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
class ProposalLayer : public Layer<Dtype> {
public:
  explicit ProposalLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Proposal"; }

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
  int m_feat_stride;
  int m_num_anchors;

  int m_pre_nms_topN;
  int m_post_nms_topN;
  Dtype m_nms_thresh;

  bool m_rem_outhanging;
  Dtype m_cut_threshold;

  Dtype m_min_size;
  Dtype m_max_size;


  Blob<Dtype> m_proposals;
  Blob<Dtype> m_base_anchors;
  Blob<Dtype> m_scores;
  Blob<int> m_indexes;
};

}

}//caffe

#endif
