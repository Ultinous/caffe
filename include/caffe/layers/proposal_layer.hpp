#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe
{

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
      
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);   
  
  virtual inline const char* type() const { return "ProposalLayer"; }

protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual inline std::vector<std::size_t> cpu_base_nms(
  const std::vector<std::size_t>& index_vec
);
  
  inline void bbox_transform_inv();
  inline void clip_boxes(const double& img_w, const double& img_h);
protected:
  int m_feat_stride;
  int m_num_anchors;
  
  std::size_t m_pre_nms_topN;  
  std::size_t m_post_nms_topN; 
  Dtype m_nms_thresh;    
  Dtype m_min_size;      
  
  Blob<Dtype> m_anchors;
  Blob<Dtype> m_proposals;
  
  vector<std::vector<double> > m_base_anchors;
  //vector<Anchor> m_img_anchors;
  //std::vector<std::vector<Dtype>> m_proposals;
  Blob<Dtype> m_scores;
  Blob<int> m_indexes;
};

}//caffe

#endif
