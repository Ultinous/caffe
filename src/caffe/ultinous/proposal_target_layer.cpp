#include "glog/logging.h"
#include <boost/concept_check.hpp>


#include "caffe/ultinous/proposal_target_layer.hpp"

namespace caffe{
namespace ultinous {

template<typename Dtype>
void ProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  ProposalTargetParameter proposal_param = this->layer_param_.proposal_target_param();
  m_num_classes = proposal_param.num_classes();
  m_fg_treshold = proposal_param.fg_treshold();
  m_fg_fraction = proposal_param.fg_fraction();
  m_bg_treshold_hi = proposal_param.bg_treshold_hi();
  m_bg_treshold_lo = proposal_param.bg_treshold_lo();
  m_minibatch_size = proposal_param.minibatch_size();
  m_bbox_normalize_targets_precomputed = proposal_param.bbox_normalize_targets_precomputed();
  m_bbox_inside_weights = std::vector<Dtype>({1, 1, 1, 1});
  m_bbox_normalize_means = std::vector<Dtype>({0, 0, 0, 0});
  m_bbox_normalize_stds = std::vector<Dtype>({0.1, 0.1, 0.2, 0.2});


  CHECK(proposal_param.bbox_inside_weights_size() == 4 || proposal_param.bbox_inside_weights_size() == 0);
  CHECK(proposal_param.bbox_normalize_means_size() == 4 || proposal_param.bbox_normalize_means_size() == 0);
  CHECK(proposal_param.bbox_normalize_stds_size() == 4 || proposal_param.bbox_normalize_stds_size() == 0);

  for (int i = 0; i < proposal_param.bbox_inside_weights_size(); ++i)
    m_bbox_inside_weights[i] = proposal_param.bbox_inside_weights(i);
  for (int i = 0; i < proposal_param.bbox_normalize_means_size(); ++i)
    m_bbox_normalize_means[i] = proposal_param.bbox_normalize_means(i);
  for (int i = 0; i < proposal_param.bbox_normalize_stds_size(); ++i)
    m_bbox_normalize_stds[i] = proposal_param.bbox_normalize_stds(i);

  top[0]->Reshape(std::vector<int>({1, 5}));
  top[1]->Reshape(std::vector<int>({1, 1}));
  top[2]->Reshape(std::vector<int>({1, m_num_classes * 4}));
  top[3]->Reshape(std::vector<int>({1, m_num_classes * 4}));
  top[4]->Reshape(std::vector<int>({1, m_num_classes * 4}));
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  int prop_box_num = bottom[0]->shape(0);
  int gt_box_num = bottom[1]->shape(0);

  m_maximum_temp.Reshape(std::vector<int>({gt_box_num + prop_box_num, 2}));
}


template<typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                             const vector<Blob<Dtype> *> &top) {
  //NOT IMPLEMENTED
}


template<typename Dtype>
void ProposalTargetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                              const vector<bool> &propagate_down,
                                              const vector<Blob<Dtype> *> &bottom) {
  // This layer dont need backward 
}

#ifdef CPU_ONLY
STUB_GPU(ProposalTargetLayer);
#endif

INSTANTIATE_CLASS(ProposalTargetLayer);

REGISTER_LAYER_CLASS(ProposalTarget);


}//ultinous
}//caffe