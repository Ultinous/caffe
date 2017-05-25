#include <vector>
#include <algorithm>
#include <fstream>

#include "glog/logging.h"
#include <boost/concept_check.hpp>

#include "caffe/util/anchors.hpp"

#include "caffe/ultinous/proposal_layer.hpp"

namespace caffe{
namespace ultinous {

template<typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  ProposalParameter proposal_param = this->layer_param_.proposal_param();

  m_feat_stride = proposal_param.feat_stride();
  m_nms_thresh = proposal_param.nms_thresh();
  m_pre_nms_topN = proposal_param.pre_nms_topn();
  m_post_nms_topN = proposal_param.post_nms_topn();
  m_min_size = proposal_param.min_size();
  m_max_size = proposal_param.max_size();
  m_cut_threshold = proposal_param.cut_threshold();
  m_rem_outhanging = proposal_param.rem_outhanging();

  std::vector<std::vector<Dtype> > base_anchors;
  std::vector<Dtype> anchor_scales;
  for (auto s : proposal_param.scales())
    anchor_scales.push_back(s);
  if (!anchor_scales.size())
    anchor_scales = std::vector<Dtype>({8, 16, 32});

  std::vector<Dtype> anchor_ratios;
  for (auto r : proposal_param.ratios())
    anchor_ratios.push_back(r);
  if (!anchor_ratios.size())
    anchor_ratios = std::vector<Dtype>({0.5, 1.0, 2.0});

  base_anchors = generate_anchors(anchor_scales, anchor_ratios, m_feat_stride);

  m_base_anchors.Reshape(std::vector<int>({(int) base_anchors.size(), (int) base_anchors[0].size()}));

  for (std::size_t i = 0; i < base_anchors.size(); ++i)
    for (std::size_t j = 0; j < base_anchors[i].size(); ++j)
      m_base_anchors.mutable_cpu_data()[m_base_anchors.offset(static_cast<int> (i),
                                                              static_cast<int> (j))] = base_anchors[i][j];

  m_num_anchors = m_base_anchors.shape(0);
  CHECK(bottom.size() == 3 || bottom.size() == 4);

  CHECK(top.size() == 2 || top.size() == 1);
  top[0]->Reshape(std::vector<int>({1, 5}));
  if (top.size() == 2)
    top[1]->Reshape(std::vector<int>({1, 1}));
}

template<typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  CHECK(bottom[0]->shape(0) == 1); //TODO rewrite for batches


  CHECK((bottom[1]->shape(1) / 4) == m_num_anchors);
  CHECK((bottom[0]->shape(1) / 2) == m_num_anchors);
  CHECK(bottom[0]->shape(2) == bottom[1]->shape(2));
  CHECK(bottom[0]->shape(3) == bottom[1]->shape(3));

  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  m_proposals.Reshape(std::vector<int>({height * width * m_num_anchors, 4}));
  m_scores.Reshape(std::vector<int>({height * width * m_num_anchors, 1}));
  m_indexes.Reshape(std::vector<int>({height * width * m_num_anchors, 1}));
  top[0]->Reshape(std::vector<int>({1, 5}));
  if (top.size() == 2)
    top[1]->Reshape(std::vector<int>({1, 1}));
}


template<typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  //Not implemented
}


template<typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                        const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {
  // This layer dont need backward 
}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);

REGISTER_LAYER_CLASS(Proposal);

} //ultinous
}//caffe