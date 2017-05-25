#include "caffe/ultinous/cascade_target_layer.hpp"

namespace caffe
{
namespace ultinous
{

template <typename Dtype>
void CascadeTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const CascadeTargetParameter param = this->layer_param().cascade_target_param();
  m_minibatch_size = param.minibatch_size();
  m_fg_fraction = param.fg_fraction();
  m_bg_treshold_hi =  param.bg_treshold_hi();
  m_bg_treshold_lo = param.bg_treshold_lo();
  m_fg_treshold = param.fg_treshold();

  //bottoms proposals, gt_boxes, labels,
  CHECK_EQ(bottom.size(), 3);
  CHECK_EQ(top.size(), 2);

}

template <typename Dtype>
void CascadeTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  int proposals_num =  bottom[0]->shape(0);
  int gt_boxes_num =  bottom[1]->shape(0);
  m_maximum.Reshape(std::vector<int>({proposals_num+gt_boxes_num,2}));

  top[0]->Reshape(std::vector<int>({1, 5}));
  top[1]->Reshape(std::vector<int>({1, 1}));

}

template <typename Dtype>
void CascadeTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

}

template <typename Dtype>
void CascadeTargetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {

}
#ifdef CPU_ONLY
STUB_GPU(CascadeTargetLayer);
#endif

INSTANTIATE_CLASS(CascadeTargetLayer);
REGISTER_LAYER_CLASS(CascadeTarget);

}
}