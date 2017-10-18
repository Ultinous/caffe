#include "caffe/ultinous/feat_reg_loss_layer.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void FeatRegLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  FeatRegLossParameter loss_param = this->layer_param_.mol_loc_loss_param();
  m_min_norm = -1;//20.0;
  m_max_norm = -1;//25.0;
}

template <typename Dtype>
void FeatRegLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  m_squares.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  m_errors.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  m_ones.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    m_ones.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void FeatRegLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FeatRegLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FeatRegLossLayer);
#endif

INSTANTIATE_CLASS(FeatRegLossLayer);
REGISTER_LAYER_CLASS(FeatRegLoss);

}  // namespace ultinous
}  // namespace caffe
