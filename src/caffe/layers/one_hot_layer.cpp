#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  K_ = this->layer_param_.embed_param().input_dim();
  CHECK_GT(K_, 0) << "OneHotLayer input_dim must be positive.";
  
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->shape(0);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(K_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  memset(top_data, static_cast<Dtype>(0), sizeof(Dtype) * M_ * K_);
  
  int index;
  for (int n = 0; n < M_; ++n) {
    index = static_cast<int>(bottom_data[n]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, K_);
    DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n]) << "non-integer input";
    DCHECK_LT(n * K_ + index, M_ * K_);
    top_data[n * K_ + index] = static_cast<Dtype>(1);    
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(OneHotLayer);
#endif

INSTANTIATE_CLASS(OneHotLayer);
REGISTER_LAYER_CLASS(OneHot);

}  // namespace caffe
