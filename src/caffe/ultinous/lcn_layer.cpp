#include <vector>


#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/ultinous/lcn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  m_window_size = this->layer_param_.lcn_param().window_size();

  int N = bottom[0]->shape(0);
  int H = bottom[0]->shape(2);
	int W = bottom[0]->shape(3);

  vector<int> shape(4);

  shape[0] = N;
  shape[1] = 1;
  shape[2] = H;
  shape[3] = W;

  m_grayscale.Reshape(shape);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void LCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int N = bottom[0]->shape(0);
  int H = bottom[0]->shape(2);
  int W = bottom[0]->shape(3);

  vector<int> shape(4);

  shape[0] = N;
  shape[1] = 1;
  shape[2] = H;
  shape[3] = W;

  m_grayscale.Reshape(shape);
  top[0]->Reshape(shape);
}


template <typename Dtype>
void LCNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
CHECK(false) << "LCNLayer::Forward_cpu is not implemented!";
}

template <typename Dtype>
void LCNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	CHECK(false) << "LCNLayer::Backward_cpu is not implemented!";
}

#ifdef CPU_ONLY
STUB_GPU(LCNLayer);
#endif

INSTANTIATE_CLASS(LCNLayer);
REGISTER_LAYER_CLASS(LCN);

}  // namespace caffe
