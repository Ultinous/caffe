#include "caffe/ultinous/nms_layer.hpp"

namespace caffe
{
namespace ultinous
{

template<typename Dtype>
void NMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  CHECK(this->layer_param_.has_nms_param());
  auto &nms_param = this->layer_param_.nms_param();

  kernel_size = nms_param.kernel_size();
}


template<typename Dtype>
void NMSLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                     << "corresponding to (num, channels, height, width)";
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top)
{
  const auto *bottom_data = bottom[0]->cpu_data();
  auto *top_data = top[0]->mutable_cpu_data();
  const auto top_count = top[0]->count();

  // Initialize
  caffe_set(top_count, Dtype(0), top_data);

  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int ph = 0; ph < bottom[0]->height(); ++ph) {
        for (int pw = 0; pw < bottom[0]->width(); ++pw) {
          int const hstart = std::max(ph - kernel_size / 2, 0);
          int const wstart = std::max(pw - kernel_size / 2, 0);
          int const hend = std::min(hstart + kernel_size, bottom[0]->height());
          int const wend = std::min(wstart + kernel_size, bottom[0]->width());
          int const pool_index = ph * bottom[0]->width() + pw;

          int max_index = -1;
          Dtype max_value = bottom_data[pool_index];
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int const index = h * bottom[0]->width() + w;
              if (bottom_data[index] > max_value) {
                max_index = index;
                max_value = bottom_data[index];
              }
            }
          }
          if (pool_index == max_index || max_value == bottom_data[pool_index])
            top_data[pool_index] = max_value;
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template<typename Dtype>
void NMSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom)
{
  // not implemented
}

#ifdef CPU_ONLY
STUB_GPU(NMSLayer);
#endif

INSTANTIATE_CLASS(NMSLayer);
REGISTER_LAYER_CLASS(NMS);

} // namespace ultinous
} // namespace caffe
