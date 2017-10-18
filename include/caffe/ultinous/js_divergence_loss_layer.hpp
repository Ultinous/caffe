#ifndef ULTINOUS_JS_DIVERGENCE_LOSS_LAYER_HPP_
#define ULTINOUS_JS_DIVERGENCE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class JSDivergenceLossLayer : public LossLayer<Dtype> {
 public:
  explicit JSDivergenceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "JSDivergenceLoss"; }

 protected:
  /// @copydoc JSDivergenceLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // ULTINOUS_JS_DIVERGENCE_LOSS_LAYER_HPP_
