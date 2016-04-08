#pragma once


#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/ultinous/FeatureMap.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
class FeatureRegistrationLayer : public Layer<Dtype> {
 public:
  explicit FeatureRegistrationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FeatureRegistrationLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "FeatureRegistration"; }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace ultinous
}  // namespace caffe

