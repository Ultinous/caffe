#pragma once

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/ultinous/PictureClassificationModel.h"
#include "caffe/ultinous/HardTripletGenerator.hpp"

namespace caffe {
namespace ultinous {


template <typename Dtype>
class TripletDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~TripletDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;

private:
  ImageClassificationModel imageClassificationModel;
  typedef boost::shared_ptr<HardTripletGenerator<Dtype> > HardTripletGeneratorPtr;
  HardTripletGeneratorPtr hardTripletGenerator;
};


}  // namespace ultinous
}  // namespace caffe
