#pragma once

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/ultinous/ImageClassificationModel.h"
#include "caffe/ultinous/HardTripletGenerator.hpp"
#include "caffe/ultinous/RandomTripletGenerator.hpp"
#include "caffe/ultinous/TripletBatchGenerator.hpp"

#include "UltinousTransformer.hpp"

namespace caffe {
namespace ultinous {


template <typename Dtype>
class TripletDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param)
      , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
  { }

  ~TripletDataLayer() override;
  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override { return "TripletData"; }
  inline int ExactNumBottomBlobs() const override { return 0; }
  inline int ExactNumTopBlobs() const override { return 2; }

protected:
  void load_batch(Batch<Dtype>* batch) override;

private:
  ImageClassificationModel m_imageClassificationModel;

  typedef boost::shared_ptr<TripletBatchGenerator<Dtype> > TripletBatchGeneratorPtr;
  TripletBatchGeneratorPtr tripletBatchGenerator;

  vector<int> m_top_shape;
  vector<int> m_label_shape;

  typedef vector<Dtype> FeatureVector;
  vector<FeatureVector> m_inputFeatures;
  uint32_t m_inputFeatureLength;

  bool m_serialize;
  bool m_outputClasses;

  UltinousTransformer m_unTransformer;
};


}  // namespace ultinous
}  // namespace caffe
