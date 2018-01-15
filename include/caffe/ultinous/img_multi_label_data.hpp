#ifndef CAFFE_IMG_MULTI__LABEL_DATA_LAYER_HPP_
#define CAFFE_IMG_MULTI__LABEL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "../ultinous/UltinousTransformer.hpp"


namespace caffe {

namespace ultinous{

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImgMultiLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImgMultiLabelDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param)
      , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
  { }
  ~ImgMultiLabelDataLayer() override;
  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override { return "ImgMultiLabelData"; }
  inline int ExactNumBottomBlobs() const override { return 0; }
  inline int ExactNumTopBlobs() const override { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  void load_batch(Batch<Dtype>* batch) override;

  vector<std::pair<std::string, std::vector<double> > > lines_;
  int lines_id_;
  
  ultinous::UltinousTransformer m_unTransformer;
};

} // namespace ultinous
} // namespace caffe

#endif  // CAFFE_IMG_MULTI_LABEL_DATA_LAYER_HPP_
