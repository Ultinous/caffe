#ifndef ULTINOUS_DEMOGRAPHY_DATA_LAYER_HPP_
#define ULTINOUS_DEMOGRAPHY_DATA_LAYER_HPP_

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
namespace ultinous {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DemographyDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DemographyDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param)
      , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
  { }
  virtual ~DemographyDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Demography"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void load_batch(Batch<Dtype>* batch);

  std::map< int, std::vector<std::string> > m_files;
  int m_maxAge;
  int m_intervalLength;
  int m_additionalIntervals;
  int m_numIntervals;

  ultinous::UltinousTransformer m_unTransformer;
};


}  // namespace ultinous
}  // namespace caffe

#endif  // ULTINOUS_DEMOGRAPHY_DATA_LAYER_HPP_
