#pragma once

#include "caffe/layers/base_data_layer.hpp"

namespace caffe
{
namespace ultinous
{

/**
 * @brief Provides data for the net from list of floats.
 * @tparam Dtype
 */
template <typename Dtype>
class QueuePredictionDataLayer
  : public BasePrefetchingDataLayer<Dtype>
{
public:
  explicit QueuePredictionDataLayer(const LayerParameter &param)
    : BasePrefetchingDataLayer<Dtype>(param)
  { }

  virtual ~QueuePredictionDataLayer();
  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  inline const char* type() const { return "QueuePredictionData"; }
  inline int ExactNumBottomBlobs() const { return 0; }
  inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  void ShuffleData();
  void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::vector<Dtype>, std::vector<Dtype>> > lines_;
  int lines_id_;

  size_t m_dataLength;
  size_t m_labelLength;
};

} // namespace ultinous
} // namespace caffe
