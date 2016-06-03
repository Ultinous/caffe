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
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImgMultiLabelDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImgMultiLabelData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::vector<double> > > lines_;
  int lines_id_;
};

}  // nemaspace ultinous
}  // namespace caffe

#endif  // CAFFE_IMG_MULTI_LABEL_DATA_LAYER_HPP_