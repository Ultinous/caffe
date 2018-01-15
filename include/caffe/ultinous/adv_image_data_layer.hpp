#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include <caffe/proto/caffe.pb.h>

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
class AdvImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit AdvImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param)
      , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
      , m_adv_params( param.adversarial_image_data_param() )
  { }
  ~AdvImageDataLayer() override;
  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override { return "AdvImageData"; }
  inline int ExactNumBottomBlobs() const override { return 0; }
  inline int ExactNumTopBlobs() const override { return 2; }

  void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  void load_batch(Batch<Dtype>* batch) override;

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;

  ultinous::UltinousTransformer m_unTransformer;

  typedef std::vector<Dtype> AdversarialImage;
  typedef std::pair<int, AdversarialImage> AdversarialImageForClass;
  typedef std::queue<AdversarialImageForClass> AdversarialImageQueue;
  AdversarialImageQueue m_advImages;

  AdversarialImageDataParamater m_adv_params;
};


}  // namespace ultinous
}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
