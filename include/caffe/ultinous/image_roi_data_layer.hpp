#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/ultinous/UltinousTransformer.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
class ImageROIDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
public:

  class Batch {
    public:
    Blob<Dtype> data_, info_, bboxes_;
    std::vector< Blob<Dtype> > labels_;
  };

  struct BBox
  {
    int x1, y1, x2, y2;
    std::vector<int> classes;

  };
  typedef std::vector<BBox> BBoxes;

  struct Sample
  {
    std::vector<std::string> image_files;
    BBoxes bboxes;
  };
  typedef std::vector<Sample> Samples;

public:
  explicit ImageROIDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual ~ImageROIDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

  virtual inline const char* type() const
  {
    return "ImageROIData";
  }
  virtual inline int ExactNumBottomBlobs() const
  {
    return 0;
  }

 protected:
  virtual void InternalThreadEntry();
  virtual void ShuffleImages();
  virtual void load_batch(Batch* batch);




protected:
  Batch prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch*> prefetch_free_;
  BlockingQueue<Batch*> prefetch_full_;
  Blob<Dtype> transformed_data_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  Samples samples;
  int sample_id_;
  int m_labels_blobs_num;

  UltinousTransformer m_unTransformer;
};

}  // namespace ultinous
}  // namespace caffe
