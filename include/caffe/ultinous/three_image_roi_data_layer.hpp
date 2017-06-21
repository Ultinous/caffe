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
class ThreeImageROIDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
public:

  class BatchWithBoxes {
    public:
    Blob<Dtype> data_, info_, bboxes_;
  };

  struct BBox
  {
    int x1, y1, x2, y2, label;

  };

  typedef std::vector<BBox> BBoxes;

  struct SequenceElement
  {
    std::string image_file;
    BBoxes bboxes;
  };

  struct Sequence{
  public:
    Sequence(std::vector<SequenceElement> elements)
    :m_elements(elements), m_current(1)
    {}

    BBoxes getCurrentBboxes()
    {
      return m_elements[m_current].bboxes;
    }

    std::string getCurrentImageFile()
    {
      return m_elements[m_current].image_file;
    }

    std::string getNextImageFile()
    {
      return m_elements[m_current+1].image_file;
    }

    std::string getPrevImageFile()
    {
      return m_elements[m_current-1].image_file;
    }

    void step()
    {
      if(m_current < m_elements.size()-2)
        ++m_current;
      else
        m_current = 1;
    }
  private:
    std::vector<SequenceElement> m_elements;
    std::size_t m_current;
  };


  typedef std::vector<Sequence> Samples;

public:
  explicit ThreeImageROIDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual ~ThreeImageROIDataLayer();
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
    return "ThreeImageROIData";
  }
  virtual inline int ExactNumBottomBlobs() const
  {
    return 0;
  }

 protected:
  virtual void InternalThreadEntry();
  virtual void ShuffleImages();
  virtual void load_batch(BatchWithBoxes* batch);




protected:
  BatchWithBoxes prefetch_[PREFETCH_COUNT];
  BlockingQueue<BatchWithBoxes*> prefetch_free_;
  BlockingQueue<BatchWithBoxes*> prefetch_full_;
  Blob<Dtype> transformed_data_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  Samples samples;
  int sample_id_;

  //UltinousTransformer m_unTransformer;
};

}  // namespace ultinous
}  // namespace caffe
