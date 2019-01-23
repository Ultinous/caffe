#pragma once

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db.hpp"
#include "caffe/ultinous/cpm_data_transformer.hpp"

namespace caffe
{
namespace ultinous
{
  /**
   * Based on:
   * https://github.com/CMU-Perceptual-Computing-Lab/caffe_train/commit/10275384a3980ffa6a99061d01e7256b31a68c43
   */
  template<typename Dtype>
  class CPMDataLayer : public BasePrefetchingDataLayer<Dtype>
  {
  public:
    explicit CPMDataLayer(const LayerParameter&);
    ~CPMDataLayer() override;

    void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

    inline const char* type() const override { return "CPMData"; }
    inline int ExactNumBottomBlobs() const override { return 0; }
    inline int ExactNumTopBlobs() const override { return 2; }

  protected:
    //virtual void ShuffleImages();
    void Next();
    bool Skip();

    void load_batch(Batch<Dtype>* batch) override;

    shared_ptr<CPMDataTransformer<Dtype> > cpm_data_transformer_;

//    shared_ptr<Caffe::RNG> prefetch_rng_;
    shared_ptr<db::DB> db_;
    shared_ptr<db::Cursor> cursor_;
    uint64_t offset_;
    Blob<Dtype> transformed_label_;
  };
} // namespace ultinous
} // namespace caffe