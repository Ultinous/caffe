#pragma once

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
namespace ultinous
{

template<typename Dtype>
class NMSLayer : public Layer<Dtype>
{
public:
  explicit NMSLayer(const LayerParameter &param)
    : Layer<Dtype>(param)
  {}

  void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;
  void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;
  virtual inline const char *type() const override { return "NMS"; }
  virtual inline int ExactNumBottomBlobs() const override { return 1; }
  virtual inline int ExactNumTopBlobs() const override { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  int kernel_size;
};

}
}
