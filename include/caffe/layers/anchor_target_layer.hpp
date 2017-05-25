#pragma once

#include <vector>

#include "caffe/util/anchors.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {

protected:
  typedef std::vector <Dtype> Anchor;
  typedef std::vector< Anchor > Boxes;
  typedef float Overlap;
  typedef std::vector< std::vector<Overlap> > Overlaps;

 public:
  explicit AnchorTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param)
      , anchorTargetParam_( this->layer_param_.anchor_target_param() )
  {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "AnchorTarget"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int MinTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  Overlaps bbox_overlaps(  Boxes& boxes, Boxes& query_boxes );

  uint32_t hardNegativeMining( uint32_t num_fg, Dtype const * scores, Dtype * labels, uint32_t width, uint32_t height );
  uint32_t randomNegativeMining( uint32_t num_fg, Dtype const * scores, Dtype * labels, uint32_t width, uint32_t height );


  std::vector<Anchor> base_anchors_;

  int feat_stride_;
  int allowed_border_;

  AnchorTargetParameter const& anchorTargetParam_;
};

}  // namespace caffe
