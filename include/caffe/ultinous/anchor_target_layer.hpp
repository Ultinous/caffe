#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace ultinous {

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {

protected:
  typedef std::vector<Dtype> Anchor;
  typedef std::vector<Anchor> Boxes;
  typedef float Overlap;
  typedef std::vector<std::vector<Overlap> > Overlaps;

public:
  explicit AnchorTargetLayer(const LayerParameter &param)
      : Layer<Dtype>(param), anchorTargetParam_(this->layer_param_.anchor_target_param()) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) {}

  virtual inline const char *type() const { return "AnchorTarget"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }

  virtual inline int MinTopBlobs() const { return 4; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {}

  Overlaps bbox_overlaps(Boxes &boxes, Boxes &query_boxes);

  uint32_t hardNegativeMining(uint32_t num_fg, Dtype const *scores, Dtype *labels, uint32_t width, uint32_t height);

  uint32_t randomNegativeMining(uint32_t num_fg, Dtype const *scores, Dtype *labels, uint32_t width, uint32_t height);
//
//  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);
//
//  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
//                            const vector<Blob<Dtype> *> &bottom);
//
//
//  void inline debugblob(const Blob<Dtype>& blob)
//  {
//    std::cout << "Blob: "<<std::endl;
//    for(int ch = 0; ch < blob.shape(1) ; ch+=4 )
//    {
//      for(int h = 0; h < blob.shape(2); ++h)
//      {
//        for(int w = 0; w < blob.shape(3); ++w)
//        {
//          std::cout<<"(";
//          for(int i = 0; i<4; ++i)
//            std::cout<<blob.data_at(0,ch+i,h,w)<<" ";
//          std::cout<<") ";
//        }
//        std::cout<<std::endl;
//      }
//      std::cout<<"_____________"<<std::endl;
//    }
//    std::cout<< "Blob_end"<<std::endl;
//  };
  //std::vector<Anchor> base_anchors_;

  Blob<Dtype> base_anchors_;
  int base_anchors_size_;

  Blob<Dtype> anchors_;
  Blob<Dtype> anchors_scores_;

  Blob<int> anchors_validation_;

  Blob<int> anchor_gt_maximal_overlap_idx_;
  Blob<int> gt_anchor_maximal_overlap_idx_;

  Blob<int> random_seq_;
  Blob<Dtype> random_temp_;

  Blob<Dtype> anchor_overlaps_;

  Blob<Dtype> rpn_bbox_inside_weights_;

  int feat_stride_;
  int allowed_border_;

  AnchorTargetParameter const &anchorTargetParam_;
};

}  // namespace ultinous

}  // namespace caffe

