#ifndef POWER_FILE_LAYER_HPP_
#define POWER_FILE_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
namespace ultinous
{

template<typename Dtype>
class ProjectiveMatrixLayer : public Layer<Dtype>
{

public:
  explicit ProjectiveMatrixLayer(const LayerParameter &param)
  : Layer<Dtype>(param)
  {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const
  { return "ProjectiveMatrix"; }

  virtual inline int ExactNumBottomBlobs() const
  { return 1; }

  virtual inline int ExactNumTopBlobs() const
  { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

private:
  Dtype m_base_scale;
  Dtype m_base_f;
  Dtype m_base_tz;

  Dtype m_min_scale, m_max_scale, m_min_f, m_max_f;
  Dtype m_min_alpha, m_max_alpha, m_min_beta;
  Dtype m_max_beta, m_min_gamma, m_max_gamma, m_min_tx;
  Dtype m_max_tx, m_min_ty, m_max_ty, m_min_tz, m_max_tz;

  Dtype m_max_diff;

  bool m_bias_scale, m_bias_U;

  Dtype m_boundary_violation_step;

  int m_iter;
};

}  // namespace ultinous
}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
