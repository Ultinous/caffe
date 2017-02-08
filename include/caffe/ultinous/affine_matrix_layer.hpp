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

namespace caffe {
namespace ultinous {

template <typename Dtype>
class AffineMatrixLayer : public Layer<Dtype> {

public:
	explicit AffineMatrixLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AffineMatrix"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  Dtype m_base_sx;
  Dtype m_base_sy;

  Dtype m_min_sx, m_max_sx, m_min_sy, m_max_sy;
  Dtype m_min_hx, m_max_hx, m_min_hy, m_max_hy;
  Dtype m_min_tx, m_max_tx, m_min_ty, m_max_ty;
  Dtype m_min_alpha, m_max_alpha;

  Dtype m_max_diff;

	bool m_normalize_angle;
	Dtype m_moving_average_angle;
	Dtype m_moving_average_fraction;
	Dtype m_normalization_coef;
};

}  // namespace ultinous
}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
