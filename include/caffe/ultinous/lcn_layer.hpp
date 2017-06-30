#ifndef ST_LAYER_HPP_
#define ST_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LCNLayer : public Layer<Dtype> {

public:
	explicit LCNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
      }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "LCN"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	inline Dtype abs(Dtype x) {
		if(x < 0) return -x; return x;
	}
	inline Dtype max(Dtype x, Dtype y) {
		if(x < y) return y; return x;
	}

	Dtype transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
	void transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
			const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);

  int m_window_size;
	Blob<Dtype> m_grayscale;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
