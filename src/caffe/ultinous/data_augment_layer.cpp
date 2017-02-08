#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/ultinous/data_augment_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void DataAugmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	m_numOcclusions = this->layer_param_.data_augment_param().num_occlusions();
	m_minOcclusionRadius = this->layer_param_.data_augment_param().min_occlusion_radius();
	m_maxOcclusionRadius = this->layer_param_.data_augment_param().max_occlusion_radius();

	top[0]->ReshapeLike(*bottom[0]);

	if( m_numOcclusions )
		m_circles.Reshape( bottom[0]->num(), 3*m_numOcclusions, 1, 1 );
}

template <typename Dtype>
void DataAugmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DataAugmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	/* not implemented */
	CHECK(false) << "Error: DataAugmentLayer<Dtype>::Forward_cpu is not implemented.";
}

template <typename Dtype>
void DataAugmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	/* not implemented */
	CHECK(false) << "Error: DataAugmentLayer<Dtype>::Backward_cpu is not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(DataAugmentLayer);
#endif

INSTANTIATE_CLASS(DataAugmentLayer);
REGISTER_LAYER_CLASS(DataAugment);

}  // namespace ultinous
}  // namespace caffe
