#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/mixup_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MixupLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	dist_filler_.reset(GetFiller<Dtype>(this->layer_param_.dist_filler()));
}

template <typename Dtype>
void MixupLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
	top[0]->Reshape(bottom[0]->shape());	
	top[1]->Reshape(bottom[1]->shape());
	
	std::vector<int> img_shape = bottom[0]->shape();
	
	img_size_ = img_shape[1] * img_shape[2] * img_shape[3];
	
	batch_size_ = bottom[1]->shape()[0];
	num_classes_ = bottom[1]->shape()[1];
	
	std::vector<int> dist_samples_shape({batch_size_});
	
	dist_samples_.Reshape(dist_samples_shape);
	
	permutation_vec_.resize(batch_size_);
}

template <typename Dtype>
void MixupLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	const Dtype* bot_img_data = bottom[0]->cpu_data();
	const Dtype* bot_class_dist_data = bottom[1]->cpu_data();
	
	Dtype* top_img_data = top[0]->mutable_cpu_data();
	Dtype* top_class_dist_data = top[1]->mutable_cpu_data();
	
	caffe_copy(batch_size_ * img_size_, bot_img_data, top_img_data);
	caffe_copy(batch_size_ * num_classes_, bot_class_dist_data, top_class_dist_data);
	
	dist_filler_->Fill(&dist_samples_);
	
	for(int i=0; i < batch_size_; ++i) {
		permutation_vec_[i] = i;
	}
	
	std::random_shuffle(permutation_vec_.begin(), permutation_vec_.end());
	
	const Dtype* dist_samples_data = dist_samples_.cpu_data();
	
	for(int i=0; i < batch_size_; ++i) {
		Dtype scale = dist_samples_data[i];
		int other_i = permutation_vec_[i];
		
		caffe_cpu_axpby(img_size_, 1 - scale, bot_img_data + other_i * img_size_, scale, top_img_data + i * img_size_);
		caffe_cpu_axpby(num_classes_, 1 - scale, bot_class_dist_data + other_i * num_classes_, scale, top_class_dist_data + i * num_classes_);
	}		
}

template <typename Dtype>
void MixupLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(MixupLayer);
#endif

INSTANTIATE_CLASS(MixupLayer);
REGISTER_LAYER_CLASS(Mixup);

}  // namespace caffe
