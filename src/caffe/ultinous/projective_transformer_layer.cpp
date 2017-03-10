#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/ultinous/projective_transformer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {



	//std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	output_H_ = bottom[0]->shape(2);
	if(this->layer_param_.proj_trans_param().has_output_h()) {
		output_H_ = this->layer_param_.proj_trans_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if(this->layer_param_.proj_trans_param().has_output_w()) {
		output_W_ = this->layer_param_.proj_trans_param().output_w();
	}


	// check the validation for the parameter theta
	CHECK(bottom[1]->count(1)  == 9) << "Theta dimension does not match" << std::endl;
	CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << "The first dimension of theta and bottom[0] should be the same" << std::endl;

	// initialize the matrix for output grid
	//std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

	vector<int> shape_output(2);
	shape_output[0] = output_H_ * output_W_; shape_output[1] = 3;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for(int i=0; i<output_H_ * output_W_; ++i) {
		data[3 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[3 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
		data[3 * i + 2] = 1;
	}

	// initialize the matrix for input grid
	//std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 3;
	input_grid.Reshape(shape_input);

	//std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);

	// reshape dTheta_tmp
	vector<int> dTheta_tmp_shape(4);

	dTheta_tmp_shape[0] = N;
	dTheta_tmp_shape[1] = 9;
	dTheta_tmp_shape[2] = 1;
	dTheta_tmp_shape[3] = output_H_ * output_W_ * C;

	dTheta_tmp.Reshape(dTheta_tmp_shape);

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] = output_H_ * output_W_ * C;
	all_ones_2.Reshape(all_ones_2_shape);

	// reshape full_theta
	vector<int> full_theta_shape(2);
	full_theta_shape[0] = N;
	full_theta_shape[1] = 9;
	full_theta.Reshape(full_theta_shape);

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 3;
	input_grid.Reshape(shape_input);
}


template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
CHECK(false) << "ProjectiveTransformerLayer::Forward_cpu is not implemented!";
}

template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	CHECK(false) << "ProjectiveTransformerLayer::Backward_cpu is not implemented!";
}

#ifdef CPU_ONLY
STUB_GPU(ProjectiveTransformerLayer);
#endif

INSTANTIATE_CLASS(ProjectiveTransformerLayer);
REGISTER_LAYER_CLASS(ProjectiveTransformer);

}  // namespace caffe
