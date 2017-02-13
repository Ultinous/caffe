#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/ultinous/affine_matrix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void AffineMatrixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK( bottom[0]->channels() == 7 );

  m_base_sx = this->layer_param_.affine_matrix_param().base_sx();
  m_base_sy = this->layer_param_.affine_matrix_param().base_sy();

  m_min_sx =  this->layer_param_.affine_matrix_param().min_sx();
  m_max_sx =  this->layer_param_.affine_matrix_param().max_sx();
  m_min_sy =  this->layer_param_.affine_matrix_param().min_sy();
  m_max_sy =  this->layer_param_.affine_matrix_param().max_sy();
  m_min_hx =  this->layer_param_.affine_matrix_param().min_hx();
  m_max_hx =  this->layer_param_.affine_matrix_param().max_hx();
  m_min_hy =  this->layer_param_.affine_matrix_param().min_hy();
  m_max_hy =  this->layer_param_.affine_matrix_param().max_hy();
  m_min_tx =  this->layer_param_.affine_matrix_param().min_tx();
  m_max_tx =  this->layer_param_.affine_matrix_param().max_tx();
  m_min_ty =  this->layer_param_.affine_matrix_param().min_ty();
  m_max_ty =  this->layer_param_.affine_matrix_param().max_ty();
  m_min_alpha =  this->layer_param_.affine_matrix_param().min_alpha();
  m_max_alpha =  this->layer_param_.affine_matrix_param().max_alpha();

  m_max_diff = this->layer_param_.affine_matrix_param().max_diff();

  m_normalize_params = (this->phase_ == TRAIN) && (this->layer_param_.affine_matrix_param().normalize_params());
  m_iter = 0;
  m_moving_average_fraction = 0.9999;

  if (this->blobs_.size() > 0) {
    Dtype const * saved_averages = this->blobs_[0]->cpu_data();
    LOG(INFO) << "Skipping parameter initialization";
    std::cout << "Saved averages:";
    for( int c = 0; c < 7; ++c )
      std::cout  << " " << saved_averages[c];
    std::cout << std::endl;
  } else {
    this->blobs_.resize(1);
    vector<int> sz;
    sz.push_back(bottom[0]->channels());
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    caffe_set(this->blobs_[0]->count(), Dtype(0),
	      this->blobs_[0]->mutable_cpu_data());
  }

  std::vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num( );
  top_shape[1] = 6;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AffineMatrixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num( );
  top_shape[1] = 6;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AffineMatrixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  ++m_iter;

  std::vector<Dtype> batch_averages(7, 0);
  Dtype const * averages = batch_averages.data();

  if( m_normalize_params ) {

    if( this->phase_ == TRAIN )
    {
      // Update running averages
      Dtype * saved_averages = this->blobs_[0]->mutable_cpu_data();
      for (int n = 0; n < bottom[0]->num(); ++n)
      {
	Dtype const *bottom_data = bottom[0]->cpu_data() + 7 * n;

	for( int c = 0; c < 7; ++c )
	  saved_averages[c] = m_moving_average_fraction*saved_averages[c]
			    + (1.0-m_moving_average_fraction) * bottom_data[c];
      }

      if( (m_iter%100)==0 )
      {
	  std::cout << "Moving averages:";
	  for( int c = 0; c < 7; ++c )
	    std::cout  << " " << saved_averages[c];
	  std::cout << std::endl;
      }

      // Compute batch averages
      for (int n = 0; n < bottom[0]->num(); ++n)
      {
	Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;

	for( int c = 0; c < 7; ++c )
	  batch_averages[c] += bottom_data[c];
      }
      for( int c = 0; c < 7; ++c )
	batch_averages[c] /= bottom[0]->num();

      if( (m_iter%100)==0 )
      {
	  std::cout << "Batch averages: ";
	  for( int c = 0; c < 7; ++c )
	    std::cout  << " " << batch_averages[c];
	  std::cout << std::endl;
      }
    }
    else
    {
      averages = this->blobs_[0]->cpu_data();
    }
  }

  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;

    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]-averages[2]; // shear param
    Dtype hy = bottom_data[3]-averages[3]; // shear param
    Dtype tx = bottom_data[4]-averages[4];             // translate param
    Dtype ty = bottom_data[5]-averages[5]; // translate param
    Dtype al = bottom_data[6]-averages[6]; // alpha - rotatio angle;

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *top_data = top[0]->mutable_cpu_data() + 6*n;
    top_data[0] = sx*ca - hy*sy*sa;
    top_data[1] = hx*sx*ca - sy*sa;
    top_data[2] = tx;
    top_data[3] = sx*sa + hy*sy*ca;
    top_data[4] = hx*sx*sa + sy*ca;
    top_data[5] = ty;
  }

}

template <typename Dtype>
void AffineMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if( !propagate_down[0] ) return;

  std::vector<Dtype> batch_averages(7, 0);
  Dtype const * averages = batch_averages.data();

  if( m_normalize_params ) {
    if( this->phase_ == TRAIN )
    {
      for (int n = 0; n < bottom[0]->num(); ++n)
      {
	Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;

	for( int c = 0; c < 7; ++c )
	  batch_averages[c] += bottom_data[c];
      }
      for( int c = 0; c < 7; ++c )
	batch_averages[c] /= bottom[0]->channels();
    }
    else
    {
      averages = this->blobs_[0]->cpu_data();
    }
  }

  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;

    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]-averages[2]; // shear param
    Dtype hy = bottom_data[3]-averages[3]; // shear param
    Dtype tx = bottom_data[4]-averages[4]; // translate param
    Dtype ty = bottom_data[5]-averages[5]; // translate param
    Dtype al = bottom_data[6]-averages[6]; // alpha - rotatio angle;

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff() + 7*n;
    Dtype const *top_diff = top[0]->cpu_diff() + 6*n;

    if( sx < m_min_sx)
      bottom_diff[0] = (sx - m_min_sx)*0.1;
    else if( sx > m_max_sx)
      bottom_diff[0] = (sx - m_max_sx)*0.1;
    else
      bottom_diff[0] = (top_diff[0]*ca) + (top_diff[1]*hx*ca) + (top_diff[3]*sa) + (top_diff[4]*hx*sa);  // sx

    if( sy < m_min_sy)
      bottom_diff[1] = (sy - m_min_sy)*0.1;
    else if( sy > m_max_sy)
      bottom_diff[1] = (sy - m_max_sy)*0.1;
    else
      bottom_diff[1] = (top_diff[0]*-hy*sa) + (top_diff[1]*-sa) + (top_diff[3]*hy*ca) + (top_diff[4]*ca);  // sy

    if( hx < m_min_hx)
      bottom_diff[2] = (hx - m_min_hx)*0.1;
    else if( hx > m_max_hx)
      bottom_diff[2] = (hx - m_max_hx)*0.1;
    else
      bottom_diff[2] = (top_diff[1]*sx*ca) + (top_diff[4]*sx*sa);  // hx

    if( hy < m_min_hy)
      bottom_diff[3] = (hy - m_min_hy)*0.1;
    else if( hy > m_max_hy)
      bottom_diff[3] = (hy - m_max_hy)*0.1;
    else
      bottom_diff[3] = (top_diff[0]*-sy*sa) + (top_diff[3]*sy*ca);  // hy

    if( tx < m_min_tx)
      bottom_diff[4] = (tx - m_min_tx)*0.1;
    else if( tx > m_max_tx)
      bottom_diff[4] = (tx - m_max_tx)*0.1;
    else
      bottom_diff[4] = top_diff[2]; // tx

    if( ty < m_min_ty)
      bottom_diff[5] = (ty - m_min_ty)*0.1;
    else if( ty > m_max_ty)
      bottom_diff[5] = (ty - m_max_ty)*0.1;
    else
      bottom_diff[5] = top_diff[5]; // ty

    if( al < m_min_alpha)
      bottom_diff[6] = (al - m_min_alpha)*0.1;
    else if( al > m_max_alpha)
      bottom_diff[6] = (al - m_max_alpha)*0.1;
    else
      bottom_diff[6] = top_diff[0]*(sx*-sa - hy*sy*ca)
                    + top_diff[1]*(hx*sx*-sa - sy*ca)
                    + top_diff[3]*(sx*ca + hy*sy*-sa)
                    + top_diff[4]*(hx*sx*ca + sy*-sa);

//    for( int i = 0; i < 7; i++ )
//	bottom_diff[i]*=0.96875;

  for( int i = 0; i < 7; i++ )
      bottom_diff[i] = std::max(-m_max_diff, std::min(m_max_diff, bottom_diff[i] ) );

  for( int i = 0; i < 7; i++ )
      bottom_diff[i] /= bottom[0]->num();

  }
}

#ifdef CPU_ONLY
STUB_GPU(AffineMatrixLayer);
#endif

INSTANTIATE_CLASS(AffineMatrixLayer);
REGISTER_LAYER_CLASS(AffineMatrix);

}  // namespace ultinous
}  // namespace caffe
