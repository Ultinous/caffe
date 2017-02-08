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

  m_normalize_angle = this->layer_param_.affine_matrix_param().normalize_angle();
  m_moving_average_angle = 0;
  m_moving_average_hx = 0;
  m_moving_average_hy = 0;
  m_moving_average_fraction = 0.9995;
  m_normalization_coef = 0.01;

m_iter = 0;

	vector<int> top_shape(2);
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

  if( m_normalize_angle ) {
    for (int n = 0; n < bottom[0]->num(); ++n) {
      Dtype const *bottom_data = bottom[0]->cpu_data() + 7 * n;

      Dtype hx = bottom_data[2]; // shear param
      Dtype hy = bottom_data[3]; // shear param
      Dtype al = bottom_data[6]; // alpha - rotatio angle;

      m_moving_average_angle = m_moving_average_fraction*m_moving_average_angle
                               + (1-m_moving_average_fraction)*al;
      m_moving_average_hx = m_moving_average_fraction*m_moving_average_hx
                               + (1-m_moving_average_fraction)*hx;
      m_moving_average_hy = m_moving_average_fraction*m_moving_average_hy
                               + (1-m_moving_average_fraction)*hy;
    }
  }

/*    if( ((++m_iter)%100)==0 )
	std::cout << "Angle normalizaton: " << m_normalize_angle
	  << " " << m_moving_average_angle
	  << " " << m_moving_average_hx
	  << " " << m_moving_average_hy
	  << std::endl;*/


  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;
    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]-m_moving_average_hx; // shear param
    Dtype hy = bottom_data[3]-m_moving_average_hy; // shear param
    Dtype tx = bottom_data[4]; // translate param
    Dtype ty = bottom_data[5]; // translate param
    Dtype al = bottom_data[6] - m_moving_average_angle; // alpha - rotatio angle;

//    std::cout << sx << " " << sy << " " << hx << " " << hy << " " << tx << " " << ty << " " << al << std::endl;

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *top_data = top[0]->mutable_cpu_data() + 6*n;
    top_data[0] = sx*ca - hy*sy*sa;
    top_data[1] = hx*sx*ca - sy*sa;
    top_data[2] = tx;
    top_data[3] = sx*sa + hy*sy*ca;
    top_data[4] = hx*sx*sa + sy*ca;
    top_data[5] = ty;


//  std::cout << top_data[0] << " " << top_data[1] << " " << top_data[2] << " " << top_data[3] << " " << top_data[4] << " " << top_data[5] << std::endl;
//  std::cout << std::endl;
  }

}

template <typename Dtype>
void AffineMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if( !propagate_down[0] ) return;

  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;
    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]-m_moving_average_hx; // shear param
    Dtype hy = bottom_data[3]-m_moving_average_hy; // shear param
    Dtype tx = bottom_data[4]; // translate param
    Dtype ty = bottom_data[5]; // translate param
    Dtype al = bottom_data[6]-m_moving_average_angle; // alpha - rotatio angle

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff() + 7*n;
    Dtype const *top_diff = top[0]->cpu_diff() + 6*n;



    if( sx < m_min_sx)
      bottom_diff[0] = sx - m_min_sx;
    else if( sx > m_max_sx)
      bottom_diff[0] = sx - m_max_sx;
    else
      bottom_diff[0] = (top_diff[0]*ca) + (top_diff[1]*hx*ca) + (top_diff[3]*sa) + (top_diff[4]*hx*sa);  // sx

    if( sy < m_min_sy)
      bottom_diff[1] = sy - m_min_sy;
    else if( sy > m_max_sy)
      bottom_diff[1] = sy - m_max_sy;
    else
      bottom_diff[1] = (top_diff[0]*-hy*sa) + (top_diff[1]*-sa) + (top_diff[3]*hy*ca) + (top_diff[4]*ca);  // sy

    if( hx < m_min_hx)
      bottom_diff[2] = hx - m_min_hx;
    else if( hx > m_max_hx)
      bottom_diff[2] = hx - m_max_hx;
    else
      bottom_diff[2] = m_normalization_coef*m_moving_average_hx + (top_diff[1]*sx*ca) + (top_diff[4]*sx*sa);  // hx

    if( hy < m_min_hy)
      bottom_diff[3] = hy - m_min_hy;
    else if( hy > m_max_hy)
      bottom_diff[3] = hy - m_max_hy;
    else
      bottom_diff[3] = m_normalization_coef*m_moving_average_hy + (top_diff[0]*-sy*sa) + (top_diff[3]*sy*ca);  // hy

    if( tx < m_min_tx)
      bottom_diff[4] = tx - m_min_tx;
    else if( tx > m_max_tx)
      bottom_diff[4] = tx - m_max_tx;
    else
      bottom_diff[4] = top_diff[2]; // tx

    if( ty < m_min_ty)
      bottom_diff[5] = ty - m_min_ty;
    else if( ty > m_max_ty)
      bottom_diff[5] = ty - m_max_ty;
    else
      bottom_diff[5] = top_diff[5]; // ty

    if( al < m_min_alpha)
      bottom_diff[6] = al - m_min_alpha;
    else if( al > m_max_alpha)
      bottom_diff[6] = al - m_max_alpha;
    else
      bottom_diff[6] = m_normalization_coef*m_moving_average_angle
                    + top_diff[0]*(sx*-sa - hy*sy*ca)
                    + top_diff[1]*(hx*sx*-sa - sy*ca)
                    + top_diff[3]*(sx*ca + hy*sy*-sa)
                    + top_diff[4]*(hx*sx*ca + sy*-sa);

    for( int i = 0; i < 7; i++ )
      bottom_diff[i] = std::max(-m_max_diff, std::min(m_max_diff, bottom_diff[i] ) );

  }
}

#ifdef CPU_ONLY
STUB_GPU(AffineMatrixLayer);
#endif

INSTANTIATE_CLASS(AffineMatrixLayer);
REGISTER_LAYER_CLASS(AffineMatrix);

}  // namespace ultinous
}  // namespace caffe
