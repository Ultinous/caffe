#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/ultinous/projective_matrix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
namespace ultinous
{

template<typename Dtype>
void ProjectiveMatrixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top)
{

  CHECK(bottom[0]->channels() == 8);

  m_base_scale = this->layer_param_.proj_matrix_param().base_scale();
  m_base_f = this->layer_param_.proj_matrix_param().base_f();

  m_min_scale = this->layer_param_.proj_matrix_param().min_scale();
  m_max_scale = this->layer_param_.proj_matrix_param().max_scale();
  m_min_f = this->layer_param_.proj_matrix_param().min_f();
  m_max_f = this->layer_param_.proj_matrix_param().max_f();
  m_min_alpha = this->layer_param_.proj_matrix_param().min_alpha();
  m_max_alpha = this->layer_param_.proj_matrix_param().max_alpha();
  m_min_beta = this->layer_param_.proj_matrix_param().min_beta();
  m_max_beta = this->layer_param_.proj_matrix_param().max_beta();
  m_min_gamma = this->layer_param_.proj_matrix_param().min_gamma();
  m_max_gamma = this->layer_param_.proj_matrix_param().max_gamma();

  m_min_tx = this->layer_param_.proj_matrix_param().min_tx();
  m_max_tx = this->layer_param_.proj_matrix_param().max_tx();
  m_min_ty = this->layer_param_.proj_matrix_param().min_ty();
  m_max_ty = this->layer_param_.proj_matrix_param().max_ty();
  m_min_tz = this->layer_param_.proj_matrix_param().min_tz();
  m_max_tz = this->layer_param_.proj_matrix_param().max_tz();

  m_bias = this->layer_param_.proj_matrix_param().bias();

  m_max_diff = this->layer_param_.proj_matrix_param().max_diff();
  m_boundary_violation_step = 0.1;
  m_iter = 0;

  if (this->blobs_.size() > 0)
  {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else
  {
    this->blobs_.resize(1);
    vector<int> sz;
    sz.push_back(bottom[0]->channels());

    this->blobs_[0].reset(new Blob<Dtype>(sz));
    caffe_set(this->blobs_[0]->count(), Dtype(0),
              this->blobs_[0]->mutable_cpu_data());
  }

  std::vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 9;
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void ProjectiveMatrixLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
{
  std::vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 9;
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void ProjectiveMatrixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top)
{
  ++m_iter;
  if (this->phase_ == TRAIN && (m_iter % 100) == 0)
  {
    Dtype const *bias = this->blobs_[0]->cpu_data();
    std::cout << "Bias:";
    for (int c = 0; c < 8; ++c)
      std::cout << " " << bias[c];
    std::cout << std::endl;
  }

  for (int n = 0; n < bottom[0]->num(); ++n)
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 8 * n;

    Dtype S = m_base_scale + bottom_data[0];
    Dtype f = m_base_f + bottom_data[1];
    Dtype alpha = bottom_data[2];
    Dtype beta = bottom_data[3];
    Dtype gamma = bottom_data[4];
    Dtype U = bottom_data[5];
    Dtype V = bottom_data[6];
    Dtype W = bottom_data[7];

/**
  Original matrix:
  H = [ cos(gamma)*cos(beta)*S*f  (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S*f  U*f;
      sin(gamma)*cos(beta)*S*f  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*S*f   V*f;
    -sin(beta)*S             cos(beta)*sin(alpha)*S                                     W+f];
*/

/**
  Inverted focal length:
  H = [ cos(gamma)*cos(beta)*S*f  (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S*f  U*f;
      sin(gamma)*cos(beta)*S*f  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*S*f   V*f;
      sin(beta)*S             -cos(beta)*sin(alpha)*S                                     f-W];
*/

    if (m_bias)
    {
      Dtype const *bias = this->blobs_[0]->cpu_data();
      S += bias[0];
      f += bias[1];
      alpha += bias[2];
      beta += bias[3];
      gamma += bias[4];
      U += bias[5];
      V += bias[6];
      W += bias[7];
    }


    Dtype *top_data = top[0]->mutable_cpu_data() + 9 * n;
    top_data[0] = cos(gamma)*cos(beta)*S*f;
    top_data[1] = (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S*f;
    top_data[2] = U*f;

    top_data[3] = sin(gamma)*cos(beta)*S*f;
    top_data[4] = (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*S*f;
    top_data[5] = V*f;

    top_data[6] = sin(beta)*S;
    top_data[7] = -cos(beta)*sin(alpha)*S;
    top_data[8] = f-W;
  }

}

template<typename Dtype>
void ProjectiveMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                                const vector<bool> &propagate_down,
                                                const vector<Blob<Dtype> *> &bottom)
{

  if (!propagate_down[0]) return;


  for (int n = 0; n < bottom[0]->num(); ++n)
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 8 * n;

    Dtype S = m_base_scale + bottom_data[0];
    Dtype f = m_base_f + bottom_data[1];
    Dtype alpha = bottom_data[2];
    Dtype beta = bottom_data[3];
    Dtype gamma = bottom_data[4];
    Dtype U = bottom_data[5];
    Dtype V = bottom_data[6];
    Dtype W = bottom_data[7];


    if (m_bias)
    {
      Dtype const *bias = this->blobs_[0]->cpu_data();
      S += bias[0];
      f += bias[1];
      alpha += bias[2];
      beta += bias[3];
      gamma += bias[4];
      U += bias[5];
      V += bias[6];
      W += bias[7];
    }

    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff() + 8 * n;
    Dtype const *top_diff = top[0]->cpu_diff() + 9 * n;


    if (S < m_min_scale)
      bottom_diff[0] = (S - m_min_scale) * m_boundary_violation_step;
    else if (S > m_max_scale)
      bottom_diff[0] = (S - m_max_scale) * m_boundary_violation_step;
    else
      bottom_diff[0] = top_diff[0] * (cos(gamma)*cos(beta)*f)
                       +top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*f)
                       +top_diff[3] * ( sin(gamma)*cos(beta)*f )
                       +top_diff[4] * (  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*f )
                       +top_diff[6] * ( sin(beta) )
                       +top_diff[7] * ( -cos(beta)*sin(alpha) );



    if (f < m_min_f)
      bottom_diff[1] = (f - m_min_f) * m_boundary_violation_step;
    else if (f > m_max_f)
      bottom_diff[1] = (f - m_max_f) * m_boundary_violation_step;
    else
      bottom_diff[1] =  top_diff[0] * (cos(gamma)*cos(beta)*S)
                        +top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S)
                         +top_diff[2] * ( U )
                        +top_diff[3] * ( sin(gamma)*cos(beta)*S )
                        +top_diff[4] * (  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*S )
                        +top_diff[5] * ( V )
                        +top_diff[8] * ( 1 );


    if (alpha < m_min_alpha)
      bottom_diff[2] = (alpha - m_min_alpha) * m_boundary_violation_step;
    else if (alpha > m_max_alpha)
      bottom_diff[2] = (alpha - m_max_alpha) * m_boundary_violation_step;
    else
      bottom_diff[2] = top_diff[1] * ( (-sin(gamma)*-sin(alpha)+cos(gamma)*sin(beta)*cos(alpha))*S*f )
                      + top_diff[4] * ((cos(gamma)*-sin(alpha)+sin(gamma)*sin(beta)*cos(alpha))*S*f)
                      + top_diff[7] * (-cos(beta)*cos(alpha)*S);


    if (beta < m_min_beta)
      bottom_diff[3] = (beta - m_min_beta) * m_boundary_violation_step;
    else if (beta > m_max_beta)
      bottom_diff[3] = (beta - m_max_beta) * m_boundary_violation_step;
    else
      bottom_diff[3] = top_diff[0] * (cos(gamma)*-sin(beta)*S*f)
                  + top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*cos(beta)*sin(alpha))*S*f)
              + top_diff[3] * (sin(gamma)*-sin(beta)*S*f)
            + top_diff[4] * ( (cos(gamma)*cos(alpha)+sin(gamma)*cos(beta)*sin(alpha))*S*f)
            + top_diff[6] * (cos(beta)*S)
            + top_diff[7] * (sin(beta)*sin(alpha)*S);

    if (gamma < m_min_gamma)
      bottom_diff[4] = (gamma - m_min_gamma) * m_boundary_violation_step;
    else if (gamma > m_max_gamma)
      bottom_diff[4] = (gamma - m_max_gamma) * m_boundary_violation_step;
    else
      bottom_diff[4] = top_diff[0] * (-sin(gamma)*cos(beta)*S*f)
              + top_diff[1] * ((-cos(gamma)*cos(alpha)+-sin(gamma)*sin(beta)*sin(alpha))*S*f)
                + top_diff[3] * (cos(gamma)*cos(beta)*S*f)
                  + top_diff[4] * ((-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S*f);

    if (U < m_min_tx)
      bottom_diff[5] = (U - m_min_tx) * m_boundary_violation_step;
    else if (U > m_max_tx)
      bottom_diff[5] = (U - m_max_tx) * m_boundary_violation_step;
    else
      bottom_diff[5] = top_diff[2]*f;

    if (V < m_min_ty)
      bottom_diff[6] = (V - m_min_ty) * m_boundary_violation_step;
    else if (V > m_max_ty)
      bottom_diff[6] = (V - m_max_ty) * m_boundary_violation_step;
    else
      bottom_diff[6] = top_diff[5]*f;

    if (W < m_min_tz)
      bottom_diff[7] = (W - m_min_tz) * m_boundary_violation_step;
    else if (W > m_max_tz)
      bottom_diff[7] = (W - m_max_tz) * m_boundary_violation_step;
    else
      bottom_diff[7] = top_diff[8] * (-1);




    if (m_max_diff != 0)
      for (int i = 0; i < 8; i++)
        bottom_diff[i] = std::max(-m_max_diff, std::min(m_max_diff, bottom_diff[i]));

    // Bias - only on specific params
    if (m_bias)
    {
      Dtype *bias_diff = this->blobs_[0]->mutable_cpu_diff();
      bias_diff[0] += bottom_diff[0]; // scale
      bias_diff[1] += bottom_diff[1]; // f
      bias_diff[5] += bottom_diff[5]; // // U
      bias_diff[7] += bottom_diff[7]; // // W
    }


  }
}

#ifdef CPU_ONLY
STUB_GPU(ProjectiveMatrixLayer);
#endif

INSTANTIATE_CLASS(ProjectiveMatrixLayer);

REGISTER_LAYER_CLASS(ProjectiveMatrix);

}  // namespace ultinous
}  // namespace caffe
