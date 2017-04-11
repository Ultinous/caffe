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

  CHECK(bottom[0]->channels() == 9);

  m_base_scale = this->layer_param_.proj_matrix_param().base_scale();
  m_base_f = this->layer_param_.proj_matrix_param().base_f();
  m_base_tz = this->layer_param_.proj_matrix_param().base_tz();

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

  m_bias_scale = this->layer_param_.proj_matrix_param().bias_scale();
  m_bias_U = this->layer_param_.proj_matrix_param().bias_u();

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
    for (int c = 0; c < 9; ++c)
      std::cout << " " << bias[c];
    std::cout << std::endl;
  }

/**
  Original matrix:
  H = [ cos(gamma)*cos(beta)*S*f  (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*S*f  U*f;
      sin(gamma)*cos(beta)*S*f  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*S*f   V*f;
    -sin(beta)*S             cos(beta)*sin(alpha)*S                                     W+f];
*/

/** Non-uniform scaling:
  Original matrix:
  H = [ cos(gamma)*cos(beta)*Sx*f  (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*Sx*f  U*f*Sx;
      sin(gamma)*cos(beta)*Sy*f  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*Sy*f   V*f*Sy;
    -sin(beta)             cos(beta)*sin(alpha)                                     W+f];
*/


  for (int n = 0; n < bottom[0]->num(); ++n)
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 9 * n;

    Dtype Sx = m_base_scale + bottom_data[0];
    Dtype Sy = m_base_scale + bottom_data[1];
    Dtype f = m_base_f + bottom_data[2];
    Dtype alpha = bottom_data[3];
    Dtype beta = bottom_data[4];
    Dtype gamma = bottom_data[5];
    Dtype U = bottom_data[6];
    Dtype V = bottom_data[7];
    Dtype W = m_base_tz + bottom_data[8];

    Dtype const *bias = this->blobs_[0]->cpu_data();
    if (m_bias_scale)
    {
      Sx += bias[0];
      Sy += bias[1];
    }
    if (m_bias_U)
    {
      U += bias[6];
    }



    Dtype *top_data = top[0]->mutable_cpu_data() + 9 * n;
    top_data[0] = cos(gamma)*cos(beta)*Sx*f;
    top_data[1] = (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*Sx*f;
    top_data[2] = U*f*Sx;

    top_data[3] = sin(gamma)*cos(beta)*Sy*f;
    top_data[4] = (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*Sy*f;
    top_data[5] = V*f*Sy;

    top_data[6] = -sin(beta);
    top_data[7] = cos(beta)*sin(alpha);
    top_data[8] = W+f;
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
    Dtype const *bottom_data = bottom[0]->cpu_data() + 9 * n;

    Dtype Sx = m_base_scale + bottom_data[0];
    Dtype Sy = m_base_scale + bottom_data[1];
    Dtype f = m_base_f + bottom_data[2];
    Dtype alpha = bottom_data[3];
    Dtype beta = bottom_data[4];
    Dtype gamma = bottom_data[5];
    Dtype U = bottom_data[6];
    Dtype V = bottom_data[7];
    Dtype W = m_base_tz + bottom_data[8];

    Dtype const *bias = this->blobs_[0]->cpu_data();
    if (m_bias_scale)
    {
      Sx += bias[0];
      Sy += bias[1];
    }
    if (m_bias_U)
    {
      U += bias[6];
    }

    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff() + 9 * n;
    Dtype const *top_diff = top[0]->cpu_diff() + 9 * n;


    if (Sx < m_min_scale)
      bottom_diff[0] = (Sx - m_min_scale) * m_boundary_violation_step;
    else if (Sx > m_max_scale)
      bottom_diff[0] = (Sx - m_max_scale) * m_boundary_violation_step;
    else
      bottom_diff[0] = top_diff[0] * (cos(gamma)*cos(beta)*f)
                       +top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*f)
                      + top_diff[2] * U*f;


    if (Sy < m_min_scale)
      bottom_diff[1] = (Sy - m_min_scale) * m_boundary_violation_step;
    else if (Sy > m_max_scale)
      bottom_diff[1] = (Sy - m_max_scale) * m_boundary_violation_step;
    else
      bottom_diff[1] = top_diff[3] * ( sin(gamma)*cos(beta)*f )
                       +top_diff[4] * (  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*f )
                      + top_diff[5] * V*f;



    if (f < m_min_f)
      bottom_diff[2] = (f - m_min_f) * m_boundary_violation_step;
    else if (f > m_max_f)
      bottom_diff[2] = (f - m_max_f) * m_boundary_violation_step;
    else
      bottom_diff[2] =  top_diff[0] * (cos(gamma)*cos(beta)*Sx)
                        +top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*Sx)
                         +top_diff[2] * ( U * Sx )
                        +top_diff[3] * ( sin(gamma)*cos(beta)*Sy )
                        +top_diff[4] * (  (cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha))*Sy )
                        +top_diff[5] * ( V * Sy)
                        +top_diff[8] * ( 1 );



    if (alpha < m_min_alpha)
      bottom_diff[3] = (alpha - m_min_alpha) * m_boundary_violation_step;
    else if (alpha > m_max_alpha)
      bottom_diff[3] = (alpha - m_max_alpha) * m_boundary_violation_step;
    else
      bottom_diff[3] = top_diff[1] * ( (-sin(gamma)*-sin(alpha)+cos(gamma)*sin(beta)*cos(alpha))*Sx*f )
                      + top_diff[4] * ((cos(gamma)*-sin(alpha)+sin(gamma)*sin(beta)*cos(alpha))*Sy*f)
                      + top_diff[7] * (cos(beta)*cos(alpha));


    if (beta < m_min_beta)
      bottom_diff[4] = (beta - m_min_beta) * m_boundary_violation_step;
    else if (beta > m_max_beta)
      bottom_diff[4] = (beta - m_max_beta) * m_boundary_violation_step;
    else
      bottom_diff[4] = top_diff[0] * (cos(gamma)*-sin(beta)*Sx*f)
                  + top_diff[1] * ( (-sin(gamma)*cos(alpha)+cos(gamma)*cos(beta)*sin(alpha))*Sx*f)
              + top_diff[3] * (sin(gamma)*-sin(beta)*Sy*f)
            + top_diff[4] * ( (cos(gamma)*cos(alpha)+sin(gamma)*cos(beta)*sin(alpha))*Sy*f)
            + top_diff[6] * (-cos(beta))
            + top_diff[7] * (-sin(beta)*sin(alpha));

    if (gamma < m_min_gamma)
      bottom_diff[5] = (gamma - m_min_gamma) * m_boundary_violation_step;
    else if (gamma > m_max_gamma)
      bottom_diff[5] = (gamma - m_max_gamma) * m_boundary_violation_step;
    else
      bottom_diff[5] = top_diff[0] * (-sin(gamma)*cos(beta)*Sx*f)
              + top_diff[1] * ((-cos(gamma)*cos(alpha)+-sin(gamma)*sin(beta)*sin(alpha))*Sx*f)
                + top_diff[3] * (cos(gamma)*cos(beta)*Sy*f)
                  + top_diff[4] * ((-sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha))*Sy*f);

    if (U < m_min_tx)
      bottom_diff[6] = (U - m_min_tx) * m_boundary_violation_step;
    else if (U > m_max_tx)
      bottom_diff[6] = (U - m_max_tx) * m_boundary_violation_step;
    else
      bottom_diff[6] = top_diff[2]*f*Sx;

    if (V < m_min_ty)
      bottom_diff[7] = (V - m_min_ty) * m_boundary_violation_step;
    else if (V > m_max_ty)
      bottom_diff[7] = (V - m_max_ty) * m_boundary_violation_step;
    else
      bottom_diff[7] = top_diff[5]*f*Sy;

    if (W < m_min_tz)
      bottom_diff[8] = (W - m_min_tz) * m_boundary_violation_step;
    else if (W > m_max_tz)
      bottom_diff[8] = (W - m_max_tz) * m_boundary_violation_step;
    else
      bottom_diff[8] = top_diff[8] * (1);




    if (m_max_diff != 0)
      for (int i = 0; i < 9; i++)
        bottom_diff[i] = std::max(-m_max_diff, std::min(m_max_diff, bottom_diff[i]));

    // Bias - only on specific params
    Dtype *bias_diff = this->blobs_[0]->mutable_cpu_diff();
    if (m_bias_scale)
    {
      bias_diff[0] += bottom_diff[0]; // scaleX
      bias_diff[1] += bottom_diff[1]; // scaleY
    }
    if (m_bias_U)
    {
      bias_diff[6] += bottom_diff[6]; // // U
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
