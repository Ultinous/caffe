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

  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;
    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]; // shear param
    Dtype hy = bottom_data[3]; // shear param
    Dtype tx = bottom_data[4]; // translate param
    Dtype ty = bottom_data[5]; // translate param
    Dtype al = bottom_data[6]; // alpha - rotatio angle;

//    std::cout << sx << " " << sy << " " << hx << " " << hy << " " << tx << " " << ty << " " << al << std::endl;

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *top_data = top[0]->mutable_cpu_data() + 6*n;
    top_data[0] = sx * ca - hy * sy * sa;
    top_data[1] = hx * sx * ca - sy * sa;
    top_data[2] = tx;
    top_data[3] = sx * sa + hy * sy * ca;
    top_data[4] = hx * sx * sa + sy * ca;
    top_data[5] = ty;
//  std::cout << top_data[0] << " " << top_data[1] << " " << top_data[2] << " " << top_data[3] << " " << top_data[4] << " " << top_data[5] << std::endl;
//  std::cout << std::endl;
  }

}

template <typename Dtype>
void AffineMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  for( int n = 0; n < bottom[0]->num(); ++n )
  {
    Dtype const *bottom_data = bottom[0]->cpu_data() + 7*n;
    Dtype sx = m_base_sx + bottom_data[0]; // scale param
    Dtype sy = m_base_sy + bottom_data[1]; // scale param
    Dtype hx = bottom_data[2]; // shear param
    Dtype hy = bottom_data[3]; // shear param
    //Dtype tx = bottom_data[4]; // translate param
    //Dtype ty = bottom_data[5]; // translate param
    Dtype al = bottom_data[6]; // alpha - rotatio angle

    Dtype ca = std::cos(al);
    Dtype sa = std::sin(al);

    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff() + 7*n;
    Dtype const *top_diff = top[0]->cpu_diff() + 6*n;

    bottom_diff[0] = (top_diff[0]*ca) + (top_diff[1]*hx*ca) + (top_diff[3]*sa) + (top_diff[4]*hx*sa);  // sx
    bottom_diff[1] = (top_diff[0]*hy*sa) + (top_diff[1]*sa) + (top_diff[3]*hy*ca) + (top_diff[4]*hx*ca);  // sy

    bottom_diff[2] = (top_diff[1]*sx*ca) + (top_diff[4]*sx*sa);  // hx
    bottom_diff[3] = (top_diff[0]*sy*sa) + (top_diff[3]*sy*ca);  // hy

    bottom_diff[4] = top_diff[2]; // tx
    bottom_diff[5] = top_diff[5]; // ty

    bottom_diff[6] = top_diff[0]*(sx*-sa - hy*sy*ca)
                    + top_diff[1]*(hx*sx*-sa - sy*ca)
                    + top_diff[3]*(sx*ca + hy*sy*-sa)
                    + top_diff[4]*(hx*sx*ca + sy*-sa);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AffineMatrixLayer);
#endif

INSTANTIATE_CLASS(AffineMatrixLayer);
REGISTER_LAYER_CLASS(AffineMatrix);

}  // namespace ultinous
}  // namespace caffe
