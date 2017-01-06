#include "caffe/ultinous/similarity_loss_layer.hpp"

namespace caffe {
namespace ultinous {


template <typename Dtype>
__global__ void ForwardKernel(const int n, const Dtype* xv, const Dtype* yv, Dtype* out, Dtype sigma, int iteration) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = xv[index];
    Dtype y = yv[index];

    out[index] = 0.0;
    if( y >= 0.5 )
    {
      if( x < 0.7 )
        out[index] = 0.5*(0.7-x)*(0.7-x);
    }
    if( y < 0.5 )
    {
      if( x > 0.3 )
        out[index] = 0.5*(x-0.3)*(x-0.3);
    }
  }
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();


//  sigma_ = 0.2f - ( (0.2f-0.1f)*iter_/50000.0 );
//  iter_ = std::min(iter_+1, 50000ul);

  //std::cout << bottom[1]->cpu_data()[0] << " " << bottom[1]->cpu_data()[1] << std::endl;

  ForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), errors_.mutable_gpu_data(), sigma_, iteration );
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void BackwardKernel(const int n, const Dtype* xv, const Dtype* yv, Dtype* out, Dtype sigma, int iteration ) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = xv[index];
    Dtype y = yv[index];

    out[index] = 0.0;
    if( y >= 0.5 )
    {
      if( x < 0.7 )
        out[index] = (x-0.7);
    }
    if( y < 0.5 )
    {
      if( x > 0.3 )
        out[index] = (x-0.3);
    }
  }
}


template <typename Dtype>
void SimilarityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();

  caffe_gpu_set( count, Dtype(0), bottom[1]->mutable_gpu_diff() );

  //Dtype loss = top[0]->cpu_data()[0];

  BackwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff(), sigma_, iteration );
  CUDA_POST_KERNEL_CHECK;

  const Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_gpu_scal(count, loss_weight, bottom[0]->mutable_gpu_diff() );

  ++iteration;
}

INSTANTIATE_LAYER_GPU_FUNCS(SimilarityLossLayer);

}  // namespace ultinous
}  // namespace caffe
