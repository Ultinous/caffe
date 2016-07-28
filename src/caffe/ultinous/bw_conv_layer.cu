#include <vector>
#include "caffe/ultinous/bw_conv_layer.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
__global__ void recalcBinaryWeightsGPUKernel (
      size_t const kernelNum
    , size_t const kernelSize
    , Dtype const* weight
    , bool * binary_weight
    , Dtype * alpha )
{
  CUDA_KERNEL_LOOP(channel, kernelNum) {
      Dtype alpha_ = 0;
      for( size_t i = 0; i < kernelSize; ++i )
      {
        alpha_ += abs( weight[channel*kernelSize + i] );
      }
      *alpha = alpha_ / kernelSize;

      for( size_t i = 0; i < kernelSize; ++i )
      {
        binary_weight[channel*kernelSize + i] =
          ((weight[channel*kernelSize + i] >= 0) ? true : false);
      }

  }
}

template <typename Dtype>
void BWConvolutionLayer<Dtype>::recalcBinaryWeightsGPU( )
{
    size_t kernelNum = this->blobs_[0]->shape(0);
    size_t kernelSize = this->blobs_[0]->count(1);

    recalcBinaryWeightsGPUKernel<Dtype>
      <<<CAFFE_GET_BLOCKS(kernelNum), CAFFE_CUDA_NUM_THREADS>>>(
        kernelNum
        , kernelSize
        , this->blobs_[0]->gpu_data()
        , this->binary_weights_
        , this->alpha_.mutable_gpu_data()
      );

    CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void my_gpu_gemm_kernel(
    const int M, const int N, const int K,
    const Dtype* A0, const bool* B0,
    Dtype* C0
    //, const Dtype * alpha
  )
{
  size_t i, j, k;
  Dtype const *A;
  Dtype  *C;
  bool const *B;
  Dtype res;

  //  A:m×k, B:k×n, C:m×n
  CUDA_KERNEL_LOOP(ix, M*N) {
    i = ix % M;
    j = ix / M;
    A = A0 + i;
    B = B0 + j * K;
    C = C0+ix;

    *C = 0;                       // !!!!!!

    res = Dtype(0);
    k = K;
    while(k--)
    {
      if( *B++ ) res += *A;
      else res -= *A;

      A += M;
    }
    *C = res;
  }
}

template<> void BWConvolutionLayer<float>::my_gpu_gemm(
    const int M, const int N, const int K,
    const bool* A, const float* B,
    float* C)
{
    my_gpu_gemm_kernel<float> <<<CAFFE_GET_BLOCKS(M*N), CAFFE_CUDA_NUM_THREADS>>> (
        N, M, K, B, A, C//, this->alpha_.gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;

//    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), M*N, this->alpha_.gpu_data(), C, 1));
    caffe_gpu_scal<float>( M*N, this->alpha_.cpu_data()[0], C );
}

template<> void BWConvolutionLayer<double>::my_gpu_gemm(
    const int M, const int N, const int K,
    const bool* A, const double* B,
    double* C)
{
    my_gpu_gemm_kernel<double> <<<CAFFE_GET_BLOCKS(M*N), CAFFE_CUDA_NUM_THREADS>>> (
        N, M, K, B, A, C//, this->alpha_.gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;

    //CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), M*N, this->alpha_.gpu_data(), C, 1));
    caffe_gpu_scal<double>( M*N, this->alpha_.cpu_data()[0], C );
}

/*template void BWConvolutionLayer<float>::my_gpu_gemm(
    const int M, const int N, const int K,
    const bool* A, const float* B, float* C);
template void BWConvolutionLayer<double>::my_gpu_gemm(
    const int M, const int N, const int K,
    const bool* A, const double* B, double* C);*/


template <typename Dtype>
void BWConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  recalcBinaryWeightsGPU( );

  const bool* weight = this->binary_weights_; //this->blobs_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_,false);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BWConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const bool* weight = this->binary_weights_; //this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BWConvolutionLayer);

}  // namespace ultinous
}  // namespace caffe
