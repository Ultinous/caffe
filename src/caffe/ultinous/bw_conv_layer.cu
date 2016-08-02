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

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
#define BLOCK_SIZE 16
__global__ void my_gpu_gemm_kernel_v2(
    const int m, const int n, const int k,
    const float* A, const bool* B,
    float* C
    //, const Dtype * alpha
  )
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ bool Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A
        const float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        const bool* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j)
          Cvalue += Bs[j][col]? As[row][j] : -As[row][j];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue;
}

template<> void BWConvolutionLayer<float>::my_gpu_gemm(
    const int M, const int N, const int K,
    const bool* A, const float* B,
    float* C)
{
/*    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(M / BLOCK_SIZE + 1, K / BLOCK_SIZE + 1);
    my_gpu_gemm_kernel_v2 <<<gridDim, blockDim>>> (
        N, M, K, B, A, C//, this->alpha_.gpu_data()
    );*/
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
