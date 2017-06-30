#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/ultinous/lcn_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void LCNForwardGPU(const int nthreads, int N, int H, int W, int winSize, const Dtype* in, Dtype* out) {
  int radius = (winSize-1)/2;
  int spatialSize = H*W;
	
	CUDA_KERNEL_LOOP(index, nthreads) {

    int channel = index / spatialSize;
    int baseOffset = channel * spatialSize;

    int centerX = (index%spatialSize) / W;
    int centerY = (index%spatialSize) % W;

    int minX = max(0, centerX-radius);
    int maxX = min(H-1, centerX+radius);
    int minY = max(0, centerY-radius);
    int maxY = min(W-1, centerY+radius);

    int count = (maxY-minY+1)*(maxX-minX+1);

    Dtype mean = Dtype(0);
    for( int x = minX; x <= maxX; ++x )
      for( int y = minY; y <= maxY; ++y )
        mean += in[baseOffset + x*W + y];

    mean /= count;

    Dtype std = Dtype(0);
    for( int x = minX; x <= maxX; ++x )
      for( int y = minY; y <= maxY; ++y )
      {
        Dtype diff = (in[baseOffset + x * W + y] - mean);
        std += diff*diff;
      }

    std = sqrt(std/count);

    out[index] = (in[index] - mean) / (std+Dtype(0.001));
  }
}

template <typename Dtype>
void LCNLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int N = bottom[0]->shape(0);
  int C = bottom[0]->shape(1);
  int H = bottom[0]->shape(2);
  int W = bottom[0]->shape(3);
  int channelSize = H*W;

  caffe_gpu_memcpy( bottom[0]->count()*sizeof(Dtype), bottom[0]->gpu_data(), m_grayscale.mutable_gpu_data() );
  for( int i = 1; i < C; ++i )
    caffe_gpu_axpy( bottom[0]->count(), Dtype(1.0), bottom[0]->gpu_data()+i*channelSize, m_grayscale.mutable_gpu_data() );


  int nthreads = N*H*W;
	LCNForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, H, W, m_window_size, m_grayscale.gpu_data(), top[0]->mutable_gpu_data() );
}




template <typename Dtype>
void LCNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


}

INSTANTIATE_LAYER_GPU_FUNCS(LCNLayer);

}	// namespace caffe
