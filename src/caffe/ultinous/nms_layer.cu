#include "caffe/ultinous/nms_layer.hpp"

namespace caffe
{
namespace ultinous
{

template<typename Dtype, uint R>
static
__global__ void nmsFilterKernel(Dtype *dst, Dtype const *src, int const width, int const height, int const channels)
{
  int const blockStartX = blockDim.x * blockIdx.x;
  int const blockStartY = blockDim.y * blockIdx.y;

  int const x = blockStartX + threadIdx.x;
  int const y = blockStartY + threadIdx.y;
  int const c = blockIdx.z % channels;
  int const ind = blockIdx.z / channels;
  int const channelStart = ind * width * height * channels + c * width * height;

  int const shmHeight = blockDim.y + 2 * R;
  int const shmWidth = blockDim.x + 2 * R;

  extern __shared__ float shm[];

#pragma unroll
  for(int j = threadIdx.y; j < shmHeight; j += blockDim.y)
  {
#pragma unroll
    for(int i = threadIdx.x; i < shmWidth; i += blockDim.x)
    {
      int const locX = max(min(blockStartX + i - R, int(width - 1) ), 0);
      int const locY = max(min(blockStartY + j - R, int(height - 1) ), 0);
      shm[j * shmWidth + i] = src[channelStart + locY * width + locX];
    }
  }

  __syncthreads();

  int const locX = threadIdx.x + R;
  int const locY = threadIdx.y + R;

  Dtype const val = shm[locY * shmWidth + locX];
  Dtype maxVal = val;

#pragma unroll
  for(int j = locY - R; j < locY + R + 1; ++j)
#pragma unroll
    for(int i = locX - R; i < locX + R + 1; ++i )
      maxVal = fmax(shm[j * shmWidth + i], maxVal);

  if(y < height && x < width)
    dst[channelStart + y * width + x] = (val == maxVal)? val : 0.f;
}

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 32

template <typename Dtype, uint R>
static inline
void nmsFilterKernelWrapper(Blob<Dtype> const *src, Blob<Dtype> *dst)
{
  Dtype const *src_data = src->gpu_data();
  Dtype *dst_data = dst->mutable_gpu_data();

  auto const num_ = src->num();
  auto const channels_ = src->channels();
  auto const height_ = src->height();
  auto const width_ = src->width();

  dim3 const block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 const grid(
    static_cast<unsigned int>((width_ + BLOCK_DIM_X - 1) / BLOCK_DIM_X)
    , static_cast<unsigned int>((height_ + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y)
    , static_cast<unsigned int>(channels_ * num_)
  );

  size_t shmInBytes = (2 * R + BLOCK_DIM_Y) * (2 * R + BLOCK_DIM_X) * sizeof(float);

  nmsFilterKernel<Dtype, R><<<grid, block, shmInBytes, Caffe::cuda_stream()>>>(dst_data, src_data, width_, height_, channels_);
}



template <typename Dtype, uint MAX>
static inline
void expandNMSFilterKernel(Blob<Dtype> const *src, Blob<Dtype> *dst, uint radius)
{
  if( radius < MAX )
  {
    expandNMSFilterKernel<Dtype, MAX - 1>(src, dst, radius);
  }
  else
    nmsFilterKernelWrapper<Dtype, MAX>(src, dst);
}


template <>
void expandNMSFilterKernel<float, 0>(Blob<float> const *src, Blob<float> *dst, uint)
{
  NOT_IMPLEMENTED;
}

template <>
void expandNMSFilterKernel<double, 0>(Blob<double> const *src, Blob<double> *dst, uint)
{
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void NMSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  constexpr uint MAX_ITER = 8;
  uint radius = kernel_size / 2;

  CHECK_LT(radius, MAX_ITER);
  CHECK_GE(radius, 1);

  expandNMSFilterKernel<Dtype, MAX_ITER>(bottom[0], top[0], radius);

  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void NMSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                   const vector<Blob<Dtype> *> &bottom)
{
  // not implemented
}

INSTANTIATE_LAYER_GPU_FUNCS(NMSLayer);

} // namespace ultinous
} // namespace caffe
