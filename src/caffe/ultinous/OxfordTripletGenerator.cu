#include <algorithm>
#include <vector>

#include "caffe/syncedmem.hpp"
#include "caffe/ultinous/OxfordTripletGenerator.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
__global__ void doComputeDistancesGPU(
      size_t const N
    , size_t const M
    , size_t const featureLength
    , Dtype* features
    , Dtype* distances)
{
  CUDA_KERNEL_LOOP(index, N*(M-1)) {

    size_t i = index / (M-1);
    size_t j = index % (M-1);

    Dtype const * f1 = features+(i*M)*featureLength;
    Dtype * f2 = features+(i*M + (1+j))*featureLength;

    Dtype res = Dtype(0);
    for( size_t i = 0; i < featureLength; ++i, ++f1, ++f2 )
    {
      *f2 -= *f1;
      res += (*f2) * (*f2);
    }

    distances[index] = res;
  }
}


template <typename Dtype>
void OxfordTripletGenerator<Dtype>::computeDistancesGPU( size_t N, size_t M, size_t featureLength )
{
  Dtype * gpFeatures = (Dtype*)m_syncedFeatures->mutable_gpu_data();
  Dtype * gpDistances = (Dtype*)m_syncedDistances->mutable_gpu_data();

  doComputeDistancesGPU<Dtype><<<CAFFE_GET_BLOCKS(N*(M-1)), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, featureLength, gpFeatures, gpDistances );

  CUDA_POST_KERNEL_CHECK;
}

template void OxfordTripletGenerator<float>::computeDistancesGPU( size_t N, size_t M, size_t featureLength );
template void OxfordTripletGenerator<double>::computeDistancesGPU( size_t N, size_t M, size_t featureLength );


} // namespace ultinous
} // namespace caffe
