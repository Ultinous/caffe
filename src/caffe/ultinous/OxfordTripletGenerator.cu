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
    , Dtype const* features
    , Dtype* distances)
{
  CUDA_KERNEL_LOOP(index, N*(M-1)) {

    size_t i = index / (M-1);
    size_t j = index % (M-1);

    Dtype const * f1 = features+(i*M)*featureLength;
    Dtype const * f2 = features+(i*M + (1+j))*featureLength;

    Dtype res = Dtype(0);
    for( size_t i = 0; i < featureLength; ++i, ++f1, ++f2 )
      res += (*f1-*f2) * (*f1-*f2);

    distances[index] = res;
  }
}


template <typename Dtype>
std::vector<std::vector<Dtype> > OxfordTripletGenerator<Dtype>::computeDistancesGPU( size_t N, size_t M, size_t featureLength )
{
  if( !m_syncedDistances )
    m_syncedDistances.reset( new SyncedMemory( N*(M-1) * sizeof(Dtype) ) );

  Dtype * gpFeatures = (Dtype*)m_syncedFeatures->gpu_data();
  Dtype * gpDistances = (Dtype*)m_syncedDistances->mutable_gpu_data();

  doComputeDistancesGPU<Dtype><<<CAFFE_GET_BLOCKS(N*(M-1)), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, featureLength, gpFeatures, gpDistances );

  CUDA_POST_KERNEL_CHECK;

  Dtype const * pDistances = (Dtype const*)m_syncedDistances->cpu_data();

  std::vector< std::vector<Dtype> > results;

  for( size_t i = 0; i < N; ++i )
  {
    std::vector<Dtype> row;
    for( size_t j = 0; j < M-1; ++j )
      row.push_back( *(pDistances+i*(M-1)+j) );
    results.push_back( row );
  }

  return results;
}

template std::vector<std::vector<float> > OxfordTripletGenerator<float>::computeDistancesGPU( size_t N, size_t M, size_t featureLength );


} // namespace ultinous
} // namespace caffe
