#include <algorithm>
#include <vector>

#include "caffe/syncedmem.hpp"
#include "caffe/ultinous/HardTripletGenerator.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
__global__ void doCalcDistancesGPU(
      int const nSample
    , int const featureLength
    , Dtype const* features
    , Dtype* distances)
{
  CUDA_KERNEL_LOOP(index, nSample*nSample) {
    int ix1 = index/nSample;
    int ix2 = index%nSample;

    Dtype const * f1 = features + ix1*featureLength;
    Dtype const * f2 = features + ix2*featureLength;

    Dtype res = Dtype(0);
    for( size_t i = 0; i < featureLength; ++i, ++f1, ++f2 )
      res += (*f1-*f2) * (*f1-*f2);

    distances[index] = res;
  }
}


template <typename Dtype>
void HardTripletGenerator<Dtype>::recalcDistancesGPU( ) {
  size_t const nSample = m_classesInSample * m_imagesInSampleClass;
  size_t const featureLength = m_featureMap.getFeatureVec( image(0) ).size();

  SyncedMemory features( nSample * featureLength * sizeof(Dtype) );
  SyncedMemory distances( nSample * nSample * sizeof(Dtype) );

  Dtype * pFeatures = (Dtype *) features.cpu_data();

  for( size_t i = 0; i < nSample; i++ )
  {
    const typename FeatureMap<Dtype>::FeatureVec& feat = m_featureMap.getFeatureVec( image(i) );
    CHECK_GT( feat.size(), 0);

    memcpy( pFeatures+i*featureLength, &(feat.at(0)), featureLength );
  }

  Dtype * gpFeatures = (Dtype*)features.gpu_data();
  Dtype * gpDistances = (Dtype*)distances.mutable_gpu_data();

  doCalcDistancesGPU<Dtype><<<CAFFE_GET_BLOCKS(nSample*nSample), CAFFE_CUDA_NUM_THREADS>>>(
      nSample*nSample, featureLength, gpFeatures, gpDistances );

  CUDA_POST_KERNEL_CHECK;

  Dtype const * pDistances = (Dtype const*)distances.cpu_data();

  for( size_t i = 0; i < nSample; i++ )
    for( size_t j = 0; j < nSample; j++ )
      m_distances[i][j] = *(pDistances+i*nSample+j);
}

template void HardTripletGenerator<float>::recalcDistancesGPU( );


} // namespace ultinous
} // namespace caffe
