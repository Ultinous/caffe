#include <algorithm>
#include <vector>

#include "caffe/syncedmem.hpp"
#include "caffe/ultinous/HardTripletGenerator.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
__global__ void doCalcDistancesGPU(
      size_t const nSample
    , size_t const featureLength
    , Dtype const* features
    , Dtype* distances)
{
  CUDA_KERNEL_LOOP(index, nSample*nSample) {
    size_t ix1 = index/nSample;
    size_t ix2 = index%nSample;

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

  if( !m_syncedFeatures )
    m_syncedFeatures.reset( new SyncedMemory( nSample * featureLength * sizeof(Dtype) ) );
  if( !m_syncedDistances )
    m_syncedDistances.reset( new SyncedMemory( nSample * nSample * sizeof(Dtype) ) );

  Dtype * pFeatures = (Dtype *) m_syncedFeatures->mutable_cpu_data();

  for( size_t i = 0; i < nSample; i++ )
  {
    const typename FeatureMap<Dtype>::FeatureVec& feat = m_featureMap.getFeatureVec( image(i) );
    CHECK_GT( feat.size(), 0);

    memcpy( pFeatures+i*featureLength, &(feat.at(0)), featureLength*sizeof(Dtype) );
  }

  Dtype * gpFeatures = (Dtype*)m_syncedFeatures->gpu_data();
  Dtype * gpDistances = (Dtype*)m_syncedDistances->mutable_gpu_data();

  doCalcDistancesGPU<Dtype><<<CAFFE_GET_BLOCKS(nSample*nSample), CAFFE_CUDA_NUM_THREADS,0,Caffe::cuda_stream()>>>(
      nSample, featureLength, gpFeatures, gpDistances );

  CUDA_POST_KERNEL_CHECK;

  Dtype const * pDistances = (Dtype const*)m_syncedDistances->cpu_data();

  for( size_t i = 0; i < nSample; i++ )
    for( size_t j = 0; j < nSample; j++ )
      m_distances[i][j] = *(pDistances+i*nSample+j);
}

template void HardTripletGenerator<float>::recalcDistancesGPU( );
template void HardTripletGenerator<double>::recalcDistancesGPU( );


} // namespace ultinous
} // namespace caffe
