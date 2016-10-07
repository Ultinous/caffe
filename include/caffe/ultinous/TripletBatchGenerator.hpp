#pragma once

#include <vector>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include "caffe/ultinous/AbstractTripletGenerator.hpp"
#include "caffe/ultinous/HardTripletPool.hpp"
#include "caffe/ultinous/AllTripletGenerator.hpp"
#include "caffe/ultinous/RandomTripletGenerator.hpp"


namespace caffe {
namespace ultinous {

template <typename Dtype>
class TripletBatchGenerator
{
public:
  typedef AbstractTripletGenerator::BasicModel BasicModel;

  typedef AbstractTripletGenerator::Triplet Triplet;
  typedef std::vector<Triplet> TripletBatch;

public:
  TripletBatchGenerator(size_t batchSize, const BasicModel& basicModel, const TripletDataParameter triplet_data_param)
    : m_batchSize(batchSize)
    , m_basicModel(basicModel)
    , m_triplet_data_param(triplet_data_param)
  {
    m_numImagesInModel = 0;
    for( int i = 0; i < basicModel.size(); ++i )
      m_numImagesInModel += basicModel[i].images.size();

    if( m_triplet_data_param.strategy()=="hard" )
    {
      allTripletGenerator = AllTripletGeneratorPtr(
        new AllTripletGenerator<Dtype>( m_basicModel )
      );

      hardTripletGenerator = HardTripletGeneratorPtr(
        new HardTripletGenerator<Dtype>(
          m_triplet_data_param.hard_triplet_param().sampledclasses()
          , m_triplet_data_param.hard_triplet_param().sampledpictures()
          , m_triplet_data_param.hard_triplet_param().margin()
          , m_basicModel
          , m_triplet_data_param.hard_triplet_param().featuremapid()
          , m_triplet_data_param.hard_triplet_param().toohardtriplets()
        )
      );

      hardTripletPool = HardTripletPoolPtr(
        new HardTripletPool<Dtype>(
          hardTripletGenerator
          , m_triplet_data_param.hard_triplet_param().poolsize()
        )
      );

      m_prefetchSize = 4*m_batchSize;
    }
    else if( m_triplet_data_param.strategy()=="random" )
    {
      randomTripletGenerator = RandomTripletGeneratorPtr(
        new RandomTripletGenerator<Dtype>( m_basicModel )
      );

      m_prefetchSize = 10*m_batchSize;
    }
    else
    {
      throw std::exception( );
    }
  }

  TripletBatch nextTripletBatch()
  {
    TripletBatch batch;

    if( m_prefetch.size() < m_batchSize ) {
      prefetch( );
      shuffle( );
    }

    while( batch.size() < m_batchSize ) {
      batch.push_back( m_prefetch.back( ) );
      m_prefetch.pop_back();
    }

    return batch;
  }

private:
  void shuffle( void ) {
     std::random_shuffle ( m_prefetch.begin(), m_prefetch.end() );
  }

  void prefetch( void ) {
    if( m_prefetch.size() >= m_prefetchSize )
      return;

    if( m_triplet_data_param.strategy()=="hard")
    {
      FeatureMap<Dtype>& featureMap = FeatureMapContainer<Dtype>::instance(
        m_triplet_data_param.hard_triplet_param().featuremapid()
      );

      if( featureMap.numFeatures() != m_numImagesInModel )
      {
        while(m_prefetch.size() < m_batchSize)
          m_prefetch.push_back( allTripletGenerator->nextTriplet() );

        return;
      }

      while(m_prefetch.size() < m_prefetchSize)
      {
        m_prefetch.push_back( hardTripletPool->nextTriplet() );
      }
    }

    while(m_prefetch.size() < m_prefetchSize)
    {
       m_prefetch.push_back( randomTripletGenerator->nextTriplet() );
    }
  }
private:
  const size_t m_batchSize;
  const BasicModel& m_basicModel;
  const TripletDataParameter m_triplet_data_param;

  int m_prefetchSize;
  TripletBatch m_prefetch;
  int m_numImagesInModel;

  typedef boost::shared_ptr<HardTripletGenerator<Dtype> > HardTripletGeneratorPtr;
  HardTripletGeneratorPtr hardTripletGenerator;

  typedef boost::shared_ptr<AllTripletGenerator<Dtype> > AllTripletGeneratorPtr;
  AllTripletGeneratorPtr allTripletGenerator;

  typedef boost::shared_ptr<RandomTripletGenerator<Dtype> > RandomTripletGeneratorPtr;
  RandomTripletGeneratorPtr randomTripletGenerator;

  typedef boost::shared_ptr<HardTripletPool<Dtype> > HardTripletPoolPtr;
  HardTripletPoolPtr hardTripletPool;

};

} // namespace ultinous
} // namespace caffe