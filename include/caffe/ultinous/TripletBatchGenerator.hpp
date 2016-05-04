#pragma once

#include <vector>
#include <caffe/ultinous/ImageClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include "caffe/ultinous/AbstractTripletGenerator.hpp"
#include "caffe/ultinous/HardTripletPool.hpp"
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
  TripletBatchGenerator(size_t batchSize, const BasicModel& basicModel, const TripletDataParameter& triplet_data_param)
    : m_batchSize(batchSize)
    , m_basicModel(basicModel)
    , m_triplet_data_param(triplet_data_param)
    , m_iteration(0)
  {
    m_numImagesInModel = 0;
    for( int i = 0; i < m_basicModel.size(); ++i )
      m_numImagesInModel += m_basicModel[i].images.size();

    if( m_triplet_data_param.strategy()=="hard" )
    {
      FeatureCollectorTripletGenerator<Dtype>::init( m_basicModel );

      hardTripletGenerator = HardTripletGeneratorPtr(
        new HardTripletGenerator<Dtype>(
          m_triplet_data_param.hard_triplet_param()
          , m_basicModel
        )
      );

      hardTripletPool = HardTripletPoolPtr(
        new HardTripletPool<Dtype>(
          hardTripletGenerator
          , m_triplet_data_param.hard_triplet_param().hard_triplet_pool_param()
        )
      );

      m_prefetchSize = 1*m_batchSize;
    }
    else if( m_triplet_data_param.strategy()=="random" )
    {
      randomTripletGenerator = RandomTripletGeneratorPtr(
        new RandomTripletGenerator<Dtype>( m_basicModel )
      );

      m_prefetchSize = 1*m_batchSize;
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

    ++m_iteration;

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
      static bool featuresCollected = false;

      FeatureMap<Dtype>& featureMap = FeatureMapContainer<Dtype>::instance(
        m_triplet_data_param.hard_triplet_param().featuremapid()
      );

      if( !featuresCollected && featureMap.numFeatures() != m_numImagesInModel )
      {
        while(m_prefetch.size() < m_batchSize)
          m_prefetch.push_back( FeatureCollectorTripletGenerator<Dtype>::getInstance().nextTriplet() );

        return;
      }

      if( !featuresCollected )
      {
        LOG(INFO) << "All features are collected!";
        featuresCollected = true;
      }

      while(m_prefetch.size() < m_prefetchSize)
      {
        m_prefetch.push_back( hardTripletPool->nextTriplet(m_iteration) );
      }
    }
    else if( m_triplet_data_param.strategy()=="random" )
    {
      while(m_prefetch.size() < m_prefetchSize)
      {
        m_prefetch.push_back( randomTripletGenerator->nextTriplet() );
      }
    }
    else
    {
      throw std::exception( );
    }

  }
private:
  const size_t m_batchSize;
  const BasicModel& m_basicModel;
  const TripletDataParameter m_triplet_data_param;

  int m_prefetchSize;
  TripletBatch m_prefetch;
  int m_numImagesInModel;
  uint64_t m_iteration;

  typedef boost::shared_ptr<HardTripletGenerator<Dtype> > HardTripletGeneratorPtr;
  HardTripletGeneratorPtr hardTripletGenerator;

  typedef boost::shared_ptr<RandomTripletGenerator<Dtype> > RandomTripletGeneratorPtr;
  RandomTripletGeneratorPtr randomTripletGenerator;

  typedef boost::shared_ptr<HardTripletPool<Dtype> > HardTripletPoolPtr;
  HardTripletPoolPtr hardTripletPool;

};

} // namespace ultinous
} // namespace caffe
