#pragma once

#include <vector>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include "caffe/ultinous/AbstractTripletGenerator.hpp"
#include "caffe/ultinous/HardTripletGenerator.hpp"
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

    allTripletGenerator = AllTripletGeneratorPtr(
      new AllTripletGenerator<Dtype>( m_basicModel )
    );

    randomTripletGenerator = RandomTripletGeneratorPtr(
      new RandomTripletGenerator<Dtype>( m_basicModel )
    );

    if( m_triplet_data_param.strategy()=="hard" )
    {
      string featureMapId = m_triplet_data_param.featuremapid();
      uint32_t sampledClasses = m_triplet_data_param.sampledclasses();
      uint32_t sampledPictures = m_triplet_data_param.sampledpictures();
      Dtype margin = m_triplet_data_param.margin();

      hardTripletGenerator = HardTripletGeneratorPtr(
        new HardTripletGenerator<Dtype>(
          sampledClasses
          , sampledPictures, margin
          , m_basicModel
          , featureMapId )
      );
    }

    if( m_triplet_data_param.strategy()=="hard" )
    {
      uint32_t sampledClasses = m_triplet_data_param.sampledclasses();
      uint32_t sampledPictures = m_triplet_data_param.sampledpictures();
      m_prefetchSize = 5*sampledClasses*sampledPictures;
    }
    else
    {
      m_prefetchSize = 10*m_batchSize;
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

    FeatureMap<Dtype>& featureMap = FeatureMapContainer<Dtype>::instance(
      m_triplet_data_param.featuremapid()
    );
    if( m_triplet_data_param.strategy()=="hard" && featureMap.numFeatures() != m_numImagesInModel )
    {
      while(m_prefetch.size() < m_prefetchSize)
        m_prefetch.push_back( allTripletGenerator->nextTriplet() );

      return;
    }

    while(m_prefetch.size() < m_prefetchSize)
    {
      Triplet t;

      if( m_triplet_data_param.strategy()=="hard" )
      {
        t = hardTripletGenerator->nextTriplet();
      }
      else
      {
        t = randomTripletGenerator->nextTriplet();
      }

       m_prefetch.push_back( t );
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
};

} // namespace ultinous
} // namespace caffe
