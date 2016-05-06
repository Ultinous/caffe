#pragma once

#include <vector>
#include <deque>
#include <boost/graph/graph_concepts.hpp>
#include <caffe/ultinous/ImageClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include <caffe/ultinous/AbstractTripletGenerator.hpp>
#include <caffe/ultinous/FeatureCollectorTripletGenerator.hpp>


namespace caffe {
namespace ultinous {

template <typename Dtype>
class OxfordTripletGenerator : public AbstractTripletGenerator
{
public:
  OxfordTripletGenerator(HardTripletParameter const &htp, BasicModel const &basicModel)
    : m_modelShuffler( basicModel )
    , m_margin(htp.margin())
    , m_featureMap(FeatureMapContainer<Dtype>::instance(htp.featuremapid()))
    , m_tooHardTriplets(htp.toohardtriplets())
  {
    reset( );
  }
private:
  typedef size_t ImageIndex;
  typedef size_t ClassIndex;
  typedef size_t SampleIndex;
  typedef ImageClassificationModel::ImageIndexes ImageIndexes;
  typedef typename FeatureMap<Dtype>::FeatureVec FeatureVec;
public:

  Triplet nextTriplet()
  {
    if( m_prefetch.size() == 0 )
      prefetch();

    Triplet t = m_prefetch.front();
    m_prefetch.pop_front();

    return t;
  }

  void prefetch()
  {
    size_t N = 1024;
    size_t M = 128;

    size_t featureLength = m_featureMap.getFeatureVec( 0 ).size();

    if( !m_syncedFeatures )
      m_syncedFeatures.reset( new SyncedMemory( N*(M+2) * featureLength * sizeof(Dtype) ) );

    std::vector<ImageIndex> indexMatrix;
    Dtype * featureMatrix = (Dtype *)m_syncedFeatures->mutable_cpu_data();

    for( size_t i = 0; i < N; ++i )
    {
      BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

      ClassIndex posClass;
      ImageIndex anchor, positive;
      nextPositivePair( posClass, anchor, positive );

      indexMatrix.push_back( anchor );
      FeatureVec const &anchorVec = m_featureMap.getFeatureVec( anchor );
      memcpy( featureMatrix+(i*(M+2))*featureLength, &(anchorVec.at(0)), featureLength*sizeof(Dtype) );

      indexMatrix.push_back( positive );
      FeatureVec const &positiveVec = m_featureMap.getFeatureVec( positive );
      memcpy( featureMatrix+(i*(M+2)+1)*featureLength, &(positiveVec.at(0)), featureLength*sizeof(Dtype) );


      for( size_t j = 0; j < M; ++j )
      {
        ClassIndex negClass = rand() % (shuffledModel.size()-1);
        if( negClass >= posClass )
          ++negClass;

        ImageIndex negative = shuffledModel[negClass].images[rand() % shuffledModel[negClass].images.size()];

        indexMatrix.push_back( negative );
        FeatureVec const &negativeVec = m_featureMap.getFeatureVec( negative );
        memcpy( featureMatrix+(i*(M+2)+2+j)*featureLength, &(negativeVec.at(0)), featureLength*sizeof(Dtype) );
      }
    }

    std::vector< std::vector<Dtype> > distMatrix = computeDistances( N, 2+M, featureLength );

    for( size_t i = 0; i < N; ++i )
    {
      Triplet t;
      for( size_t j = 0; j < M; j++ )
      {
        if( distMatrix[i][1+j] < distMatrix[i][0] + m_margin )
        {
          t.push_back( indexMatrix[i*(M+2)+0] );
          t.push_back( indexMatrix[i*(M+2)+1] );
          t.push_back( indexMatrix[i*(M+2)+2+j] );
          break;
        }
      }
      if( t.size() == 0 )
        t = FeatureCollectorTripletGenerator<Dtype>::getInstance().nextTriplet();

      m_prefetch.push_back(t);
    }
  }

private:
  void reset()
  {
    m_modelShuffler.shuffleModel();
    BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

    m_totalRemainingPairs = 0;
    m_rulettSizes.clear();

    for( size_t i = 0; i < shuffledModel.size(); i++ )
    {
      ImageClassificationModel::ClassModel const &cm = shuffledModel[i];

      size_t pairsInClass = cm.images.size() * (cm.images.size()-1);

      m_totalRemainingPairs += pairsInClass;
      m_rulettSizes.push_back( pairsInClass );
    }
  }

  void nextPositivePair( ClassIndex &clIndex, ImageIndex &anchor, ImageIndex &positive )
  {
    if( m_totalRemainingPairs == 0 )
      reset( );

    BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

    uint64_t pairIndex = bigRandom( ) % m_totalRemainingPairs;

    // compute classIndex
    clIndex = 0;
    while( pairIndex >= m_rulettSizes[clIndex] )
    {
      pairIndex -= m_rulettSizes[clIndex];
      ++clIndex;
    }

    --m_rulettSizes[clIndex];
    --m_totalRemainingPairs;

    ImageIndex nImages = shuffledModel[clIndex].images.size();
    anchor = pairIndex % nImages;
    positive = (1+anchor+pairIndex/nImages) % nImages;

    anchor = shuffledModel[clIndex].images[anchor];
    positive = shuffledModel[clIndex].images[positive];
  }

  std::vector<std::vector<Dtype> > computeDistancesGPU( size_t N, size_t M, size_t featureLength );

  std::vector<std::vector<Dtype> > computeDistancesCPU( size_t N, size_t M, size_t featureLength )
  {
    throw std::exception(); // TODO
    /*CHECK_GT( features.size(), 0 );
    CHECK_GT( features[0].size(), 0 );
    FeatureVec sqr( features[0][0].size() );

    std::vector<std::vector<Dtype> > results;

    for( size_t c = 0; c < features.size(); ++c )
    {
      std::vector<Dtype> row;
      for( size_t i = 1; i < features.size(); ++i )
      {
        FeatureVec const &f1 = features[c*M+0];
        FeatureVec const &f2 = features[c*M+i];

        CHECK_GT( f1.size(), 0 );
        CHECK_EQ( f1.size(), f2.size() );

        caffe_sub( f1.size(), &(f1[0]), &(f2[0]), &(sqr[0]) );
        Dtype dist = caffe_cpu_dot( sqr.size(), &(sqr[0]), &(sqr[0]) );

        row.push_back( dist );
      }
      results.push_back(row);
    }

    return results;*/
  }

  std::vector<std::vector<Dtype> > computeDistances( size_t N, size_t M, size_t featureLength )
  {
    #ifdef CPU_ONLY
    return computeDistancesCPU(N, M, featureLength );
    #else
    return computeDistancesGPU(N, M, featureLength );
    #endif
  }

  uint64_t bigRandom( )
  {
    uint64_t r = 0;
    r += uint64_t(rand() % 65536);
    r += uint64_t(rand() % 65536)<<16;
    r += uint64_t(rand() % 65536)<<32;
    r += uint64_t(rand() % 65536)<<48;
    return r;
  }

private:
  typedef std::vector<Dtype> Vec;
  typedef std::vector<Vec> Mat;
  typedef ImageSampler::Sample Sample;
private:
  ImageClassificationModelShuffle m_modelShuffler;
  Dtype m_margin;
  const FeatureMap<Dtype>& m_featureMap;
  bool m_tooHardTriplets;

  shared_ptr<SyncedMemory> m_syncedFeatures;
  shared_ptr<SyncedMemory> m_syncedDistances;

  std::deque<Triplet> m_prefetch;

  vector<size_t> m_rulettSizes;
  uint64_t m_totalRemainingPairs;
};

} // namespace ultinous
} // namespace caffe
