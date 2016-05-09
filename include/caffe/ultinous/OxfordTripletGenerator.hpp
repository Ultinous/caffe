#pragma once

#include <cstdlib>
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
  OxfordTripletGenerator(OxfordTripletParameter const &otp, BasicModel const &basicModel)
    : m_modelShuffler( basicModel )
    , m_margin(otp.margin())
    , m_featureMap(FeatureMapContainer<Dtype>::instance(otp.featuremapid()))
    , m_sampledPositivePairs( otp.sampledpositivepairs() )
    , m_sampledNegatives( otp.samplednegatives() )
    , m_indexMatrix( m_sampledPositivePairs * (m_sampledNegatives+2) )
  {
    reset( );
    LOG(INFO) << "OxfordTripletGenerator - total number of positive pairs: " << m_totalRemainingPairs << std::endl;
  }
private:
  typedef size_t ImageIndex;
  typedef size_t ClassIndex;
  typedef size_t SampleIndex;
  typedef ImageClassificationModel::ImageIndexes ImageIndices;
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
    size_t N = m_sampledPositivePairs;
    size_t M = m_sampledNegatives;

    size_t featureLength = m_featureMap.getFeatureVec( 0 ).size();

    if( !m_syncedFeatures )
      m_syncedFeatures.reset( new SyncedMemory( m_sampledPositivePairs*(m_sampledNegatives+2) * featureLength * sizeof(Dtype) ) );
    if( !m_syncedDistances )
      m_syncedDistances.reset( new SyncedMemory( m_sampledPositivePairs*(m_sampledNegatives+1) * sizeof(Dtype) ) );

    Dtype * featureMatrix = (Dtype *)m_syncedFeatures->mutable_cpu_data();

    typename ImageIndices::iterator indexMatrixIt = m_indexMatrix.begin( );

    ClassIndex posClass, negClass;
    ImageIndex anchor, positive, negative;

    for( size_t i = 0; i < N; ++i )
    {
      BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

      nextPositivePair( posClass, anchor, positive );

      *indexMatrixIt++ = anchor;
      FeatureVec const &anchorVec = m_featureMap.getFeatureVec( anchor );
      memcpy( featureMatrix+(i*(M+2))*featureLength, &(anchorVec.at(0)), featureLength*sizeof(Dtype) );

      *indexMatrixIt++ = positive;
      FeatureVec const &positiveVec = m_featureMap.getFeatureVec( positive );
      memcpy( featureMatrix+(i*(M+2)+1)*featureLength, &(positiveVec.at(0)), featureLength*sizeof(Dtype) );

      for( size_t j = 0; j < M; ++j )
      {
        negClass = rand() % (shuffledModel.size()-1);
        if( negClass >= posClass )
          ++negClass;

        negative = shuffledModel[negClass].images[rand() % shuffledModel[negClass].images.size()];

        *indexMatrixIt++ = negative;
        FeatureVec const &negativeVec = m_featureMap.getFeatureVec( negative );
        memcpy( featureMatrix+(i*(M+2)+2+j)*featureLength, &(negativeVec.at(0)), featureLength*sizeof(Dtype) );
      }
    }

    computeDistances( N, 2+M, featureLength );
    Dtype const * pDistances = (Dtype const*)m_syncedDistances->cpu_data();

    Triplet t(3);
    size_t j;
    for( size_t i = 0; i < N; ++i )
    {
      for( j = 0; j < M; j++ )
      {
        if( pDistances[i*(M+1)+1+j] < pDistances[i*(M+1)] + m_margin )
        {
          t[0] = m_indexMatrix[i*(M+2)+0];
          t[1] = m_indexMatrix[i*(M+2)+1];
          t[2] = m_indexMatrix[i*(M+2)+2+j];
          break;
        }
      }
      if( j == M )
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
    m_remainingPairsInClasses.clear();

    for( size_t i = 0; i < shuffledModel.size(); i++ )
    {
      ImageClassificationModel::ClassModel const &cm = shuffledModel[i];

      size_t pairsInClass = cm.images.size() * (cm.images.size()-1);

      m_totalRemainingPairs += pairsInClass;
      m_remainingPairsInClasses.push_back( pairsInClass );
    }
  }

  void nextPositivePair( ClassIndex &clIndex, ImageIndex &anchor, ImageIndex &positive )
  {
    if( m_totalRemainingPairs == 0 )
      reset( );

    BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

    uint64_t pairIndex = myRandom( m_totalRemainingPairs );

    // compute classIndex
    clIndex = 0;
    while( pairIndex >= m_remainingPairsInClasses[clIndex] )
    {
      pairIndex -= m_remainingPairsInClasses[clIndex];
      ++clIndex;
    }

    --m_remainingPairsInClasses[clIndex];
    --m_totalRemainingPairs;

    ImageIndex nImages = shuffledModel[clIndex].images.size();
    anchor = pairIndex % nImages;
    positive = (1+anchor+pairIndex/nImages) % nImages;

    anchor = shuffledModel[clIndex].images[anchor];
    positive = shuffledModel[clIndex].images[positive];
  }

  void computeDistancesGPU( size_t N, size_t M, size_t featureLength );

  void computeDistancesCPU( size_t N, size_t M, size_t featureLength )
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

  void computeDistances( size_t N, size_t M, size_t featureLength )
  {
    #ifdef CPU_ONLY
    computeDistancesCPU(N, M, featureLength );
    #else
    computeDistancesGPU(N, M, featureLength );
    #endif
  }

  uint64_t myRandom( uint64_t N )
  {
    if( N-1 <= RAND_MAX )
    {
      return uint64_t(rand()) % N;
    }

    uint64_t r = 0;
    r += uint64_t(rand() % 65536);
    r += uint64_t(rand() % 65536)<<16;
    r += uint64_t(rand() % 65536)<<32;
    r += uint64_t(rand() % 65536)<<48;
    return r;
  }

private:
  typedef std::vector<Dtype> DistMatrix;
private:
  ImageClassificationModelShuffle m_modelShuffler;
  Dtype m_margin;
  const FeatureMap<Dtype>& m_featureMap;
  size_t m_sampledPositivePairs;
  size_t m_sampledNegatives;

  shared_ptr<SyncedMemory> m_syncedFeatures;
  shared_ptr<SyncedMemory> m_syncedDistances;
  ImageIndices m_indexMatrix;

  std::deque<Triplet> m_prefetch;

  vector<size_t> m_remainingPairsInClasses;
  uint64_t m_totalRemainingPairs;
};

} // namespace ultinous
} // namespace caffe
