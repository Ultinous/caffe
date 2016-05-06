#pragma once

#include <vector>
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
public:

  Triplet nextTriplet()
  {
    if( m_totalRemainingPairs == 0 )
      reset( );

    ClassIndex posClass;
    ImageIndex anchor, positive;
    nextPositivePair( posClass, anchor, positive );


  /*  BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();
    ClassIndex negClass = rand() % (shuffledModel.size()-1);
    if( negClass >= posClass )
      ++negClass;

    ImageIndex negative = shuffledModel[negClass].images[rand() % shuffledModel[negClass].images.size()];

    Triplet t;
      t.push_back( anchor );
      t.push_back( positive );
      t.push_back( negative );

      getNegative( posClass, anchor, positive, negative );

      t[2] = negative ;
*/
//      return t;

    ImageIndex negative;
    if( getNegative( posClass, anchor, positive, negative ) )
    {
      Triplet t;

      t.push_back( anchor );
      t.push_back( positive );
      t.push_back( negative );

      return t;
    }

    return FeatureCollectorTripletGenerator<Dtype>::getInstance().nextTriplet();
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

  bool getNegative( ClassIndex posClass, ImageIndex anchor, ImageIndex positive, ImageIndex &negative )
  {
    BasicModel const &shuffledModel = m_modelShuffler.shuffledModel();

    Dtype posDistance = computeDistance( anchor, positive );

    for( size_t i = 0; i < 1024; ++i )
    {
      ClassIndex negClass = rand() % (shuffledModel.size()-1);
      if( negClass >= posClass )
        ++negClass;

      negative = shuffledModel[negClass].images[rand() % shuffledModel[negClass].images.size()];

      Dtype negDistance = computeDistance( anchor, negative );

      if( negDistance < posDistance + m_margin )
        return true;
    }

    return false;
  }

  Dtype computeDistance( ImageIndex t1, ImageIndex t2 )
  {
    const typename FeatureMap<Dtype>::FeatureVec& f1 = m_featureMap.getFeatureVec( t1 );
    const typename FeatureMap<Dtype>::FeatureVec& f2 = m_featureMap.getFeatureVec( t2 );

    CHECK_GT( f1.size(), 0 );
    CHECK_EQ( f1.size(), f2.size() );

    typename FeatureMap<Dtype>::FeatureVec sqr( f1.size() );

    caffe_sub( f1.size(), &(f1[0]), &(f2[0]), &(sqr[0]) );
    Dtype dist = caffe_cpu_dot( sqr.size(), &(sqr[0]), &(sqr[0]) );

    return dist;
  }



  /*void recalcDistancesGPU(); // src/caffe/ultinous/HardTripletGenerator.cu

  void recalcDistancesCPU()
  {
  }
  void recalcDistances()
  {
    #ifdef CPU_ONLY
    recalcDistancesCPU();
    #else
    recalcDistancesGPU();
    #endif
  }*/

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

  vector<size_t> m_rulettSizes;
  uint64_t m_totalRemainingPairs;
};

} // namespace ultinous
} // namespace caffe
