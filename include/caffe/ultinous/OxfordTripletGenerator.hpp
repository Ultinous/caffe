#pragma once

//#include <chrono>
#include <vector>
#include <deque>
#include <limits>
#include <list>
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
  OxfordTripletGenerator(OxfordTripletParameter const &otp, ImageClassificationModel const &icm)
    : m_icm( icm )
    , m_modelShuffler( icm.getBasicModel() )
    , m_margin(otp.margin())
    , m_marginMultiplier(otp.marginmultiplier())
    , m_featureMap(FeatureMapContainer<Dtype>::instance(otp.featuremapid()))
    , m_featureLength(otp.featurelength() )
    , m_sampledPositivePairs( otp.sampledpositivepairs() )
    , m_sampledNegatives( otp.samplednegatives() )
    , m_indexMatrix( m_sampledPositivePairs * (m_sampledNegatives+2) )
    , m_tooHardTriplets(otp.toohardtriplets())
    , m_numImagesInModel( icm.getImageNum() )
    , m_hardNegativesForClasses( icm.getBasicModel().size() )
    , m_maxExaminedNegatives( 10000 )
    , m_negativesToExamine( m_maxExaminedNegatives )
    , m_avgExaminedNegatives( double(m_maxExaminedNegatives) )
  {
    reset( );
    LOG(INFO) << "OxfordTripletGenerator - total number of positive pairs: " << m_totalRemainingPairs << std::endl;

    xorshf96_x=123456789;
    xorshf96_y=362436069;
    xorshf96_z=521288629;

    m_featureMap.resize( m_numImagesInModel );
    FeatureCollectorTripletGenerator<Dtype>::init( icm.getBasicModel() );

    m_shuffledImages.resize( m_numImagesInModel );
    for( ImageIndex i = 0; i < m_numImagesInModel; ++i )
      m_shuffledImages[i] = i;
    shuffle( m_shuffledImages.begin(), m_shuffledImages.end() );

    m_syncedFeatures.reset( new SyncedMemory( m_sampledPositivePairs*(m_sampledNegatives+2) * m_featureLength * sizeof(Dtype) ) );
    m_syncedDistances.reset( new SyncedMemory( m_sampledPositivePairs*(m_sampledNegatives+2) * sizeof(Dtype) ) );
    m_syncedFeatures->mutable_gpu_data();
    m_syncedDistances->mutable_gpu_data();
  }
private:
  typedef size_t ImageIndex;
  typedef size_t ClassIndex;
  typedef size_t SampleIndex;
  typedef ImageClassificationModel::ImageIndexes ImageIndices;
  typedef typename FeatureMap<Dtype>::FeatureVec FeatureVec;

  class PositivePair {
  public:
    ClassIndex m_classIndex;
    ImageIndex m_anchor;
    ImageIndex m_positive;
    ImageIndex m_negativeIndex;
    size_t m_examinedNegatives;

  public:
    PositivePair( ClassIndex classIndex, ImageIndex anchor, ImageIndex positive, ImageIndex negativeIndex )
      : m_classIndex(classIndex)
      , m_anchor(anchor)
      , m_positive(positive)
      , m_negativeIndex( negativeIndex )
      , m_examinedNegatives(0)
    { }
  };

  typedef std::list<PositivePair> PositivePairList;
  typedef std::list<ImageIndex> HardNegatives;
  typedef std::vector<HardNegatives> HardNegativesForClasses;
public:

  Triplet nextTriplet()
  {
    static bool featuresCollected = false;

    if( !featuresCollected )
    {
      if( m_featureMap.numFeatures() != m_numImagesInModel )
        return FeatureCollectorTripletGenerator<Dtype>::getInstance().nextTriplet();

      LOG(INFO) << "All features are collected!";
      featuresCollected = true;
    }

    if( m_prefetch.size() == 0 )
      prefetch();

    if ( m_prefetch.size() > 0 )
    {
      Triplet t = m_prefetch.front();
      m_prefetch.pop_front();
      return t;
    }

    LOG(INFO) << "No hard-triplet found!";
    return FeatureCollectorTripletGenerator<Dtype>::getInstance().nextTriplet();
  }

private:
  void prefetch( )
  {
    m_negativesToExamine = std::min( m_maxExaminedNegatives, (size_t)(2.0 * m_avgExaminedNegatives) );

    uint32_t trials = 3*(1+m_negativesToExamine / m_sampledNegatives);
    while( trials-- > 0 && m_prefetch.size() == 0 )
      doPrefetch();
  }

  void doPrefetch()
  {
    //auto start = std::chrono::high_resolution_clock::now();
    //uint64_t sumTime = 0;

    refreshPositivePairList( );

    size_t N = m_sampledPositivePairs;
    size_t M = m_sampledNegatives;

    size_t featureLength = m_featureMap.getFeatureVec( 0 ).size();
    size_t featureBytes = featureLength*sizeof(Dtype);

    Dtype * featureMatrix = (Dtype *)m_syncedFeatures->mutable_cpu_data();

    typename ImageIndices::iterator indexMatrixIt = m_indexMatrix.begin( );

    ClassIndex posClass; // In shuffledModel
    ImageIndex anchor, positive, negative;

    typename OxfordTripletGenerator<Dtype>::PositivePairList::iterator ppIt=m_positivePairList.begin();
    for( size_t i = 0; i < N; ++i, ++ppIt )
    {
      posClass = ppIt->m_classIndex;
      anchor = ppIt->m_anchor;
      positive = ppIt->m_positive;

      *indexMatrixIt++ = anchor;
      FeatureVec const &anchorVec = m_featureMap.getFeatureVec( anchor );
      memcpy( featureMatrix, &(anchorVec[0]), featureBytes );
      featureMatrix += featureLength;

      *indexMatrixIt++ = positive;
      FeatureVec const &positiveVec = m_featureMap.getFeatureVec( positive );
      memcpy( featureMatrix, &(positiveVec[0]), featureBytes );
      featureMatrix += featureLength;

      HardNegatives& hn = m_hardNegativesForClasses[ m_icm.getImageClass(anchor) ];
      size_t shownHardNegatives = 0;
      for( size_t j = 0; j < M; ++j )
      {
        if( shownHardNegatives < hn.size()
          && ppIt->m_examinedNegatives + j > m_avgExaminedNegatives/2
          && j%2
        )
        {
          ++shownHardNegatives;
          negative = hn.front();
          hn.pop_front();
          hn.push_back( negative );
        }
        else
        {
          do
          {
            negative = m_shuffledImages[ ppIt->m_negativeIndex ];

            (++ppIt->m_negativeIndex) %= m_numImagesInModel;
          } while( m_icm.getImageClass(negative) == m_icm.getImageClass(anchor) );
        }

        *indexMatrixIt++ = negative;
        FeatureVec const &negativeVec = m_featureMap.getFeatureVec( negative );
        memcpy( featureMatrix, &(negativeVec[0]), featureBytes );
        featureMatrix += featureLength;
      }
    }

    computeDistances( N, 2+M, featureLength );
    Dtype const * pDistances = (Dtype const*)m_syncedDistances->cpu_data();

    Triplet t(3);
    size_t j;

    ppIt = m_positivePairList.begin();
    for( size_t i = 0; i < N; ++i )
    {
      for( j = 0; j < M; j++ )
      {
        if( pDistances[i*(M+1)+1+j] < pDistances[i*(M+1)] + m_margin*m_marginMultiplier
          && (m_tooHardTriplets || pDistances[i*(M+1)+1+j] >= pDistances[i*(M+1)])
        )
        {
          t[0] = m_indexMatrix[i*(M+2)+0];
          t[1] = m_indexMatrix[i*(M+2)+1];
          t[2] = m_indexMatrix[i*(M+2)+2+j];
          break;
        }
      }

      ppIt->m_examinedNegatives += m_sampledNegatives;

      if( j != M )
      {
        m_prefetch.push_back(t);

        m_avgExaminedNegatives = ((m_avgExaminedNegatives*999.0)+ppIt->m_examinedNegatives)/1000.0;

        ImageIndex negative = t[2];
        HardNegatives& hn = m_hardNegativesForClasses[ m_icm.getImageClass(t[0]) ];

        if( std::find( hn.begin(), hn.end(), negative ) == hn.end() )
        {
          hn.push_back( negative );
          while( hn.size( ) > m_hardNegativePoolSize )
            hn.pop_front( );
        }
      }


      if( j != M || ppIt->m_examinedNegatives >= m_negativesToExamine )
        ppIt = m_positivePairList.erase(ppIt);
      else
        ++ppIt;
    }
    //sumTime+=std::chrono::duration_cast < std::chrono::nanoseconds > (std::chrono::high_resolution_clock::now() - start).count();
    //std::cout << (sumTime / 1000) << std::endl;
  }

  void refreshPositivePairList( )
  {
    ClassIndex posClass;
    ImageIndex anchor, positive;

    while( m_positivePairList.size() < m_sampledPositivePairs )
    {
      nextPositivePair( posClass, anchor, positive );

      m_positivePairList.push_back( PositivePair( posClass, anchor, positive
            , xorshf96()%m_numImagesInModel ) );
    }
  }

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

    // compute classIndex
    uint64_t R = myRandom( m_totalRemainingPairs );
    vector<size_t>::iterator it = m_remainingPairsInClasses.begin();
    while( R >= *it )
      R -= *it++;
    clIndex = it - m_remainingPairsInClasses.begin();

    ImageClassificationModel::ImageIndexes const &images = shuffledModel[clIndex].images;

    --m_remainingPairsInClasses[clIndex];
    --m_totalRemainingPairs;

    if( (m_totalRemainingPairs % 640000) == 0 )
      LOG(INFO) << "Current settings: m_avgExNeg: " << m_avgExaminedNegatives
          << " m_negsToEx: " << m_negativesToExamine;


    size_t pairIndex = m_remainingPairsInClasses[clIndex];

    size_t pairsInClass = images.size() * (images.size()-1);

    if( pairIndex < pairsInClass / 2 )
      pairIndex *= 2;
    else
      pairIndex = 1+2*( pairIndex - pairsInClass/2 );

    positive = pairIndex % images.size();
    anchor = (1+positive+pairIndex/images.size()) % images.size();

    anchor = images[anchor];
    positive = images[positive];
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

  uint32_t xorshf96(void) {          //period 2^96-1
    xorshf96_x ^= xorshf96_x << 16;
    xorshf96_x ^= xorshf96_x >> 5;
    xorshf96_x ^= xorshf96_x << 1;
    xorshf96_t = xorshf96_x;
    xorshf96_x = xorshf96_y;
    xorshf96_y = xorshf96_z;
    xorshf96_z = xorshf96_t ^ xorshf96_x ^ xorshf96_y;
    return xorshf96_z;
  }

  uint64_t myRandom( uint64_t N )
  {
    if( N-1 <= std::numeric_limits<uint32_t>::max() )
    {
      return uint64_t(xorshf96()) % N;
    }
    uint64_t r = 0;
    r += uint64_t(xorshf96());
    r += uint64_t(xorshf96())<<32;
    return r % N;
  }

private:
  typedef std::vector<Dtype> DistMatrix;
private:
  ImageClassificationModel const& m_icm;
  ImageClassificationModelShuffle m_modelShuffler;
  Dtype m_margin;
  Dtype m_marginMultiplier;
  FeatureMap<Dtype>& m_featureMap;
  size_t m_featureLength;
  size_t m_sampledPositivePairs;
  size_t m_sampledNegatives;

  shared_ptr<SyncedMemory> m_syncedFeatures;
  shared_ptr<SyncedMemory> m_syncedDistances;
  ImageIndices m_indexMatrix;

  bool m_tooHardTriplets;

  size_t m_numImagesInModel;

  std::deque<Triplet> m_prefetch;

  std::vector<size_t> m_remainingPairsInClasses;
  uint64_t m_totalRemainingPairs;

  PositivePairList m_positivePairList;
  HardNegativesForClasses m_hardNegativesForClasses;


  size_t m_maxExaminedNegatives;
  size_t m_negativesToExamine;
  double m_avgExaminedNegatives;

  std::vector<ImageIndex> m_shuffledImages;

  static const size_t m_prefetchTrials = 50;
  static const size_t m_hardNegativePoolSize = 32;

  /* Random generator variables */
  uint32_t xorshf96_x, xorshf96_y, xorshf96_z;
  uint32_t xorshf96_t;
};

} // namespace ultinous
} // namespace caffe
