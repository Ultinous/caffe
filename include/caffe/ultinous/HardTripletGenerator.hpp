#pragma once

#include <vector>
#include <boost/graph/graph_concepts.hpp>
#include <caffe/ultinous/ImageClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include <caffe/ultinous/AbstractTripletGenerator.hpp>

namespace caffe {
namespace ultinous {

template <typename Dtype>
class HardTripletGenerator : public AbstractTripletGenerator
{
public:
  HardTripletGenerator(HardTripletParameter const &htp, BasicModel const &basicModel)
    : m_sampler(basicModel)
    , m_classesInSample(htp.sampledclasses())
    , m_imagesInSampleClass(htp.sampledpictures())
    , m_margin(htp.margin())
    , m_indexInSample(m_classesInSample*m_imagesInSampleClass)
    , m_featureMap(FeatureMapContainer<Dtype>::instance(htp.featuremapid()))
    , m_tooHardTriplets(htp.toohardtriplets())
    , m_hardestPositive(htp.hardestpositive())
    , m_hardestNegative(htp.hardestnegative())
    , m_isLastTripletHard(false)
  {
    CHECK_GT( m_classesInSample, 0 );
    CHECK_GT( m_imagesInSampleClass, 0 );
    ImageSampler::initSample( m_classesInSample, m_imagesInSampleClass, m_sample );

    for( size_t i = 0; i < m_classesInSample*m_imagesInSampleClass; ++i)
      m_shuffle.push_back(i);
  }
private:
  typedef size_t ImageIndex;
  typedef size_t ClassIndex;
  typedef size_t SampleIndex;
  typedef ImageClassificationModel::ImageIndexes ImageIndexes;
public:

  Triplet nextTriplet()
  {
    if(classIndex(m_indexInSample) >= m_classesInSample)
      resample();

    Triplet t;
    m_isLastTripletHard = false;

    SampleIndex anchorIndex = m_shuffle[m_indexInSample];

    t.push_back(image(anchorIndex)); // anchor

    const Vec& dvec = m_distances[anchorIndex];
    SampleIndex posSampleBegin = classIndex(anchorIndex)*m_imagesInSampleClass;
    SampleIndex posSampleEnd = posSampleBegin + m_imagesInSampleClass;
    SampleIndex posIndex = 0;

    if( m_hardestPositive )
    {
      Dtype maxPosDistance = 0;
      for(size_t posSample = posSampleBegin; posSample<posSampleEnd; ++posSample)
      {
        if(posSample==anchorIndex)
          continue;
        if(dvec[posSample] > maxPosDistance)
        {
          maxPosDistance = dvec[posSample];
          posIndex = posSample;
        }
      }
    }
    else
    {
      posIndex = 1+anchorIndex;
      if( (posIndex % m_imagesInSampleClass) == 0 )
        posIndex = posSampleBegin;
    }

    t.push_back(image(posIndex)); // hard positive

    Dtype posDistance = dvec[posIndex];
    Dtype posDistanceWithMargin = posDistance+m_margin;

    size_t nSample = m_classesInSample*m_imagesInSampleClass;
    Dtype closeNegDistance = std::numeric_limits<Dtype>::max();
    size_t negIndex = std::numeric_limits<size_t>::max();

    std::vector<size_t> hardNegIndices;

    for(size_t negSample = 0; negSample<nSample; ++negSample)
    {
      if( negSample == posSampleBegin )
      {
        negSample = posSampleEnd-1;
        continue;
      }

      if(negIndex == std::numeric_limits<size_t>::max())
        negIndex = negSample; // the first negative selected as "random"

      if( dvec[negSample] >= posDistance && dvec[negSample] < posDistanceWithMargin )
      {
          m_isLastTripletHard = true;
          if( m_hardestNegative )
          {
            if( dvec[negSample] < closeNegDistance )
            {
              closeNegDistance = dvec[negSample];
              negIndex = negSample;
            }
          }
          else
          {
            hardNegIndices.push_back( negSample );
          }
      }
    }

    if( m_tooHardTriplets && !m_isLastTripletHard )
    {
      closeNegDistance = 0;
      for(size_t negSample = 0; negSample<nSample; ++negSample)
      {
        if( negSample == posSampleBegin )
        {
          negSample = posSampleEnd-1;
          continue;
        }

        if( dvec[negSample] < posDistance )
        {
          m_isLastTripletHard = true;
          if( m_hardestNegative )
          {
            if( dvec[negSample] > closeNegDistance )
            {
              closeNegDistance = dvec[negSample];
              negIndex = negSample;
            }
          }
          else
          {
            hardNegIndices.push_back( negSample );
          }
        }
      }
    }

    if( !m_hardestNegative )
    {
      if( hardNegIndices.size() > 0 )
      {
        negIndex = hardNegIndices[rand()%hardNegIndices.size()];
        m_isLastTripletHard = true;
      }
      else
      {
        negIndex = rand()%(nSample-m_imagesInSampleClass);
        if( negIndex >= posSampleBegin )
          negIndex += m_imagesInSampleClass;
        m_isLastTripletHard = false;
      }
    }

    t.push_back(image(negIndex)); // hard negative

    ++m_indexInSample;

    return t;
  }

  bool isLastTripletHard( ) const
  {
    return m_isLastTripletHard;
  }

  const FeatureMap<Dtype>& getFeatureMap( ) const
  {
    return m_featureMap;
  }

  Dtype getMargin( ) const
  {
    return m_margin;
  }
private:
  ClassIndex classIndex(SampleIndex idx) const { return idx/m_imagesInSampleClass; }
  ImageIndex imageIndex(SampleIndex idx) const { return idx%m_imagesInSampleClass; }
  const ImageIndexes& images(SampleIndex idx) const { return m_sample[classIndex(idx)].images; }
  ImageIndex image(SampleIndex idx) const { return images(idx)[imageIndex(idx)]; }
private:
  void resample()
  {
    m_sampler.sample(m_sample);
    recalcDistances();
    shuffle( m_shuffle.begin(), m_shuffle.end() );
    m_indexInSample = 0;
  }
  void recalcDistances()
  {
    size_t const nSample = m_classesInSample * m_imagesInSampleClass;

    CHECK_GT( nSample, 0 );

    if( m_distances.size() != nSample )
      m_distances = Mat( nSample, Vec(nSample, 0) );

    CHECK_EQ( m_distances.size(), nSample );
    for( size_t i = 0; i < nSample; i++ )
      CHECK_EQ( m_distances[i].size(), nSample );

    typename FeatureMap<Dtype>::FeatureVec sqr;

    for( size_t i = 0; i < nSample-1; i++ )
    {
      const typename FeatureMap<Dtype>::FeatureVec& feat1 = m_featureMap.getFeatureVec( image(i) );
      CHECK_GT( feat1.size(), 0);

      sqr.resize( feat1.size() );

      for( size_t j = i+1; j < nSample; j++ )
      {
        const typename FeatureMap<Dtype>::FeatureVec& feat2 = m_featureMap.getFeatureVec( image(j) );

        CHECK_GT( feat2.size(), 0 );
        CHECK_EQ( feat1.size(), feat2.size() );

        caffe_sub( feat1.size(), &(feat1[0]), &(feat2[0]), &(sqr[0]) );
        Dtype dist = caffe_cpu_dot( sqr.size(), &(sqr[0]), &(sqr[0]) );

        m_distances[i][j] = dist;
        m_distances[j][i] = dist;
      }
    }
  }
private:
  typedef std::vector<Dtype> Vec;
  typedef std::vector<Vec> Mat;
  typedef ImageSampler::Sample Sample;
private:
  ImageSampler m_sampler;
  size_t m_classesInSample;
  size_t m_imagesInSampleClass;
  Mat m_distances;
  Dtype m_margin;
  Sample m_sample;
  SampleIndex m_indexInSample;
  const FeatureMap<Dtype>& m_featureMap;
  bool m_tooHardTriplets;
  bool m_hardestPositive;
  bool m_hardestNegative;
  bool m_isLastTripletHard;

  std::vector<SampleIndex> m_shuffle;
};

} // namespace ultinous
} // namespace caffe
