#pragma once

#include <vector>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>

namespace caffe {
namespace ultinous {

template <typename Dtype>
class HardTripletGenerator
{
public:
  typedef std::vector<size_t> Triplet;
  typedef ImageClassificationModel::BasicModel BasicModel;

public:
  HardTripletGenerator(size_t classes, size_t pictures, Dtype margin, const BasicModel& pictureOfClasses, const std::string& featureMapName)
    : m_classesInSample(classes)
    , m_imagesInSampleClass(pictures)
    , m_margin(margin)
    , m_sampler(pictureOfClasses)
    , m_sample(classes, pictures)
    , m_indexInSample(0)
    , m_featureMap(FeatureMapContainer<Dtype>::instance(featureMapName))
  {
    resample();
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

    t.push_back(image(m_indexInSample)); // anchor

    SampleIndex posSampleBegin = classIndex(m_indexInSample)*m_imagesInSampleClass;
    SampleIndex posSampleEnd = posSampleBegin + m_imagesInSampleClass;
    const Vec& dvec = m_distances[m_indexInSample];
    
    Dtype maxPosDistance = 0;
    SampleIndex maxPosIndex = 0;
    for(size_t posSample = posSampleBegin; posSample<posSampleEnd; ++posSample)
    {
      if(posSample==m_indexInSample)
        continue;
      if(dvec[posSample] > maxPosDistance)
      {
        maxPosDistance = dvec[posSample];
        maxPosIndex = posSample;
      }
    }

    t.push_back(image(maxPosIndex)); // hard positive


    Dtype maxPosDistanceWithMargin = maxPosDistance+m_margin;

    Dtype closeNegDistance = std::numeric_limits<Dtype>::max();
    size_t closeNegIndex = std::numeric_limits<size_t>::max();
    
    bool already_found_hard_neg = false;

    for(size_t negSample = 0; negSample<posSampleBegin; ++negSample)
    {
      if(closeNegIndex == std::numeric_limits<size_t>::max())
        closeNegIndex = negSample; // the first negative selected as "random"
      if(dvec[negSample] >= maxPosDistanceWithMargin)
        continue; // is is not a hard negative
      if(dvec[negSample] < maxPosDistance)
      {
        // it is too hard, but could be the best
        if(already_found_hard_neg)
          continue;
        // use the less hard (bigger) from the two too hards
        if(closeNegDistance < maxPosDistance && dvec[negSample] > closeNegDistance)
        {
          closeNegDistance = dvec[negSample];
          closeNegIndex = negSample;
        }
      }
      else
      {
        // it is a hard, but not too hard
        // use the hardest
        if(dvec[negSample]<closeNegDistance)
        {
          closeNegDistance = dvec[negSample];
          closeNegIndex = negSample;
          already_found_hard_neg = true;
        }
      }
    }
    size_t endSample = m_classesInSample*m_imagesInSampleClass;
    for(size_t negSample = posSampleEnd; negSample<endSample; ++negSample)
    {
      if(closeNegIndex == std::numeric_limits<size_t>::max())
        closeNegIndex = negSample; // the first negative selected as "random"
      if(dvec[negSample] >= maxPosDistanceWithMargin)
        continue; // is is not a hard negative
      if(dvec[negSample] < maxPosDistance)
      {
        // it is too hard, but could be the best
        if(already_found_hard_neg)
          continue;
        // use the less hard (bigger) from the two too hards
        if(closeNegDistance < maxPosDistance && dvec[negSample] > closeNegDistance)
        {
          closeNegDistance = dvec[negSample];
          closeNegIndex = negSample;
        }
      }
      else
      {
        // it is a hard, but not too hard
        // use the hardest
        if(dvec[negSample]<closeNegDistance)
        {
          closeNegDistance = dvec[negSample];
          closeNegIndex = negSample;
          already_found_hard_neg = true;
        }
      }
    }

    t.push_back(image(closeNegIndex)); // hard negative

    ++m_indexInSample;
    return t;
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
    m_indexInSample = 0;
  }
  void recalcDistances()
  {
    // TODO here the m_distances matrix must be calculated
    // m_sample and m_featureMap will be used here
  }
private:
  typedef std::vector<Dtype> Vec;
  typedef std::vector<Vec> Mat;
  typedef ImageSampler::Sample Sample;
private:
  size_t m_classesInSample;
  size_t m_imagesInSampleClass;
  Mat m_distances;
  Dtype m_margin;
  ImageSampler m_sampler;
  Sample m_sample;
  SampleIndex m_indexInSample;
  const FeatureMap<Dtype>& m_featureMap;
};

} // namespace ultinous
} // namespace caffe
