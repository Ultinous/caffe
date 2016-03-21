#pragma once

#include <vector>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>

template <Dtype>
class HardTripletGenerator
{
public:
  typedef std::vector<size_t> Triplet;

public:
  HardTripletGenerator(size_t classes, size_t pictures, Dtype margin, const PicturesOfClasses& pictures, const std::string& featureMapName)
    : m_classes(classes)
    , m_pictures(pictures)
    , m_margin(margin)
    , m_sampler(pictures)
    , m_sample(classes, pictures)
    , m_indexInSample(0)
    , m_featureMap(FeatureMapContainer<Dtype>::instance())
  {
    resample();
  }
public:
  Triplet nextTriplet()
  {
    if(m_indexInSample / m_pictures >= m_classes)
      resample();
    size_t classIndex = m_indexInSample / m_pictures;
    size_t pictureIndex = m_indexInSample % m_pictures;

    Triplet t;
    t.push_back(m_sample.m_pictures[classIndex][pictureIndex]); // anchor
    size_t pbegin = classIndex*m_pictures;
    size_t pend = pbegin + m_pictures;
    const Vec& dvec = m_distances[m_indexInSample];
    Dtype maxpd = 0;
    size_t posIndex = 0;
    for(size_t p = pbegin; p<pend; ++p)
    {
      if(p==m_indexInSample)
        continue;
      if(dvec[p] > maxpd)
      {
        maxpd = dvec[p];
        posIndex = p % m_pictures;
      }
    }
    t.push_back(m_sample.m_pictures[classIndex][posIndex]); // hard positive
    Dtype maxdp_w_margin = maxdp+m_margin;
    Dtype minnd = std::numeric_limits<Dtype>::max();
    size_t negIndex = std::numeric_limits<size_t>::max();
    bool already_found_hard_neg = false;
    for(size_t n = 0; n<pbegin; ++n)
    {
      if(negIndex == std::numeric_limits<size_t>::max())
        negindex = n;
      if(dvec[n] >= maxdp_w_margin)
        continue; // is is not a hard negative
      if(dvec[n] < maxdp)
      {
        // it is too hard, but could be the best
        if(already_found_hard_neg)
          continue;
        // use the less hard (bigger) from the two too hards
        if(mindp < maxdp && dvec[n] > mindp)
        {
          mindp = dvec[n];
          negIndex = n;
        }
      }
      else
      {
        // it is a hard, but not too hard
        // use the hardest
        if(dvec[n]>mindp)
        {
          mindp = dvec[n];
          negindex = n;
          already_found_hard_neg = true;
        }
      }
    }
    size_t iend = m_classes*m_pictures;
    for(size_t n = pend; n<iend; ++n)
    {
      if(negIndex == std::numeric_limits<size_t>::max())
        negindex = n;
      if(dvec[n] >= maxdp_w_margin)
        continue; // is is not a hard negative
      if(dvec[n] < maxdp)
      {
        // it is too hard, but could be the best
        if(already_found_hard_neg)
          continue;
        // use the less hard (bigger) from the two too hards
        if(mindp < maxdp && dvec[n] > mindp)
        {
          mindp = dvec[n];
          negIndex = n;
        }
      }
      else
      {
        // it is a hard, but not too hard
        // use the hardest
        if(dvec[n]>mindp)
        {
          mindp = dvec[n];
          negindex = n;
          already_found_hard_neg = true;
        }
      }
    }
    t.push_back(m_sample.m_pictures[mindp/m_pictures][mindp%m_pictures]); // hard negative
    ++m_indexInSample;
  }
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
  typedef PictureSampler::Sample Sample;
private:
  size_t m_classes;
  size_t m_pictures;
  Mat m_distances;
  Dtype m_margin;
  PictureSampler m_sampler;
  Sample m_sample;
  size_t m_indexInSample;
  const FeatureMap& m_featureMap;
};
