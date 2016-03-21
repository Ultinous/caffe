#pragma once

#include <vector>
#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

namespace caffe
{
namespace ultinous
{

template <typename Dtype>
class FeatureMap
{
public:
  typedef size_t Index;
  typedef std::vector<Dtype> FeatureVec;
  typedef std::map<Index, FeatureVec> FeatureVecMap;
public:
  virtual void update(Index index, const FeatureVec& featureVec)
  {
    m_features[index] = featureVec;
  }
  virtual bool getFeatureVec(Index index, FeatureVec& featureVec)
  {
    typename FeatureVecMap::const_iterator it = m_features.find(index);
    if(it == m_features.end())
      return false;
    featureVec = it->second;
    return true;
  }
private:
  FeatureVecMap m_features;
};

template<typename Dtype>
class FeatureMapContainer
{
public:
  typedef std::string Key;
public:
  static FeatureMap<Dtype>& instance(const Key& key)
  {
    typedef std::map<Key, FeatureMap<Dtype> > Registry;
    static Registry registry;
    return registry[key];
  }
};

} // namespace ultinous
} // namespace caffe
