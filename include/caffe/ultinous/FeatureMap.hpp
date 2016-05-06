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
  virtual const FeatureVec& getFeatureVec(Index index) const
  {
    typename FeatureVecMap::const_iterator it = m_features.find(index);
    return (it == m_features.end())?m_default:it->second;
  }
  int numFeatures( ) const
  {
    return m_features.size();
  }
private:
  FeatureVecMap m_features;
  FeatureVec m_default;
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
