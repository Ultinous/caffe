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
  typedef std::vector<FeatureVec> FeatureVecMap;
public:
  virtual void resize( size_t length )
  {
    m_features.resize( length );
    m_validFeatures.clear( );
  }

  virtual void update(Index index, const FeatureVec& featureVec)
  {
    m_features[index] = featureVec;
    m_validFeatures.insert(index);
  }
  virtual const FeatureVec& getFeatureVec(Index index) const
  {
    return m_features[index];
  }
  int numFeatures( ) const
  {
    return m_validFeatures.size();
  }
private:
  FeatureVecMap m_features;
  std::set<Index> m_validFeatures;
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
