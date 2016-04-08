#pragma once

#include <deque>
#include <boost/concept_check.hpp>
#include <caffe/ultinous/HardTripletGenerator.hpp>

namespace caffe {
namespace ultinous {

template <typename Dtype>
class HardTripletPool : public AbstractTripletGenerator
{

private:
  typedef boost::shared_ptr<HardTripletGenerator<Dtype> > HardTripletGeneratorPtr;
public:
  HardTripletPool( HardTripletGeneratorPtr const &htg )
    : m_htg(htg)
    , m_featureMap(m_htg->getFeatureMap())
    , m_margin(m_htg->getMargin())
  { }
public:

  Triplet nextTriplet()
  {
    Triplet t = m_htg->nextTriplet( );

    if( m_htg->isLastTripletHard( ) )
    {
      if( m_pool.size() < MAX_POOL_SIZE )
        m_pool.push_back( t );
    }
    else
    {
      while( !m_pool.empty() )
      {
        t = m_pool.front( );
        m_pool.pop_front( );

        if( isHardTriplet( t ) )
        {
          m_pool.push_back( t );
          break;
        }
      }
    }

    return t;
  }

  bool isHardTriplet( Triplet const &t )
  {
    if( t.size() != 3 ) return false;

    const typename FeatureMap<Dtype>::FeatureVec& f1 = m_featureMap.getFeatureVec( t[0] );
    const typename FeatureMap<Dtype>::FeatureVec& f2 = m_featureMap.getFeatureVec( t[1] );
    const typename FeatureMap<Dtype>::FeatureVec& f3 = m_featureMap.getFeatureVec( t[2] );

    CHECK_GT( f1.size(), 0 );
    CHECK_EQ( f1.size(), f2.size() );
    CHECK_EQ( f1.size(), f3.size() );

    typename FeatureMap<Dtype>::FeatureVec sqr( f1.size() );

    caffe_sub( f1.size(), &(f1[0]), &(f2[0]), &(sqr[0]) );
    Dtype dist1 = caffe_cpu_dot( sqr.size(), &(sqr[0]), &(sqr[0]) );

    caffe_sub( f1.size(), &(f1[0]), &(f3[0]), &(sqr[0]) );
    Dtype dist2 = caffe_cpu_dot( sqr.size(), &(sqr[0]), &(sqr[0]) );


    return  dist2 < dist1 + m_margin;
  }
private:
  HardTripletGeneratorPtr const m_htg;
  FeatureMap<Dtype>const &m_featureMap;
  Dtype m_margin;

  std::deque<Triplet> m_pool;
  const static size_t MAX_POOL_SIZE = 1000000;
};

} // namespace ultinous
} // namespace caffe
