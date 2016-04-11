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

  typedef int16_t Lives;

  struct StoredTriplet
  {
    StoredTriplet( Triplet triplet, Lives lives = START_LIVES )
      : m_triplet( triplet )
      , m_lives( lives )
    { }

    Triplet m_triplet;
    Lives m_lives;
  };

  typedef std::deque<StoredTriplet> Pool;

public:
  HardTripletPool( HardTripletGeneratorPtr const &htg, size_t maxPoolSize)
    : m_htg(htg)
    , m_featureMap(m_htg->getFeatureMap())
    , m_margin(m_htg->getMargin())
    , m_maxPoolSize( maxPoolSize )
  { }

  Triplet nextTriplet()
  {
    Triplet triplet = m_htg->nextTriplet( );

    if( m_htg->isLastTripletHard( ) )
    {
      if( m_pool.size() < m_maxPoolSize )
        storeTriplet( triplet );
    }
    else
    {
      while( !m_pool.empty() )
      {
        StoredTriplet storedTriplet = m_pool.front( );
        m_pool.pop_front( );

        triplet = storedTriplet.m_triplet;

        if( isHardTriplet( triplet ) )
        {
          --storedTriplet.m_lives;
          if( storedTriplet.m_lives > 0)
            m_pool.push_back( storedTriplet );

          break;
        }
      }
    }

    return triplet;
  }

private:
  void storeTriplet( Triplet t )
  {
    m_pool.push_back( StoredTriplet(t) );
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

  Pool m_pool;
  size_t const m_maxPoolSize;

  static const Lives START_LIVES = 3;
};

} // namespace ultinous
} // namespace caffe
