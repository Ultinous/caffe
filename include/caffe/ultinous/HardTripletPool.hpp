#pragma once

#include <deque>
#include <boost/concept_check.hpp>
#include <caffe/ultinous/HardTripletGenerator.hpp>
#include <caffe/ultinous/AllTripletGenerator.hpp>

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
    StoredTriplet( Triplet triplet, uint64_t lastShown, Lives lives )
      : m_triplet( triplet )
      , m_lastShown( lastShown )
      , m_lives( lives )
    { }

    Triplet m_triplet;
    uint64_t m_lastShown;
    Lives m_lives;
  };

  typedef std::deque<StoredTriplet> Pool;

public:
  HardTripletPool( HardTripletGeneratorPtr const &htg, const HardTripletPoolParameter& hard_triplet_pool_param )
    : m_htg(htg)
    , m_featureMap(m_htg->getFeatureMap())
    , m_margin(m_htg->getMargin())
    , m_maxPoolSize( hard_triplet_pool_param.poolsize() )
    , m_showCount( hard_triplet_pool_param.showcount() )
    , m_showCycle( hard_triplet_pool_param.showcycle() )
  { }

  Triplet nextTriplet()
  {
    throw std::exception( );
  }

  Triplet nextTriplet(uint64_t iteration)
  {
    Triplet triplet = m_htg->nextTriplet( );

    if( isHardTriplet(triplet) /*m_htg->isLastTripletHard( )*/ )
    {
      if( m_pool.size() < m_maxPoolSize )
        storeTriplet( triplet, iteration );
      return triplet;
    }

    for( typename HardTripletPool<Dtype>::Pool::iterator it = m_pool.begin(); it!= m_pool.end(); /* NOP */ )
    {
      if( iteration - it->m_lastShown >= m_showCycle )
      {
        if( isHardTriplet( it->m_triplet ) )
        {
          Triplet triplet = it->m_triplet;

          --it->m_lives;
          if( it->m_lives == 0 )
            it = m_pool.erase(it);
          else
            it->m_lastShown = iteration;

          return triplet;
        }
        it = m_pool.erase(it);
      }
      else
      {
        ++it;
      }
    }

    return AllTripletGenerator<Dtype>::getInstance().nextTriplet();
  }

private:
  void storeTriplet( Triplet t, uint64_t iteration )
  {
    m_pool.push_back( StoredTriplet(t, iteration, m_showCount) );
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

  size_t const m_maxPoolSize;
  Lives const m_showCount;
  uint64_t const m_showCycle;

  Pool m_pool;
};

} // namespace ultinous
} // namespace caffe
