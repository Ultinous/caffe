#pragma once

#include <vector>
#include <boost/concept_check.hpp>
#include <boost/make_shared.hpp>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include <caffe/ultinous/AbstractTripletGenerator.hpp>


namespace caffe {
namespace ultinous {

template <typename Dtype>
class FeatureCollectorTripletGenerator : public AbstractTripletGenerator
{
public:
  typedef boost::shared_ptr<FeatureCollectorTripletGenerator<Dtype> > FeatureCollectorTripletGeneratorPtr;

private:
  FeatureCollectorTripletGenerator(const BasicModel& basicModel)
    : m_basicModel(basicModel)
    , m_imageIndices( vector<int>(m_basicModel.size(), 0) )
    , m_currentClass(0)
  {
  }

  static FeatureCollectorTripletGeneratorPtr& s_instance( )
  {
    static FeatureCollectorTripletGeneratorPtr data;
    return data;
  }

public:
  static void init( const BasicModel& basicModel )
  {
    if( s_instance() ) throw std::exception();
    s_instance().reset( new FeatureCollectorTripletGenerator<Dtype>( basicModel ) );
  }

  static FeatureCollectorTripletGenerator& getInstance( )
  {
    if( !s_instance() ) throw std::exception();
    return *s_instance();
  }

  Triplet nextTriplet()
  {
    Triplet t;

    t.push_back( m_basicModel[m_currentClass].images[m_imageIndices[m_currentClass]] );

    ++m_imageIndices[m_currentClass];
    if( m_imageIndices[m_currentClass] >= m_basicModel[m_currentClass].images.size() )
      m_imageIndices[m_currentClass] = 0;

    t.push_back( m_basicModel[m_currentClass].images[m_imageIndices[m_currentClass]] );

    ++m_imageIndices[m_currentClass];
    if( m_imageIndices[m_currentClass] >= m_basicModel[m_currentClass].images.size() )
      m_imageIndices[m_currentClass] = 0;


    int numClasses = m_basicModel.size();
    int otherClass = rand() % (numClasses-1);
    if( otherClass >= m_currentClass ) otherClass++;

    int otherImage = m_imageIndices[otherClass]; //rand() % m_basicModel[otherClass].images.size();
    t.push_back( m_basicModel[otherClass].images[otherImage] );

    ++m_imageIndices[otherClass];
    if( m_imageIndices[otherClass] >= m_basicModel[otherClass].images.size() )
      m_imageIndices[otherClass] = 0;

    ++m_currentClass;
    if( m_currentClass >= m_basicModel.size() )
      m_currentClass = 0;


    return t;
  }

private:
  const BasicModel& m_basicModel;
  vector<int> m_imageIndices;
  int m_currentClass;
};


} // namespace ultinous
} // namespace caffe
