#pragma once

#include <vector>
#include <boost/concept_check.hpp>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include <caffe/ultinous/AbstractTripletGenerator.hpp>


namespace caffe {
namespace ultinous {

template <typename Dtype>
class AllTripletGenerator : public AbstractTripletGenerator
{
public:
  AllTripletGenerator(const BasicModel& basicModel)
    : m_basicModel(basicModel)
  {
    m_currentClass = 0;
    m_currentImage = 0;
    m_generation = 0;
  }

public:
  Triplet nextTriplet()
  {
    Triplet t;

    t.push_back( m_basicModel[m_currentClass].images[m_currentImage] );
    t.push_back( m_basicModel[m_currentClass].images[m_currentImage] );

    int numClasses = m_basicModel.size();
    int otherClass = rand() % (numClasses-1);
    if( otherClass >= m_currentClass ) otherClass++;
    int otherImage = rand() % m_basicModel[otherClass].images.size();

    t.push_back( m_basicModel[otherClass].images[otherImage] );

    ++m_currentImage;
    if( m_currentImage >= m_basicModel[m_currentClass].images.size() ) {
      m_currentImage = 0;
      ++m_currentClass;
      if( m_currentClass >= m_basicModel.size() ) {
        m_currentClass = 0;
        ++m_generation;
      }
    }

    return t;
  }

  int getGeneration()
  {
    return m_generation;
  }

private:
  const BasicModel& m_basicModel;
  int m_currentClass;
  int m_currentImage;
  int m_generation;
};

} // namespace ultinous
} // namespace caffe
