#pragma once

#include <vector>
#include <boost/concept_check.hpp>
#include <caffe/ultinous/PictureClassificationModel.h>
#include <caffe/ultinous/FeatureMap.hpp>
#include <caffe/ultinous/AbstractTripletGenerator.hpp>


namespace caffe {
namespace ultinous {

template <typename Dtype>
class RandomTripletGenerator : public AbstractTripletGenerator
{
public:
  RandomTripletGenerator(const BasicModel& basicModel)
    : m_basicModel(basicModel)
  {
  }

public:
  Triplet nextTriplet()
  {
    Triplet t;

    int labIx1, labIx2;
    labIx1 = rand() % m_basicModel.size();
    do{ labIx2 = rand() % m_basicModel.size(); } while(labIx1==labIx2);
    int imageIxA, imageIxP, imageIxN;
    imageIxA = rand() % m_basicModel[labIx1].images.size();
    do{ imageIxP = rand() % m_basicModel[labIx1].images.size(); } while(imageIxA==imageIxP);
    imageIxN = rand() % m_basicModel[labIx2].images.size();

    t.push_back( m_basicModel[labIx1].images[imageIxA] );
    t.push_back( m_basicModel[labIx1].images[imageIxP] );
    t.push_back( m_basicModel[labIx2].images[imageIxN] );

    return t;
  }

private:
  const BasicModel& m_basicModel;
};

} // namespace ultinous
} // namespace caffe
