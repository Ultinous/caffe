#pragma once

namespace caffe {
namespace ultinous {

class AbstractTripletGenerator
{
public:
  typedef std::vector<size_t> Triplet;
  typedef ImageClassificationModel::BasicModel BasicModel;
public:
  virtual Triplet nextTriplet() = 0;
};

} // namespace ultinous
} // namespace caffe
