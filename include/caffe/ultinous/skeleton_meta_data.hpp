#pragma once

namespace caffe
{
namespace ultinous
{
enum class Visibility {
  OCCLUDED = 0,
  VISIBLE = 1,
  OUTSIDE_IMAGE = 2,
  NOT_AVAILABLE = 3
};

template<typename CoordinateType>
struct Skeleton {
  vector <CoordinateType> joints;
  vector <Visibility> isVisible;
  vector <caffe::ultinous::proto::skeleton::SkeletonPointType> jointType;
};

template<typename CoordinateType>
struct SkeletonMetaData {
  using Point = CoordinateType;
  using SkeletonType = Skeleton<CoordinateType>;

  std::string dataset;
  std::size_t focusIndex;

  std::vector<CoordinateType> targetPositions;
  std::vector<typename CoordinateType::value_type> targetScales;
  std::vector<SkeletonType> targetSkeletons;
};
} // namespace ultinous
} // namespace caffe
