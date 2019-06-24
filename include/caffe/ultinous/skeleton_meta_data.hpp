#pragma once

namespace caffe
{
namespace ultinous
{
enum class Visibility
{
    OCCLUDED = 0,
    VISIBLE = 1,
    OUTSIDE_IMAGE = 2,
    NOT_AVAILABLE = 3
};

template<typename PointType>
struct Skeleton
{
    vector<PointType> joints;
    vector<Visibility> isVisible;
    vector<caffe::ultinous::proto::skeleton::SkeletonPointType> jointType;
};

template<typename PointType>
struct SkeletonMetaData
{
    using Point = PointType;
    using SkeletonType = Skeleton<PointType>;

    std::string dataset;
    std::size_t focusIndex;

    std::vector<PointType> targetPositions;
    std::vector<typename PointType::value_type> targetScales;
    std::vector<SkeletonType> targetSkeletons;
};
} // namespace ultinous
} // namespace caffe
