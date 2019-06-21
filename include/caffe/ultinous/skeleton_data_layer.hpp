#pragma once

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <bitset>
#include <boost/format.hpp>
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/ultinous/skeleton_meta_data.hpp"
#include "caffe/ultinous/skeleton_data_transformer.hpp"


namespace caffe
{
namespace ultinous
{
template<
    typename Image,
    typename MetaData,
    typename Mask = Image
>
struct IGraphDataSource
{
    virtual ~IGraphDataSource() = default;

    virtual int batchSize() const = 0;
    virtual void current(Image&, MetaData&, Mask&) = 0;
    virtual void next() = 0;
};

template<
    typename Image,
    typename MetaData,
    typename Mask = Image
>
struct DBDataSource
    : public IGraphDataSource<Image, MetaData, Mask>
{
    DBDataSource(const DataParameter& param);

    int batchSize() const override { return param_.batch_size(); }
    void next() override;
    void current(Image& img, MetaData& meta, Mask& mask) override;

protected:
    DataParameter param_;
    shared_ptr<db::DB> db_;
    shared_ptr<db::Cursor> cursor_;        
};

struct EnumClassHash
{
    using result_type = std::size_t;

    template<typename T>
    result_type operator()(T const& value) const noexcept
    {
        return static_cast<result_type>(value);
    }
};

template<typename Dtype>
class SkeletonDataLayer : public BasePrefetchingDataLayer<Dtype>
{
public:
    explicit SkeletonDataLayer(const LayerParameter&);
    ~SkeletonDataLayer() override;

    void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

    inline const char* type() const override { return "SkeletonDataLayer"; }
    inline int ExactNumBottomBlobs() const override { return 0; }
    inline int ExactNumTopBlobs() const override { return 2; }

protected:
    using PointType = proto::skeleton::SkeletonPointType;
    using Edge = std::pair<PointType, PointType>;
    using Edges = std::vector<Edge>;
#ifdef USE_OPENCV
    using ImageType = cv::Mat;
    using CoordinateType = cv::Point_<Dtype>;
    using MetaData = SkeletonMetaData<CoordinateType>;
#endif
    using Skeleton = std::unordered_map<PointType, CoordinateType, EnumClassHash>;
    using Skeletons = std::vector<Skeleton>;


    void load_batch(Batch<Dtype>* batch) override;
    std::vector<int> getLabelSize(const std::vector<int>& dataSize);
    void generatePoints(Skeleton&);

    std::vector<PointType> output_points;
    std::vector<PointType> generated_output_points;
    std::vector<Edge> output_edges;
    std::bitset<proto::skeleton::SkeletonPointType_ARRAYSIZE> has_point;

    Blob<Dtype> transformed_label_;
    std::shared_ptr<IGraphDataSource<ImageType, MetaData>> data_source_;
    std::shared_ptr<SkeletonDataTransformer<Dtype>> data_transformer_;

#ifdef USE_OPENCV
    cv::Mat bufferRGB_;
    cv::Mat bufferMask_;
#endif
};
} // namespace ultinous
} // namespace caffe