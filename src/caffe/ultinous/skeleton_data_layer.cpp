#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include "caffe/ultinous/skeleton_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe 
{
namespace ultinous
{

template<
    typename Image,
    typename Mask
>
void datumToImageAndMask(Image&, Mask&, const Datum&);

#ifdef USE_OPENCV
template<>
void datumToImageAndMask(cv::Mat& image, cv::Mat& mask, const Datum& datum)
{
    if (datum.encoded()) {
        LOG(FATAL) << "Encoded datum is not supported!";
        return;
    }

    const auto &data = datum.data();
    const auto datumHeight = datum.height();
    const auto datumWidth = datum.width();
    const auto channelSize = datumHeight * datumWidth;
    const auto maskChannel = 4;
    const auto hasMask = datum.channels() > maskChannel;

    image = cv::Mat::zeros(datumHeight, datumWidth, CV_8UC3);
    mask = (hasMask)? cv::Mat::zeros(datumHeight, datumWidth, CV_8UC1) : cv::Mat();

    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            auto &rgb = image.at<cv::Vec3b>(i, j);
            const auto ind = i * image.cols + j;
            rgb[0] = data[0 * channelSize + ind];
            rgb[1] = data[1 * channelSize + ind];
            rgb[2] = data[2 * channelSize + ind];

            if (hasMask)
                mask.at<uint8_t>(i, j) = data[maskChannel * channelSize + ind];
        }
    }
}
#endif

template<typename PointType>
struct DatumToSkeletonDecoder
{
    using SkeletonMetaDataType = SkeletonMetaData<PointType>;

    SkeletonMetaDataType datumToMetaData(const Datum &datum) const
    {
        SkeletonMetaDataType meta;
        std::vector<proto::skeleton::SkeletonPointType> IND_TO_PROTO =
        {
            proto::skeleton::NOSE,
            proto::skeleton::LEFT_EYE,
            proto::skeleton::RIGHT_EYE,
            proto::skeleton::LEFT_EAR,
            proto::skeleton::RIGHT_EAR,
            proto::skeleton::LEFT_SHOULDER,
            proto::skeleton::RIGHT_SHOULDER,
            proto::skeleton::LEFT_ELBOW,
            proto::skeleton::RIGHT_ELBOW,
            proto::skeleton::LEFT_WRIST,
            proto::skeleton::RIGHT_WRIST,
            proto::skeleton::LEFT_HIP,
            proto::skeleton::RIGHT_HIP,
            proto::skeleton::LEFT_KNEE,
            proto::skeleton::RIGHT_KNEE,
            proto::skeleton::LEFT_ANKLE,
            proto::skeleton::RIGHT_ANKLE,
        };
        const auto datumHeight = datum.height();
        const auto datumWidth = datum.width();

        const auto metaChannel = 3;

        const auto metaOffset = static_cast<std::size_t>(datumHeight * datumWidth * metaChannel);
        const auto rowOffset = static_cast<std::size_t>(datumWidth);

        const auto &data = datum.data();
        const auto numberOfJointsInDatum = 17;        

        // 1st row (metaOffset)
        // dataset: string
        meta.dataset = decode(data, metaOffset);
        
        // 2nd row (metaOffset + rowOffset)
        // height: float
        const auto height = int(decode<float>(data, metaOffset + rowOffset));
        // width: float
        const auto width = int(decode<float>(data, metaOffset + rowOffset + 4));
        
        // 3rd row (metaOffset + 2 * rowOffset)
        // is_validation: byte
        // numOtherPeople: byte
        const auto numOtherPeople = static_cast<int>(data[metaOffset + 2 * rowOffset + 1]);
        // people index: byte
        // annolist_index: float
        // write_number: float
        // total_write_number: float

        auto visibilityConvert = [](const float _visibility, const PointType &_joint, const int _width, const int _height) {
            if (_visibility == 3)
            {
                return Visibility::NOT_AVAILABLE;
            }
            else if (_joint.x < 0 || _joint.y < 0 || _joint.x >= _width || _joint.y >= _height)
            {
                return Visibility::OUTSIDE_IMAGE;
            }
            else
            {
                return (_visibility == 0) ? Visibility::OCCLUDED : Visibility::VISIBLE;
            }
        };

        meta.targetPositions.resize(numOtherPeople + 1);
        meta.targetScales.resize(numOtherPeople + 1);
        meta.targetSkeletons.resize(numOtherPeople + 1);

        // 3rd row (metaOffset + 3 * rowOffset)
        // objpos.x: float
        meta.targetPositions[0].x = decode<float>(data, metaOffset + 3 * rowOffset);
        // objpos.y: float
        meta.targetPositions[0].y = decode<float>(data, metaOffset + 3 * rowOffset + 4);
        meta.targetPositions[0] -= PointType(1, 1); // from matlab 1-index to c++ 0-index

        // 4th row (metaOffset + 4 * rowOffset)
        // scale_self: float
        meta.targetScales[0] = decode<float>(data, metaOffset + 4 * rowOffset);

        // 5th (x: float), 6th (y: float) and 7th (visibilty: float) rows
        meta.targetSkeletons[0].joints.resize(numberOfJointsInDatum);
        meta.targetSkeletons[0].isVisible.resize(numberOfJointsInDatum);
        meta.targetSkeletons[0].jointType.resize(numberOfJointsInDatum);
        for (int i = 0; i < numberOfJointsInDatum; i++)
        {
            meta.targetSkeletons[0].joints[i].x = decode<float>(data, metaOffset + 5 * rowOffset + 4 * i);
            meta.targetSkeletons[0].joints[i].y = decode<float>(data, metaOffset + 6 * rowOffset + 4 * i);
            meta.targetSkeletons[0].joints[i] -= PointType(1, 1); // from matlab 1-index to c++ 0-index
            
            const auto isVisible = decode<float>(data, metaOffset + 7 * rowOffset + 4 * i);
            meta.targetSkeletons[0].isVisible[i] = visibilityConvert(isVisible, meta.targetSkeletons[0].joints[i], width, height);
            meta.targetSkeletons[0].jointType[i] = IND_TO_PROTO[i];
        }
        meta.focusIndex = 0;

        // (8 .. 8+numOtherPeople)th rows
        for (int p = 0; p < numOtherPeople; p++)
        {
            // x: float
            meta.targetPositions[p+1].x = decode<float>(data, metaOffset + (8 + p) * rowOffset);
            // y: float
            meta.targetPositions[p+1].y = decode<float>(data, metaOffset + (8 + p) * rowOffset + 4);
            meta.targetPositions[p+1] -= PointType(1, 1); // from matlab 1-index to c++ 0-index
            // scale: float
            meta.targetScales[p+1] = decode<float>(data, metaOffset + (8 + numOtherPeople) * rowOffset + 4 * p);
        }

        // (9+numOtherPeople)th+ rows
        for (int p = 0; p < numOtherPeople; p++)
        {
            meta.targetSkeletons[p+1].joints.resize(numberOfJointsInDatum);
            meta.targetSkeletons[p+1].isVisible.resize(numberOfJointsInDatum);
            meta.targetSkeletons[p+1].jointType.resize(numberOfJointsInDatum);
            for (int i = 0; i < numberOfJointsInDatum; i++)
            {
                meta.targetSkeletons[p+1].joints[i].x = decode<float>(data, metaOffset + (9 + numOtherPeople + 3 * p + 0) * rowOffset + 4 * i);
                meta.targetSkeletons[p+1].joints[i].y = decode<float>(data, metaOffset + (9 + numOtherPeople + 3 * p + 1) * rowOffset + 4 * i);
                meta.targetSkeletons[p+1].joints[i] -= PointType(1, 1); // from matlab 1-index to c++ 0-index
                const auto isVisible =                decode<float>(data, metaOffset + (9 + numOtherPeople + 3 * p + 2) * rowOffset + 4 * i);
                meta.targetSkeletons[p+1].isVisible[i] = visibilityConvert(isVisible, meta.targetSkeletons[p+1].joints[i], width, height);
                meta.targetSkeletons[p+1].jointType[i] = IND_TO_PROTO[i];
            }
        }

        return meta;
    }
private:
    template<class Result>
    Result* decode(const string &data, const std::size_t idx, Result *buffer, const std::size_t len = 1) const
    {
        if (len) 
            memcpy(buffer, static_cast<const void *>(data.data()+idx), len * sizeof(Result));
        return buffer;
    }

    template<class Result>
    Result decode(const std::string &data, const std::size_t idx) const
    {
        Result res;
        decode(data, idx, &res, 1);
        return res;
    }

    std::string decode(const std::string &data, const std::size_t idx) const
    {
        return std::string(data.data() + idx);
    }
};


template<typename MetaData>
void datumToSkeletonMeta(MetaData& md, const Datum& datum)
{
    md = DatumToSkeletonDecoder<typename MetaData::Point>().datumToMetaData(datum);
}

template<
    typename Image,
    typename MetaData,
    typename Mask
>
DBDataSource<Image, MetaData, Mask>::DBDataSource(const DataParameter& param)
    : param_(param)
{
    db_.reset(db::GetDB(param.backend()));
    db_->Open(param.source(), db::READ);
    cursor_.reset(db_->NewCursor());
}

template<
    typename Image,
    typename MetaData,
    typename Mask
>
void DBDataSource<Image, MetaData, Mask>::next()
{
    cursor_->Next();
    if (!cursor_->valid()) 
    {
        LOG_IF(INFO, Caffe::root_solver()) << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();
    }
}

template<
    typename Image,
    typename MetaData,
    typename Mask
>
void DBDataSource<Image, MetaData, Mask>::current(Image& img, MetaData& meta, Mask& mask)
{
    Datum datum;
    datum.ParseFromString(cursor_->value());
    datumToImageAndMask(img, mask, datum);
    datumToSkeletonMeta(meta, datum);
}

template<
    typename Dtype,
    typename CoordinateType,
    typename MaskType
>
struct HeatmapCreator
{
    using SkeletonPointType = caffe::ultinous::proto::skeleton::SkeletonPointType;
    using Skeleton = std::unordered_map<SkeletonPointType, CoordinateType, EnumClassHash>;
    using Skeletons = std::vector<Skeleton>;

    explicit HeatmapCreator(const caffe::SkeletonHeatmapGenerator& params, const std::vector<SkeletonPointType>& pointList)
        : params(params)
        , points(pointList)
    {}

    std::size_t generate(Dtype* output, const MaskType& mask, const Skeletons& skeletons, const int width, const int height, const int stride) const
    {
        const auto gridX = width / stride;
        const auto gridY = height / stride;
        const auto channelSize = gridX * gridY;
        std::fill(output, output + points.size() * channelSize, Dtype(0));
        for (std::size_t i=0; i < points.size(); ++i)
        {
            for (const auto &skel : skeletons)
            {
                const auto target = skel.find(points[i]);
                if (target != skel.end())
                {
                    putGaussian(output + i * channelSize, target->second, stride, width / stride, height / stride, params.sigma());
                }
            }
        }
        for (std::size_t i=0; i < points.size(); ++i)
        {
            auto *data = output + (points.size()+i)*channelSize;
            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    const auto value = mask(y, x);
                    const auto ind = y * mask.cols + x;
                    data[ind] = static_cast<Dtype>(value / 255.0);
                }
            }
        }

        return points.size() * 2;
    }

private:
    void putGaussian(Dtype* output, const CoordinateType& center, const int stride, const int gridX, const int gridY, const Dtype sigma) const
    {
        const auto grid_center = stride/2.0 - 0.5;
        for (int g_y = 0; g_y < gridY; g_y++) {
            for (int g_x = 0; g_x < gridX; g_x++) {
                const auto x = grid_center + g_x * stride;
                const auto y = grid_center + g_y * stride;
                const auto d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
                const auto exponent = d2 / 2.0 / sigma / sigma;
                if (exponent > 4.6052) //ln(100) = -ln(1%)
                    continue;
                const auto index = g_y*gridX + g_x;
                output[index] = aggregator(output[index], exp(-exponent));
                if (output[index] > 1)
                    output[index] = 1;
            }
        }
    }
    
    const caffe::SkeletonHeatmapGenerator params;
    const std::vector<SkeletonPointType>& points;
    const std::function<Dtype(Dtype,Dtype)> aggregator = [](Dtype a, Dtype b) { return (a+b); };
};

template<
    typename Dtype,
    typename CoordinateType,
    typename MaskType    
>
struct PAFCreator
{
    using SkeletonPointType = caffe::ultinous::proto::skeleton::SkeletonPointType;
    using Skeleton = std::unordered_map<SkeletonPointType, CoordinateType, EnumClassHash>;
    using Skeletons = std::vector<Skeleton>;

#ifdef USE_OPENCV
    using CounterValueType = unsigned char;
    using CounterType = cv::Mat_<CounterValueType>;
#endif

    explicit PAFCreator(const caffe::SkeletonPAFGenerator& params, const std::vector<std::pair<SkeletonPointType, SkeletonPointType>>& edgeList)
        : params(params)
        , edges(edgeList)
    {}

    std::size_t generate(Dtype* output, const MaskType& mask, const Skeletons& skeletons, const int width, const int height, const int stride) const
    {
        const auto gridX = width / stride;
        const auto gridY = height / stride;
        const auto channelSize = gridX * gridY;
        std::fill(output, output + 2*edges.size() * channelSize, Dtype(0));
        for (std::size_t i = 0; i < edges.size(); ++i)
        {
            CounterType count = CounterType(gridX, gridY, CounterValueType(0));
            const auto u = edges[i].first;
            const auto v = edges[i].second;
            for (const auto &skel : skeletons)
            {
                const auto uPoint = skel.find(u);
                const auto vPoint = skel.find(v);
                if (uPoint != skel.end() && vPoint != skel.end())
                {
                    putVecMaps(output + i*2*channelSize, output + (i*2+1)*channelSize, count, 
                        uPoint->second, vPoint->second, stride, gridX, gridY, params.vec_distance_threshold());
                }
            }
        }
        for (std::size_t i=0; i < edges.size(); ++i)
        {
            auto *data = output + (edges.size()*2+i*2)*channelSize;
            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    const auto value = mask(y, x);
                    const auto ind = y * mask.cols + x;
                    data[0 * channelSize + ind] = static_cast<Dtype>(value / 255.0);
                    data[1 * channelSize + ind] = static_cast<Dtype>(value / 255.0);
                }
            }
        }

        return edges.size() * 4;
    }

private:
    void putVecMaps(Dtype *entryX, Dtype *entryY, CounterType &count, const CoordinateType centerA, const CoordinateType centerB, const int stride, const int gridX, const int gridY, const int distThresh) const
    {
        const auto stridedCenterA = centerA / stride;
        const auto stridedCenterB = centerB / stride;

        if (0 > stridedCenterA.x || stridedCenterA.x >= gridX || 0 > stridedCenterB.x || stridedCenterB.x >= gridX
            || 0 > stridedCenterA.y || stridedCenterA.y >= gridY || 0 > stridedCenterB.y || stridedCenterB.y >= gridY)
        {
            return;
        }

        const auto min_x = std::max(boost::numeric_cast<int>(std::round(std::min(stridedCenterA.x, stridedCenterB.x) - distThresh)), 0);
        const auto max_x = std::min(boost::numeric_cast<int>(std::round(std::max(stridedCenterA.x, stridedCenterB.x) + distThresh)), gridX);

        const auto min_y = std::max(boost::numeric_cast<int>(std::round(std::min(stridedCenterA.y, stridedCenterB.y) - distThresh)), 0);
        const auto max_y = std::min(boost::numeric_cast<int>(std::round(std::max(stridedCenterA.y, stridedCenterB.y) + distThresh)), gridY);

        auto BA_dir = stridedCenterB - stridedCenterA;
        const auto norm_ba = std::sqrt(BA_dir.x * BA_dir.x + BA_dir.y * BA_dir.y);
        if (norm_ba > 1e-10)
            BA_dir /= norm_ba;

        for (int g_y = min_y; g_y < max_y; ++g_y)
        {
            for (int g_x = min_x; g_x < max_x; ++g_x)
            {
                const CoordinateType point(g_x - stridedCenterA.x, g_y - stridedCenterA.y);
                const auto dist = std::abs(point.x * BA_dir.y - point.y * BA_dir.x);

                if (dist <= distThresh)
                {
                    const auto cnt = count(g_y, g_x);
                    const auto index = g_y * gridX + g_x;
                    if (cnt == 0)
                    {
                        entryX[index] = BA_dir.x;
                        entryY[index] = BA_dir.y;
                    }
                    else
                    {
                        entryX[index] = (entryX[index] * cnt + BA_dir.x) / (cnt + 1);
                        entryY[index] = (entryY[index] * cnt + BA_dir.y) / (cnt + 1);
                    }
                    count(g_y, g_x) = cnt + 1;
                }
            }
        }
    }

    const caffe::SkeletonPAFGenerator params;
    const std::vector<std::pair<SkeletonPointType, SkeletonPointType>> edges;
};

template<typename Dtype>
SkeletonDataLayer<Dtype>::SkeletonDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param)
{
    CHECK(this->layer_param_.has_skeleton_data_param());
    auto &skeleton_data_param = this->layer_param_.skeleton_data_param();
    if (skeleton_data_param.has_db_data_param())
        this->data_source_.reset(new DBDataSource<ImageType, MetaData>(skeleton_data_param.db_data_param()));
    else
        LOG(FATAL) << "No supported data source given!";

    if (skeleton_data_param.has_output_model())
    {
        std::transform(skeleton_data_param.output_model().points().begin(), skeleton_data_param.output_model().points().end(), std::back_inserter(output_points),
            [](const int p) {
                return PointType(p);
            }        
        );
        std::transform(skeleton_data_param.output_model().edges().begin(), skeleton_data_param.output_model().edges().end(), std::back_inserter(output_edges), 
            [](const proto::skeleton::SkeletonEdge &p) {
                return Edge(p.u(), p.v());
            }
        );

        std::copy_if(output_points.begin(), output_points.end(), std::back_inserter(generated_output_points), 
            [](const proto::skeleton::SkeletonPointType p) { return p > proto::skeleton::SkeletonPointType::GEN_FIRST; });
        
        has_point.reset();
        for (const auto pt : output_points) 
            has_point.set(static_cast<std::size_t>(pt));
    }

    data_transformer_.reset(new SkeletonDataTransformer<Dtype>(skeleton_data_param.skeleton_transform_param(), param.phase()));
}

template<typename Dtype>
SkeletonDataLayer<Dtype>::~SkeletonDataLayer()
{
    this->StopInternalThread();
}

template<typename Dtype>
std::vector<int> SkeletonDataLayer<Dtype>::getLabelSize(const std::vector<int>& dataSize)
{
    const auto stride = this->layer_param_.skeleton_data_param().skeleton_transform_param().stride();
    auto result = dataSize;
    result[0] = 1;
    result[1] = output_points.size() * 2 + output_edges.size() * 4;
    result[2] /= stride;
    result[3] /= stride;
    return result;
}

template<typename Dtype>
void SkeletonDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{    
    ImageType image, mask;
    MetaData md;
    data_source_->current(image, md, mask);
    // Use cpm_data_transformer to infer the expected blob shape from datum.
    auto top_shape = this->data_transformer_->InferDataBlobShape(image);
    
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = data_source_->batchSize();
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i)
    {
        this->prefetch_[i]->data_.Reshape(top_shape);
    }

    if (this->output_labels_)
    {
        std::vector<int> label_shape = getLabelSize(top_shape);
        this->transformed_label_.Reshape(label_shape);
        label_shape[0] = data_source_->batchSize();
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
        {
            this->prefetch_[i]->label_.Reshape(label_shape);
        }
    }
}

template<typename Dtype>
void SkeletonDataLayer<Dtype>::generatePoints(Skeleton& skeleton)
{
    auto midpointGenerator = [](const Skeleton& _skeleton, const PointType u, const PointType v)
    {
        auto uValue = _skeleton.find(u);
        auto vValue = _skeleton.find(v);
        if (uValue != _skeleton.end() && vValue != _skeleton.end())
        {
            return std::make_pair(true, (uValue->second + vValue->second) / 2.0);
        }
        else
            return std::make_pair(false, typename Skeleton::mapped_type());
    };
    
    for (const auto gen : generated_output_points)
    {
        const auto result = [&midpointGenerator](const Skeleton& _skeleton, const PointType target)
        {
            switch(target)
            {
                case proto::skeleton::SkeletonPointType::GEN_NECK: 
                    return midpointGenerator(_skeleton, 
                        proto::skeleton::SkeletonPointType::LEFT_SHOULDER, 
                        proto::skeleton::SkeletonPointType::RIGHT_SHOULDER);
                default: return std::make_pair(false, typename Skeleton::mapped_type());
            }            
        }(skeleton, gen);
        if (result.first)
            skeleton[gen] = result.second;
    }
}

template<typename Dtype>
void SkeletonDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
    CPUTimer batchTimer;
    batchTimer.Start();
    double readTime = 0;
    double transTime = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    const int batchSize = data_source_->batchSize();

    auto* top_data = batch->data_.mutable_cpu_data();
    auto* top_label = (this->output_labels_)? batch->label_.mutable_cpu_data() : nullptr;

    for (int item_id = 0; item_id < batchSize; ++item_id) 
    {
        timer.Start();
        ImageType img, mask;
        MetaData md;
        data_source_->current(img, md, mask);
        if (mask.size().empty())
        {
            mask = cv::Mat::ones(img.size(), CV_8U) * 255;
        }

        readTime += timer.MicroSeconds();

        if (item_id == 0) 
        {
            // Reshape according to the first datum of each batch
            // on single input batches allows for inputs of varying dimension.
            // Use data_transformer to infer the expected blob shape from datum.
            auto top_shape = this->data_transformer_->InferDataBlobShape(img);            
            this->transformed_data_.Reshape(top_shape);

            bufferRGB_ = cv::Mat::zeros(top_shape[2], top_shape[3], CV_8UC3);            

            auto label_shape = getLabelSize(top_shape);
            this->transformed_label_.Reshape(label_shape);
            
            bufferMask_ = cv::Mat::zeros(label_shape[2], label_shape[3], CV_8U);

            // Reshape batch according to the batchSize.
            top_shape[0] = batchSize;
            batch->data_.Reshape(top_shape);
            label_shape[0] = batchSize;
            batch->label_.Reshape(label_shape);
        }

        // Apply data transformations (mirror, scale, crop...)
        timer.Start();
        data_transformer_->Transform(img, mask, md, bufferRGB_, bufferMask_);

        if (top_label)
        {
            Skeletons target;
            for (const auto &skel : md.targetSkeletons)
            {
                Skeleton s;
                for (std::size_t i = 0; i < skel.joints.size(); ++i)
                {
                    if (skel.isVisible[i] == Visibility::VISIBLE && has_point.test(static_cast<std::size_t>(skel.jointType[i])))
                    {
                        s[skel.jointType[i]] = skel.joints[i];
                    }
                }
                generatePoints(s);
                target.push_back(s);
            }

            const auto channelSize = this->transformed_label_.count(2);
            const auto offset_label = batch->label_.offset(item_id);
            std::size_t channelIndex = 0;
            const auto &skeletonParam = this->layer_param_.skeleton_data_param();

            if (skeletonParam.has_skeleton_heatmap_generator())
            {
                HeatmapCreator<Dtype, typename MetaData::Point, cv::Mat_<uint8_t>> heatmapCreator(skeletonParam.skeleton_heatmap_generator(), output_points);
                this->transformed_label_.set_cpu_data(top_label + offset_label + channelIndex * channelSize);
                channelIndex += heatmapCreator.generate(this->transformed_label_.mutable_cpu_data(), bufferMask_, target, bufferRGB_.cols, bufferRGB_.rows, 
                                                            this->layer_param_.skeleton_data_param().skeleton_transform_param().stride());
            }
            
            if (skeletonParam.has_skeleton_paf_generator())
            {
                PAFCreator<Dtype, typename MetaData::Point, cv::Mat_<uint8_t>> pafCreator(skeletonParam.skeleton_paf_generator(), output_edges);
                this->transformed_label_.set_cpu_data(top_label + offset_label + channelIndex * channelSize);
                channelIndex += pafCreator.generate(this->transformed_label_.mutable_cpu_data(), bufferMask_, target, bufferRGB_.cols, bufferRGB_.rows, 
                                                        this->layer_param_.skeleton_data_param().skeleton_transform_param().stride());
            }
        }

        const auto offset_data = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset_data);

        const auto channelSize = bufferRGB_.cols * bufferRGB_.rows;
        auto data = this->transformed_data_.mutable_cpu_data();
        for (int i = 0; i < bufferRGB_.rows; ++i)
        {
            for (int j = 0; j < bufferRGB_.cols; ++j)
            {
                auto &rgb = bufferRGB_.at<cv::Vec3b>(i, j);
                const auto ind = i * bufferRGB_.cols + j;
                data[0 * channelSize + ind] = rgb[0];
                data[1 * channelSize + ind] = rgb[1];
                data[2 * channelSize + ind] = rgb[2];
            }
        }

        transTime += timer.MicroSeconds();
        data_source_->next();
    }
    timer.Stop();
    batchTimer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batchTimer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << readTime / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << transTime / 1000 << " ms.";
}

INSTANTIATE_CLASS(SkeletonDataLayer);
REGISTER_LAYER_CLASS(SkeletonData);

} // namespace ultinous
} // namespace caffe