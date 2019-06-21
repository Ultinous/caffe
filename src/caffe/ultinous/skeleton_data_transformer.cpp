#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#include "caffe/ultinous/skeleton_data_transformer.hpp"
#include <boost/numeric/conversion/cast.hpp>
#include "caffe/util/math_functions.hpp"

namespace caffe
{
namespace ultinous
{

template<typename Dtype>
SkeletonDataTransformer<Dtype>::SkeletonDataTransformer(const SkeletonTransformationParameter& param, Phase phase)
    : param_(param)
    , phase_(phase)
{
    if (param_.mean_value_size() > 0) 
    {
        CHECK(!param_.has_mean_file()) << "Cannot specify mean_file and mean_value at the same time";
        std::copy(param_.mean_value().begin(), param_.mean_value().end(), std::back_inserter(mean_values_));
    }
    else
    {
        mean_values_ = {128, 128, 128};
    }

    if (phase == TRAIN) 
    {
        const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
    } 
    else 
    {
        rng_.reset();
    }
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> SkeletonDataTransformer<Dtype>::InferDataBlobShape(const cv::Mat& cv_img)
{
    const auto crop_size = param_.crop_size();
    const auto img_channels = (cv_img.type() == CV_8UC3)? 3 : 0;
    const auto img_height = cv_img.rows;
    const auto img_width = cv_img.cols;
    
    // Check dimensions.
    CHECK_GT(img_channels, 0);

    return {
        1,
        img_channels,
        (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : img_height,
        (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : img_width
    };
}

template<typename Dtype>
void SkeletonDataTransformer<Dtype>::Transform(const cv::Mat& img, const cv::Mat& mask, MetaData& meta, 
    cv::Mat& transformed_img, cv::Mat& transformed_mask)
{
    AugmentParameters<Dtype> aug(param_, rng_);

    if (phase_ == TRAIN) 
    {
        aug.randomize();
        auto& targetPosition = meta.targetPositions[meta.focusIndex];
        TransformImage(img, aug, mean_values_, targetPosition, transformed_img);

        aug.grayscale = false;        
        cv::Mat maskAug = cv::Mat::zeros(mask.size(), mask.type());
        TransformImage(mask, aug, {255}, targetPosition, maskAug);
        if (maskAug.size() != transformed_mask.size())
            cv::resize(maskAug, transformed_mask, transformed_mask.size(), 0.0, 0.0, cv::INTER_NEAREST);
        else
            transformed_mask = maskAug;
        TransformMetaJoints(meta, aug, img.size(), transformed_img.size(), targetPosition);
    } else {
        transformed_img = img;
        transformed_mask = mask;
    }
}

template<typename Point>
Point transform(const cv::Mat_<typename Point::value_type> &A, const Point &p) {
    return Point(
        A(0, 0) * p.x + A(0, 1) * p.y + A(0, 2),
        A(1, 0) * p.x + A(1, 1) * p.y + A(1, 2)
    );
}


template<typename Dtype, typename Size>
cv::Rect_<Dtype> transformBoundingBox(const Size& input, const cv::Mat_<Dtype> A)
{
    const std::vector<cv::Point_<Dtype>> bb = {
        cv::Point_<Dtype>(0, 0),
        cv::Point_<Dtype>(0, input.height),
        cv::Point_<Dtype>(input.width, input.height),
        cv::Point_<Dtype>(input.width, 0),
    };

    Dtype maxX = std::numeric_limits<Dtype>::min();
    Dtype maxY = std::numeric_limits<Dtype>::min();
    Dtype minX = std::numeric_limits<Dtype>::max();
    Dtype minY = std::numeric_limits<Dtype>::max();
    for (const auto& p : bb)
    {
        const auto trf = transform(A, p);
        if (trf.x < minX) minX = trf.x;
        if (trf.y < minY) minY = trf.y;
        if (trf.x > maxX) maxX = trf.x;
        if (trf.y > maxY) maxY = trf.y;
    }

    return cv::Rect_<Dtype>(minX, minY, maxX-minX, maxY-minY);
}

template<typename Dtype>
void SkeletonDataTransformer<Dtype>::TransformImage(const cv::Mat& input, const AugmentParameters<Dtype>& aug, 
    const std::vector<Dtype>& meanValues, const cv::Point2f& focusPoint, cv::Mat& output) const
{
    CHECK_EQ(input.type(), output.type());

    cv::Scalar mean;
    if (meanValues.size() == 3)
        mean = cv::Scalar(meanValues[0], meanValues[1], meanValues[3]);
    else
        mean = cv::Scalar(meanValues[0]);
    
    cv::Mat result, resultHelper;
    cv::Point_<Dtype> center;
    if (aug.rotation || aug.scale)
    {
        center = cv::Point2f(input.cols/2.0f, input.rows/2.0f);
        cv::Mat_<Dtype> A = cv::getRotationMatrix2D(center, aug.degree, aug.scaleAmount);
        const auto bbox = transformBoundingBox<Dtype>(input.size(), A);
        A(0, 2) -= bbox.x;
        A(1, 2) -= bbox.y;        

        cv::warpAffine(input, result, A, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, mean);

        //LOG(INFO) << "input: " << input.size() << ", focus: " << focusPoint;
        //LOG(INFO) << "A: " << A;
        //LOG(INFO) << "bbox: " << bbox;
        center = transform(A, focusPoint);
        //LOG(INFO) << "center: " << center;
    }
    else
    {
        result = input.clone();
        center = focusPoint;
    }

    if (aug.crop)
    {
      center += cv::Point_<Dtype>(aug.centroidX, aug.centroidY);
      //LOG(INFO) << "center: " << center;
      const auto offsetLeft = boost::numeric_cast<int>(center.x - (aug.cropWidth / 2.0));
      const auto offsetTop = boost::numeric_cast<int>(center.y - (aug.cropHeight / 2.0));

      resultHelper = cv::Mat::zeros(aug.cropHeight, aug.cropWidth, result.type()) + mean;
      const auto inputRoi = cv::Rect(
              std::max(0, offsetLeft),
              std::max(0, offsetTop),
              std::min(static_cast<std::size_t>(result.cols), offsetLeft + aug.cropWidth) - std::max(0, offsetLeft),
              std::min(static_cast<std::size_t>(result.rows), offsetTop + aug.cropHeight) - std::max(0, offsetTop)
      );
      const auto dstRoi = cv::Rect(
              inputRoi.x-offsetLeft,
              inputRoi.y-offsetTop,
              inputRoi.width,
              inputRoi.height
      );
      //LOG(INFO) << "Offset: " << offsetLeft << " " << offsetTop;
      //LOG(INFO) << result.size() << " " << resultHelper.size();
      //LOG(INFO) << "input roi: " << inputRoi;
      //LOG(INFO) << "dst roi: " << dstRoi;
      result(inputRoi).copyTo(resultHelper(dstRoi));
    }
    else
    {
      resultHelper = result;
    }

    if (aug.flip)
    {
      cv::flip(resultHelper, result, 1);
    }
    else
    {
      result = resultHelper;
    }

    if (aug.grayscale)
    {
        if (output.size() != result.size())
            result.copyTo(output);

        for (int i = 0; i < result.rows; ++i)
        {
            for (int j = 0; j < result.cols; ++j)
            {
                auto &bgr = result.at<cv::Vec3b>(i, j);
                auto gray = bgr[0] * 0.114 + bgr[1] * 0.587 + bgr[2] * 0.299;
                auto &outputRGB = output.at<cv::Vec3b>(i, j);
                outputRGB[0] = gray;
                outputRGB[1] = gray;
                outputRGB[2] = gray;
            }
        }
    }
    else
    {
        output = result;
    }
}

template<typename Dtype>
void SkeletonDataTransformer<Dtype>::SwapLeftRight(typename MetaData::SkeletonType &target) const
{
    std::unordered_map<std::string, std::pair<int, int>> mapper;
    auto getInited = [&mapper](const std::string& key) -> decltype(mapper)::value_type&
    {
        auto check = mapper.find(key);
        if (check != mapper.end())
            return *check;
        else
            return *(mapper.insert(std::make_pair(key, std::make_pair(-1, -1))).first);
    };

    auto pointTypeDesc = google::protobuf::GetEnumDescriptor<::caffe::ultinous::proto::skeleton::SkeletonPointType>();
    for (int i = 0; i < pointTypeDesc->value_count(); ++i)
    {
        auto name = pointTypeDesc->value(i)->name();
        if (name.length() >= 4 && name.substr(0, 4) == "LEFT")
        {
            auto& value = getInited(name.substr(4));
            value.second.first = i;
        }
        else if (name.length() >= 5 && name.substr(0, 5) == "RIGHT")
        {
            auto& value = getInited(name.substr(5));
            value.second.second = i;
        }
    }

    std::unordered_map<int, int> mirrorType;
    for (const auto &v : mapper)
    {
        if (v.second.first != -1 && v.second.second != -1)
        {
            mirrorType[v.second.first] = v.second.second;
        }
    }

    for (std::size_t i = 0; i < target.joints.size(); ++i)
    {
        auto it = mirrorType.find(target.jointType[i]);
        if (it != mirrorType.end())
        {            
            auto other = std::find(target.jointType.begin(), target.jointType.end(), it->second);
            if (other == target.jointType.end())
                continue;
            int j = std::distance(target.jointType.begin(), other);

            if (i < j)
            {
                std::swap(target.joints[i], target.joints[j]);
                std::swap(target.isVisible[i], target.isVisible[j]);
                std::swap(target.jointType[i], target.jointType[j]);
            }
        }
    }
}


template<typename Dtype>
void SkeletonDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta, const AugmentParameters<Dtype>& aug, 
    const cv::Size& origSize, const cv::Size& finalSize, const cv::Point2f& focusPoint) const
{
    cv::Mat_<Dtype> A;
    if (aug.rotation || aug.scale)
    {
        A = cv::getRotationMatrix2D(cv::Point2f(origSize.width / 2.0f, origSize.height / 2.0f), aug.degree, aug.scaleAmount);
        const auto bbox = transformBoundingBox<Dtype>(origSize, A);
        A(0, 2) -= bbox.x;
        A(1, 2) -= bbox.y;
    }
    else
    {
        A = cv::Mat::eye(2, 3, CV_32FC1);
    }

    if (aug.crop)
    {
      auto center = (aug.rotation || aug.scale)? transform(A, focusPoint) : focusPoint;
      center.x += aug.centroidX;
      center.y += aug.centroidY;
      const auto offsetLeft = boost::numeric_cast<int>(center.x - (aug.cropWidth / 2.0));
      const auto offsetTop = boost::numeric_cast<int>(center.y - (aug.cropHeight / 2.0));

      //LOG(INFO) << "Offset: " << offsetLeft << " " << offsetTop;

      A(0, 2) -= offsetLeft;
      A(1, 2) -= offsetTop;
    }
    //LOG(INFO) << A;

    auto transformPoint = [&finalSize, &A, &aug](const cv::Point2f& p)
    {
        auto res = transform(A, p);
        if (aug.flip)
            res.x = finalSize.width - 1 - res.x;
        return res;
    };

    for (auto& pos : meta.targetPositions)
        pos = transformPoint(pos);
    
    for (auto& person : meta.targetSkeletons)
    {
        for (auto& point : person.joints)
            point = transformPoint(point);

        if (aug.flip)
            SwapLeftRight(person);
    }
}

#endif

INSTANTIATE_CLASS(SkeletonDataTransformer);
} // namespace ultinous
} // namespace caffe
