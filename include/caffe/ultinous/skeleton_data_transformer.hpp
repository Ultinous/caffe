#pragma once

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include "caffe/blob.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/ultinous/skeleton_meta_data.hpp"

namespace caffe
{
namespace ultinous
{
template<typename Dtype>
struct AugmentParameters
{
    AugmentParameters(SkeletonTransformationParameter param, shared_ptr<Caffe::RNG> rng)
        : param_(std::move(param))
        , rng_(std::move(rng))
    {}

    void randomize()
    {
        generate_scale_multiplier();
        generate_croppad_offset();
        generate_do_mirror();
        generate_rotation_degree();
        generate_force_grayscale();
    }

    bool flip = false;

    bool rotation = false;
    Dtype degree = 0.0;

    bool scale = false;
    Dtype scaleAmount = 1.0;

    bool crop = false;
    Dtype centroidX = 0;
    Dtype centroidY = 0;
    std::size_t cropWidth = 0;
    std::size_t cropHeight = 0;

    bool grayscale = false;
private:
    void generate_scale_multiplier()
    {
        scale = (Rand() < param_.scale_prob());
        scaleAmount = (scale)? ((param_.scale_max() - param_.scale_min()) * Rand() + param_.scale_min()) : 1.0f;
    }

    void generate_croppad_offset()
    {
        crop = param_.crop_size() > 0;
        if (crop)
        {
            const auto dice_x = Rand();
            const auto dice_y = Rand();
            centroidX = (dice_x - 0.5f) * 2.f * param_.center_perterb_max();
            centroidY = (dice_y - 0.5f) * 2.f * param_.center_perterb_max();
            cropWidth = param_.crop_size();
            cropHeight = param_.crop_size();
        }
    }

    void generate_do_mirror()
    {
        flip = param_.mirror() && Rand(2);
    }

    void generate_rotation_degree()
    {
        rotation = param_.rotation();
        degree = (rotation)? ((Rand() - 0.5f) * 2.f * param_.max_rotate_degree()) : 0.0f;
    }

    void generate_force_grayscale()
    {
        grayscale = Rand() < param_.grayscale_prob();
    }

    int Rand(int n)
    {
        CHECK(rng_);
        CHECK_GT(n, 0);
        auto* rng = static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
    }

    Dtype Rand()
    {
        CHECK(rng_);
        auto* rng = static_cast<caffe::rng_t*>(rng_->generator());
        return static_cast<Dtype>((*rng)()) / static_cast<Dtype>(rng->max());
    }

    const SkeletonTransformationParameter param_;
    shared_ptr<Caffe::RNG> rng_;
};


template<typename Dtype>
class SkeletonDataTransformer
{
public:
    SkeletonDataTransformer(const SkeletonTransformationParameter& param, Phase phase);

    /**
    * @brief Initialize the Random number generations if needed by the
    *    transformation.
    */
    void InitRand();

#ifdef USE_OPENCV
    using MetaData = SkeletonMetaData<cv::Point_<Dtype>>;
    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     */
    vector<int> InferDataBlobShape(const cv::Mat& cv_img);

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a cv::Mat
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     * @param cv_mask
     *    cv::Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See image_data_layer.cpp for an example.
     */
    void Transform(const cv::Mat& cv_img, const cv::Mat& cv_mask, MetaData& meta, 
        cv::Mat& transformed_img, cv::Mat& transformed_mask);
#endif  // USE_OPENCV


protected:
#ifdef USE_OPENCV
    void TransformImage(const cv::Mat&, const AugmentParameters<Dtype>&, const std::vector<Dtype>&, const cv::Point2f&, cv::Mat&) const;
    void TransformMetaJoints(MetaData& meta, const AugmentParameters<Dtype>&, const cv::Size&, const cv::Size&, const cv::Point2f&) const;
#endif  // USE_OPENCV

    void SwapLeftRight(typename MetaData::SkeletonType &j) const;

    SkeletonTransformationParameter param_;
    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;
};
} // namespace ultinous
} // namespace caffe
