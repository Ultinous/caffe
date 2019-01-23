#pragma once

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include "caffe/blob.hpp"
#include <caffe/util/rng.hpp>

namespace caffe
{
namespace ultinous
{
  template<typename Dtype>
  class CPMDataTransformer
  {
  public:
    CPMDataTransformer(const CPMTransformationParameter& param, Phase phase);

    /**
    * @brief Initialize the Random number generations if needed by the
    *    transformation.
    */
    void InitRand();

    std::vector<int> InferBlobShape(const Datum& datum);
#ifdef USE_OPENCV
    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     */
    vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

    std::vector<int> InferLabelBlobShape(const Datum& datum);

    void Transform(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label = nullptr);
#ifdef USE_OPENCV
    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a cv::Mat
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See image_data_layer.cpp for an example.
     */
    void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label = nullptr);
#endif  // USE_OPENCV


  protected:
    struct AugmentParameters
    {
      AugmentParameters(CPMTransformationParameter param, shared_ptr<Caffe::RNG> rng)
        : param_(std::move(param))
        , rng_(std::move(rng))
      {}

      void randomize()
      {
        generate_scale_multiplier();
        generate_croppad_offset();
        generate_do_mirror();
        generate_rotation_degree();
      }

      bool flip = false;

      bool rotation = false;
      float degree = 0.0;

      bool scale = false;
      float scaleAmount = 1.0;

      bool crop = false;
      cv::Point2f cropOffset = cv::Point2f();
      cv::Size cropSize = cv::Size();

    private:
      void generate_scale_multiplier()
      {
        scale = (Rand() > param_.scale_prob());
        scaleAmount = (scale)? param_.scale_max() - param_.scale_min() * Rand() + param_.scale_min() : 1.0f;
      }

      void generate_croppad_offset()
      {
        crop = param_.crop_size() > 0;
        if (crop)
        {
          const auto dice_x = Rand();
          const auto dice_y = Rand();
          cropOffset = cv::Point2f((dice_x - 0.5f) * 2.f * param_.center_perterb_max(), (dice_y - 0.5f) * 2.f * param_.center_perterb_max());
          cropSize = cv::Size(param_.crop_size(), param_.crop_size());
        }
      }

      void generate_do_mirror()
      {
        flip = param_.mirror() && Rand(2);
      }

      void generate_rotation_degree()
      {
        rotation = param_.rotation();
        degree = (rotation)? (Rand() - 0.5f) * 2.f * param_.max_rotate_degree() : 0.0f;
      }

      int Rand(int n)
      {
        CHECK(rng_);
        CHECK_GT(n, 0);
        auto* rng = static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
      }

      float Rand()
      {
        CHECK(rng_);
        auto* rng = static_cast<caffe::rng_t*>(rng_->generator());
        return static_cast<float>((*rng)()) / static_cast<float>(rng->max());
      }

      const CPMTransformationParameter param_;
      shared_ptr<Caffe::RNG> rng_;
    };

    struct Joints
    {
      enum Visibility
      {
        OCCLUDED = 0,
        VISIBLE = 1,
        OUTSIDE_IMAGE = 2,
        NOT_AVAILABLE = 3
      };

      vector<cv::Point2f> joints;
      vector<Visibility> isVisible;
    };

    struct MetaData
    {
      std::string dataset;
      cv::Size img_size;
      bool isValidation;
      int numOtherPeople;
      int people_index;
      int annolist_index;
      int write_number;
      int total_write_number;
      int epoch;
      cv::Point2f objpos; //objpos_x(float), objpos_y (float)
      float scale_self;
      Joints joint_self; //(3*16)

      std::vector<cv::Point2f> objpos_other; //length is numOtherPeople
      std::vector<float> scale_other; //length is numOtherPeople
      std::vector<Joints> joint_others; //length is numOtherPeople
    };

    /**
     * @brief very specific to genLMDB.py
     */
    void ReadMetaData(MetaData& meta, const Datum& datum) const;

    cv::Mat TransformImage(const cv::Mat&, const AugmentParameters&, const cv::Scalar&, const cv::Point2f&) const;
    void TransformMetaJoints(MetaData& meta, const AugmentParameters&, const cv::Size&, const cv::Size&, const cv::Point2f&) const;
    void Transform(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label = nullptr);

#ifdef USE_OPENCV
    void Transform(const cv::Mat& cv_img, Dtype* transformed_data, Dtype* transformed_label = nullptr);
#endif  // USE_OPENCV

    void generateLabel(Dtype* transformed_label, const MetaData&, const cv::Mat&) const;
    void PutGaussianMaps(Dtype* entry, const cv::Point2f& center, const int stride, const int grid_x, const int grid_y, const float sigma, std::function<Dtype(Dtype,Dtype)> aggregator) const;
    void PutVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f centerA, const cv::Point2f centerB, const int stride, const int grid_x, const int grid_y, const int thre) const;

    void ConvertMetaJoints(MetaData& meta) const;
    void ConvertJoints(Joints &j) const;

    void SwapLeftRight(Joints &j) const;

    template<typename ModelType>
    static std::size_t getNumberOfOutput(const ModelType& model)
    {
      if (model == CPMTransformationParameter::COCO_WITH_NECK)
      {
        return 56;
      }

      return 0;
    }

    template<typename ModelType>
    static std::size_t getNumberOfKeypoints(const ModelType& model)
    {
      switch (model)
      {
        case CPMTransformationParameter::COCO: return 17;
        case CPMTransformationParameter::COCO_WITH_NECK: return 18;
      }
      return 0;
    }

    CPMTransformationParameter param_;
    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;
  };
} // namespace ultinous
} // namespace caffe
