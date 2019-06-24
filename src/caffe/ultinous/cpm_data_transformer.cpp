#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>
#endif

#include "caffe/ultinous/cpm_data_transformer.hpp"
#include <caffe/proto/caffe.pb.h>
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/db.hpp"

namespace caffe
{
namespace ultinous
{
  template<class Result>
  Result *decode(const string &data, const std::size_t idx, Result *buffer, const std::size_t len = 1) {
    if (len) {
      memcpy(buffer, static_cast<const void *>(data.data()+idx), len * sizeof(Result));
    }
    return buffer;
  }

  template<class Result>
  Result decode(const std::string &data, const std::size_t idx) {
    Result res;
    decode(data, idx, &res, 1);
    return res;
  }

  template<>
  std::string decode<std::string>(const std::string &data, const std::size_t idx) {
    return std::string(data.data() + idx);
  }

  template<>
  cv::Point2f decode<cv::Point2f>(const std::string &data, const std::size_t idx) {
    cv::Point2f res;
    res.x = decode<float>(data, idx);
    res.y = decode<float>(data, idx + 4);
    return res;
  }

  SkeletonDatasets extractDatasets(const caffe::CPMDatasetDB& db)
  {
    CHECK_EQ(db.backend(), caffe::CPMDatasetDB::LMDB) << "Only LMDB backend is supported.";
    auto src_db = std::shared_ptr<db::DB>(db::GetDB("lmdb"));
    src_db->Open(db.source(), db::READ);
    auto cursor = std::shared_ptr<db::Cursor>(src_db->NewCursor());

    SkeletonDatasets output;
    while (cursor->valid())
    {
      proto::skeleton::SkeletonDataset dataset;
      dataset.ParseFromString(cursor->value());
      auto &out_dataset = output[dataset.name()];
      for (auto data : dataset.available_types())
      {
        out_dataset.push_back(proto::skeleton::SkeletonPointType(data));
      }
      cursor->Next();
    }
    return output;
  }

  template<typename Dtype>
  CPMDataTransformer<Dtype>::CPMDataTransformer(const CPMTransformationParameter &param, Phase phase)
    : param_(param)
    , phase_(phase)
  {
    // check if we want to use mean_file
    if (param_.has_mean_file()) {
      CHECK_EQ(param_.mean_value_size(), 0) << "Cannot specify mean_file and mean_value at the same time";
      const string &mean_file = param.mean_file();
      if (Caffe::root_solver()) {
        LOG(INFO) << "Loading mean file from: " << mean_file;
      }
      BlobProto blob_proto;
      ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
      data_mean_.FromProto(blob_proto);
    }
    // check if we want to use mean_value
    if (param_.mean_value_size() > 0) {
      CHECK(!param_.has_mean_file()) << "Cannot specify mean_file and mean_value at the same time";
      for (int c = 0; c < param_.mean_value_size(); ++c) {
        mean_values_.push_back(param_.mean_value(c));
      }
    }

    if (param_.has_dataset_db())
    {
      datasets_ = extractDatasets(param_.dataset_db());
    }

    InitRand();
  }

  template<typename Dtype>
  std::vector<int> CPMDataTransformer<Dtype>::InferBlobShape(const Datum &datum) {
    if (datum.encoded()) {
#ifdef USE_OPENCV
      CHECK(!(param_.force_color() && param_.force_gray()))
      << "cannot set both force_color and force_gray";
      cv::Mat cv_img;
      if (param_.force_color() || param_.force_gray()) {
        // If force_color then decode in color otherwise decode in gray.
        cv_img = DecodeDatumToCVMat(datum, param_.force_color());
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      // InferBlobShape using the cv::image.
      return InferBlobShape(cv_img);
#else
      LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    }
    const auto crop_size = param_.crop_size();
    const auto datum_channels = datum.channels();
    const auto datum_height = datum.height();
    const auto datum_width = datum.width();
    // Check dimensions.
    CHECK_GT(datum_channels, 0);

    // Build BlobShape.
    const vector<int> shape = {
            1,
            datum_channels,
            (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : datum_height,
            (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : datum_width
    };
    return shape;
  }

#ifdef USE_OPENCV

    template<typename Dtype>
    vector<int> CPMDataTransformer<Dtype>::InferBlobShape(const cv::Mat &cv_img) {
      const auto crop_size = param_.crop_size();
      const auto img_channels = cv_img.channels();
      const auto img_height = cv_img.rows;
      const auto img_width = cv_img.cols;
      // Check dimensions.
      CHECK_GT(img_channels, 0);

      // Build BlobShape.
      const vector<int> shape = {
              1,
              img_channels,
              (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : img_height,
              (crop_size && phase_ == TRAIN) ? boost::numeric_cast<int>(crop_size) : img_width
      };
      return shape;
    }

#endif

    template<typename Dtype>
    std::vector<int> CPMDataTransformer<Dtype>::InferLabelBlobShape(const Datum &datum) {
      const auto stride = param_.stride();
      CHECK_GT(stride, 0);
      const auto output_model = param_.output_model();

      auto shape = InferBlobShape(datum);
      shape[1] = 2*(boost::numeric_cast<int>(getNumberOfOutput(output_model))+1);
      shape[2] /= stride;
      shape[3] /= stride;
      return shape;
    }

    template<typename Dtype>
    void CPMDataTransformer<Dtype>::Transform(const Datum &datum, Blob<Dtype> *transformed_data,
                                              Blob<Dtype> *transformed_label) {
      // If datum is encoded, decode and transform the cv::image.
      if (datum.encoded()) { // TODO(zssanta): Should be unsupported (FATAL)
#ifdef USE_OPENCV
        CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
        cv::Mat cv_img;
        if (param_.force_color() || param_.force_gray()) {
          // If force_color then decode in color otherwise decode in gray.
          cv_img = DecodeDatumToCVMat(datum, param_.force_color());
        } else {
          cv_img = DecodeDatumToCVMatNative(datum);
        }
        // Transform the cv::image into blob.
        Transform(cv_img, transformed_data, transformed_label);
        return;
#else
        LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
      } else {
        if (param_.force_color() || param_.force_gray()) {
          LOG(ERROR) << "force_color and force_gray only for encoded datum";
        }
      }

      const int crop_size = param_.crop_size();
      const int datum_channels = datum.channels();
      const int datum_height = datum.height();
      const int datum_width = datum.width();

      // Check dimensions.
      const auto data_shape = transformed_data->shape();
      CHECK_GE(data_shape[0], 1);
      CHECK_EQ(data_shape[1], datum_channels);

      LOG_IF(WARNING, data_shape[0] > 1 && !crop_size) <<
        "Batch size is bigger than 1 while there is no crop. Every image of the batch should have the same resolution.";

      if (crop_size) {
        CHECK_EQ(crop_size, data_shape[2]);
        CHECK_EQ(crop_size, data_shape[3]);
      } else {
        CHECK_EQ(datum_height, data_shape[2]);
        CHECK_EQ(datum_width, data_shape[3]);
      }

      if (transformed_label) {
        Transform(datum, transformed_data->mutable_cpu_data(), transformed_label->mutable_cpu_data());
        /*
        LOG(INFO) << transformed_data->shape_string();
        auto* imgSource = transformed_data->cpu_data();
        cv::Mat inputImg = cv::Mat::zeros(data_shape[2], data_shape[3], CV_8UC3);
        for (auto x = 0; x < data_shape[3]; ++x)
        {
          for (auto y = 0; y < data_shape[2]; ++y)
          {
            auto &rgb = inputImg.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; c++) {
              const auto dindex = c * data_shape[2] * data_shape[3] + y * data_shape[3] + x;
              rgb[c] = imgSource[dindex] * 256.0 + 128.0;
            }
          }
        }
        auto extract = [&inputImg](Blob<Dtype>* blob)
        {
          std::vector<cv::Mat> result;
          auto resultVector = blob->mutable_cpu_data();
          const auto resultStep = blob->count(2);
          const auto imageSize = cv::Size(blob->shape(3), blob->shape(2));
          for (int i = 0; i < blob->shape(1); ++i, resultVector += resultStep)
          {
            cv::Mat valueImage(imageSize, CV_32FC1, static_cast<void *>(resultVector));
            cv::resize(valueImage, valueImage, inputImg.size());
            result.push_back(valueImage);
          }
          return result;
        };
        auto labels = extract(transformed_label);
        for (auto i = 0; i < labels.size(); ++i)
        {
          cv::Mat result, lutImg;
          cv::convertScaleAbs(labels[i], result, 255);

          cv::applyColorMap(result, lutImg, cv::COLORMAP_JET);

          cv::addWeighted(inputImg, 0.5, lutImg, 0.5, 0, result);
          cv::imwrite((boost::format("labels_%05d_%05d.png") % (imgSource) % i).str(), result);
        }
        */
      } else {
        Transform(datum, transformed_data->mutable_cpu_data());
      }
    }

#ifdef USE_OPENCV

    template<typename Dtype>
    void CPMDataTransformer<Dtype>::Transform(const cv::Mat &cv_img, Blob<Dtype> *transformed_data,
                                              Blob<Dtype> *transformed_label) {

    }

#endif  // USE_OPENCV

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::Transform(const Datum &datum, Dtype *transformed_data, Dtype *transformed_label)
  {
    const auto &data = datum.data();
    const auto datum_height = datum.height();
    const auto datum_width = datum.width();
    // To do: make this a parameter in caffe.proto
    //const int mode = 5; //related to datum.channels();

    const bool has_uint8 = data.size() > 0;

    CPUTimer timer1;
    timer1.Start();
    cv::Mat img = cv::Mat::zeros(datum_height, datum_width, CV_8UC3);
    cv::Mat mask_miss = cv::Mat::zeros(datum_height, datum_width, CV_8UC1);
    // auto mask_all = cv::Mat::zeros(datum_height, datum_width, CV_8UC1); // only for mode == 6

    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        auto &rgb = img.at<cv::Vec3b>(i, j);
        for (int c = 0; c < 3; c++) {
          const auto dindex = c * img.rows * img.cols + i * img.cols + j;
          rgb[c] = (has_uint8) ? static_cast<Dtype>(static_cast<uint8_t>(data[dindex])) : datum.float_data(dindex);
        }

        {
          const auto dindex = 4 * img.rows * img.cols + i * img.cols + j;
          mask_miss.at<uchar>(i, j) = (has_uint8) ? static_cast<Dtype>(static_cast<uint8_t>(data[dindex]))
                                                  : datum.float_data(dindex);
        }

//        // only for mode == 6
//        {
//          const auto dindex = 5*img.rows.img.cols + i*img.cols + j;
//          mask_all.at<uchar>(i, j) = (has_uint8)? static_cast<Dtype>(static_cast<uint8_t>(data[dindex])) : datum.float_data(dindex);
//        }
      }
    }
    VLOG(2) << "  rgb[:] = datum: " << timer1.MicroSeconds() / 1000.0 << " ms";
    timer1.Start();

    /*
    //color, contract
    int clahe_tileSize = param_.clahe_tile_size();
    int clahe_clipLimit = param_.clahe_clip_limit();
    if(param_.do_clahe())
      clahe(img, clahe_tileSize, clahe_clipLimit);
    if(param_.gray() == 1){
      cv::cvtColor(img, img, CV_BGR2GRAY);
      cv::cvtColor(img, img, CV_GRAY2BGR);
    }
    VLOG(2) << "  color: " << timer1.MicroSeconds()/1000.0 << " ms";
    timer1.Start();
    */

    MetaData meta{};
    ReadMetaData(meta, datum);
    if (param_.input_model() != param_.output_model())
      ConvertMetaJoints(meta);
    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds() / 1000.0 << " ms";
/*
    auto outputImg = img.clone();
    for (const auto& point : meta.joint_self.joints)
    {
      cv::circle(outputImg, point, 5, cv::Scalar(255, 0, 0), CV_FILLED);
    }
    cv::circle(outputImg, meta.objpos, 5, cv::Scalar(0, 0, 255), CV_FILLED);
    for (const auto& other : meta.objpos_other)
    {
      cv::circle(outputImg, other, 5, cv::Scalar(0, 0, 255), CV_FILLED);
    }
    cv::imwrite((boost::format("labels_%05d.png") % (transformed_data)).str(), outputImg);
*/
    timer1.Start();
    AugmentParameters aug(param_, rng_);
    cv::Mat img_aug, mask_miss_aug;
    if (phase_ == TRAIN) {
      aug.randomize();
      img_aug = TransformImage(img, aug, cv::Scalar(128, 128, 128), meta.objpos); // TODO(zssanta): mean value
      if (aug.grayscale)
      {
        cv::Mat gray;
        cv::cvtColor(img_aug, gray, cv::COLOR_BGR2GRAY);
        for (int i = 0; i < img_aug.rows; ++i)
        {
          for (int j = 0; j < img_aug.cols; ++j)
          {
            auto &rgb = img_aug.at<cv::Vec3b>(i, j);
            rgb[0] = gray.at<unsigned char>(i, j);
            rgb[1] = gray.at<unsigned char>(i, j);
            rgb[2] = gray.at<unsigned char>(i, j);
          }
        }
      }
      mask_miss_aug = TransformImage(mask_miss, aug, cv::Scalar(255), meta.objpos);
      TransformMetaJoints(meta, aug, img.size(), img_aug.size(), meta.objpos);
    } else {
      img_aug = img;
      mask_miss_aug = mask_miss;
    }


    const auto offset = img_aug.rows * img_aug.cols;
    for (int i = 0; i < img_aug.rows; ++i)
    {
      for (int j = 0; j < img_aug.cols; ++j)
      {
        const auto& rgb = img_aug.at<cv::Vec3b>(i, j);
        transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
      }
    }

    if (transformed_label)
    {
      generateLabel(transformed_label, meta, mask_miss_aug);

    }
  }

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::PutGaussianMaps(Dtype* entry, const cv::Point2f& center, const int stride, const int grid_x, const int grid_y, const float sigma, std::function<Dtype(Dtype,Dtype)> aggregator) const
  {
    const auto grid_center = stride/2.0 - 0.5;
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        const auto x = grid_center + g_x * stride;
        const auto y = grid_center + g_y * stride;
        const auto d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
        const auto exponent = d2 / 2.0 / sigma / sigma;
        if(exponent > 4.6052) //ln(100) = -ln(1%)
        {
          continue;
        }
        const auto index = g_y*grid_x + g_x;
        entry[index] = aggregator(entry[index], exp(-exponent));
        if(entry[index] > 1)
          entry[index] = 1;
      }
    }
  }

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::PutVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f centerA, const cv::Point2f centerB, const int stride, const int grid_x, const int grid_y, const int distThresh) const
  {
    const auto stridedCenterA = centerA / stride;
    const auto stridedCenterB = centerB / stride;

    const auto min_x = std::max( boost::numeric_cast<int>(std::round(std::min(stridedCenterA.x, stridedCenterB.x)-distThresh)), 0);
    const auto max_x = std::min( boost::numeric_cast<int>(std::round(std::max(stridedCenterA.x, stridedCenterB.x)+distThresh)), grid_x);

    const auto min_y = std::max( boost::numeric_cast<int>(std::round(std::min(stridedCenterA.y, stridedCenterB.y)-distThresh)), 0);
    const auto max_y = std::min( boost::numeric_cast<int>(std::round(std::max(stridedCenterA.y, stridedCenterB.y)+distThresh)), grid_y);

    auto BA_dir = stridedCenterB - stridedCenterA;
    const auto norm_ba = std::sqrt(BA_dir.x*BA_dir.x + BA_dir.y*BA_dir.y);
    if (norm_ba > 1e-10)
      BA_dir /= norm_ba;

    for (int g_y = min_y; g_y < max_y; ++g_y)
    {
      for (int g_x = min_x; g_x < max_x; ++g_x)
      {
        const cv::Point2f point(g_x - stridedCenterA.x, g_y - stridedCenterA.y);
        const auto dist = std::abs(point.x*BA_dir.y - point.y*BA_dir.x);

        if(dist <= distThresh)
        {
          const auto cnt = count.at<uchar>(g_y, g_x);
          const auto index = g_y*grid_x + g_x;
          if (cnt == 0)
          {
            entryX[index] = BA_dir.x;
            entryY[index] = BA_dir.y;
          }
          else
          {
            entryX[index] = (entryX[index]*cnt + BA_dir.x) / (cnt + 1);
            entryY[index] = (entryY[index]*cnt + BA_dir.y) / (cnt + 1);
          }
          count.at<uchar>(g_y, g_x) = boost::numeric_cast<uchar>(cnt + 1);
        }
      }
    }
  }

  struct EnumClassHash
  {
    using result_type = std::size_t;

    template<typename T>
    result_type operator()(T const& value) const noexcept
    {
      return static_cast<result_type>(value);
    }
  };

  // TODO(zssanta): separate labels to multiple tops
  template<typename Dtype>
  void CPMDataTransformer<Dtype>::generateLabel(Dtype* transformed_label, const MetaData& meta, const cv::Mat& mask_miss) const
  {
    CHECK(transformed_label);

    std::vector<proto::skeleton::SkeletonPointType> IND_TO_PROTO =
    {
      proto::skeleton::NOSE,
      proto::skeleton::NECK,
      proto::skeleton::RIGHT_SHOULDER,
      proto::skeleton::RIGHT_ELBOW,
      proto::skeleton::RIGHT_WRIST,
      proto::skeleton::LEFT_SHOULDER,
      proto::skeleton::LEFT_ELBOW,
      proto::skeleton::LEFT_WRIST,
      proto::skeleton::RIGHT_HIP,
      proto::skeleton::RIGHT_KNEE,
      proto::skeleton::RIGHT_ANKLE,
      proto::skeleton::LEFT_HIP,
      proto::skeleton::LEFT_KNEE,
      proto::skeleton::LEFT_ANKLE,
      proto::skeleton::RIGHT_EYE,
      proto::skeleton::LEFT_EYE,
      proto::skeleton::RIGHT_EAR,
      proto::skeleton::LEFT_EAR,
    };

    const auto stride = param_.stride();
    const int grid_x = mask_miss.cols / stride;
    const int grid_y = mask_miss.rows / stride;
    const auto np = getNumberOfOutput(param_.output_model());
    const auto channelOffset = grid_x * grid_y;

    const auto numberOfKeypoints = getNumberOfKeypoints(param_.output_model());
    const auto numberOfEdges = 19;


    // TODO(zssanta): model dependent
    int edges_u[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int edges_v[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};

    const auto dataset = datasets_.find(meta.dataset);
    if (dataset != datasets_.end())
    {
      // weights
      for (int g_x = 0; g_x < grid_x; ++g_x)
      {
        for (int g_y = 0; g_y < grid_y; ++g_y)
        {
          const auto maskMissValue = static_cast<float>(mask_miss.at<uchar>(g_y, g_x)) / 255.0f;
          // vector fields
          for (int i = 0; i < numberOfEdges; ++i)
          {
            const auto has_u = std::find(dataset->second.begin(), dataset->second.end(), IND_TO_PROTO[edges_u[i]-1]) != dataset->second.end();
            const auto has_v = std::find(dataset->second.begin(), dataset->second.end(), IND_TO_PROTO[edges_v[i]-1]) != dataset->second.end();
            const auto weight = (has_u && has_v)? maskMissValue : 0.0;
            transformed_label[(i*2+0)*channelOffset + g_y*grid_x + g_x] = weight;
            transformed_label[(i*2+1)*channelOffset + g_y*grid_x + g_x] = weight;
          }
          // heat maps
          for (int i = 0; i < numberOfKeypoints; ++i)
          {
            const auto has_i = std::find(dataset->second.begin(), dataset->second.end(), IND_TO_PROTO[i]) != dataset->second.end();
            const auto weight = (has_i)? maskMissValue : 0.0;
            transformed_label[(2*numberOfEdges+i)*channelOffset + g_y*grid_x + g_x] = weight;
          }
          // heat map background
          transformed_label[(2*numberOfEdges+numberOfKeypoints)*channelOffset + g_y*grid_x + g_x] = maskMissValue;
          // init others
          for (int i = np+1; i < 2*(np+1); ++i)
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0.0;
        }
      }
    }
    else
    {
      //LOG(WARNING) << "Unknow dataset given: " << meta.dataset;
      for (int g_x = 0; g_x < grid_x; ++g_x)
      {
        for (int g_y = 0; g_y < grid_y; ++g_y)
        {
          for (int i = 0; i < 2*(np+1); ++i)
          {
            const auto weight = (i<=np)? static_cast<float>(mask_miss.at<uchar>(g_y, g_x)) / 255.0f : 0.0f;
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
          }
        }
      }
    }

    for (int i = 0; i < meta.joint_self.joints.size(); ++i)
    {
      if (meta.joint_self.isVisible[i] <= 1)
      {
        PutGaussianMaps(transformed_label + (i + np + 2*numberOfEdges + 1) * channelOffset, meta.joint_self.joints[i], param_.stride(),
                grid_x, grid_y, param_.sigma_heat(), [](Dtype a, Dtype b) { return a+b; }); //self
      }

      for (const auto& other_joint : meta.joint_others)
      {
        if (other_joint.isVisible[i] <= 1)
        {
          PutGaussianMaps(transformed_label + (i + np + 2*numberOfEdges + 1) * channelOffset, other_joint.joints[i], param_.stride(),
                  grid_x, grid_y, param_.sigma_heat(), [](Dtype a, Dtype b) { return a+b; });
        }
      }
    }

    for (int i = 0; i < numberOfEdges; ++i)
    {
      cv::Mat count = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[edges_u[i]-1]<=1 && jo.isVisible[edges_v[i]-1]<=1)
      {
        PutVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                   count, jo.joints[edges_u[i]-1], jo.joints[edges_v[i]-1], param_.stride(), grid_x, grid_y, param_.vec_distance_threshold());
      }

      for(int j = 0; j < meta.numOtherPeople; j++) //for every other person
      {
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[edges_u[i]-1]<=1 && jo2.isVisible[edges_v[i]-1]<=1)
        {
          PutVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                     count, jo2.joints[edges_u[i]-1], jo2.joints[edges_v[i]-1], param_.stride(), grid_x, grid_y, param_.vec_distance_threshold());
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; ++g_y)
    {
      for (int g_x = 0; g_x < grid_x; ++g_x)
      {
        Dtype maximum = 0;
        //second background channel
        for (auto i = np+2*numberOfEdges+1; i < 2*np+1; ++i)
        {
          maximum = std::max(maximum, transformed_label[i*channelOffset + g_y*grid_x + g_x]);
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = std::max(1.0-maximum, 0.0);
      }
    }
  }

  template<typename Point>
  Point transform(const cv::Mat_<typename Point::value_type> &A, const Point &p) {
    // LOG(INFO) << A(1, 0) << " " << A(1, 1) << " " << A(1, 2);
    // LOG(INFO) << A(0, 0) << " " << A(0, 1) << " " << A(0, 2);
    return Point(
      A(0, 0) * p.x + A(0, 1) * p.y + A(0, 2),
      A(1, 0) * p.x + A(1, 1) * p.y + A(1, 2)
    );
  }

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::SwapLeftRight(Joints &j) const
  {
    // TODO(zssanta): model dependent
    if (param_.output_model() == CPMTransformationParameter::COCO_WITH_NECK)
    {
      int right[8] = {3, 4, 5, 9, 10, 11, 15, 17};
      int left[8] = {6, 7, 8, 12, 13, 14, 16, 18};
      for (int i = 0; i < 8; i++)
      {
        int ri = right[i] - 1;
        int li = left[i] - 1;
        std::swap(j.joints[ri], j.joints[li]);
        std::swap(j.isVisible[ri], j.isVisible[li]);
      }
    }
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
  void CPMDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta, const AugmentParameters& aug, const cv::Size& origSize, const cv::Size& finalSize, const cv::Point2f& focusPoint) const
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
      center += aug.cropOffset;
      const auto offsetLeft = boost::numeric_cast<int>(center.x - (aug.cropSize.width / 2.0));
      const auto offsetTop = boost::numeric_cast<int>(center.y - (aug.cropSize.height / 2.0));

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

    meta.objpos = transform(A, meta.objpos);
    for (auto& joint : meta.joint_self.joints)
    {
      joint = transformPoint(joint);
    }
    if (aug.flip)
      SwapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++)
    {
      meta.objpos_other[p] = transformPoint(meta.objpos_other[p]);
      for (auto& joint : meta.joint_others[p].joints)
      {
        joint = transformPoint(joint);
      }
      if (aug.flip)
        SwapLeftRight(meta.joint_others[p]);
    }
  }

  template<typename Dtype>
  cv::Mat CPMDataTransformer<Dtype>::TransformImage(const cv::Mat& input, const AugmentParameters& aug, const cv::Scalar& mean_values, const cv::Point2f& focusPoint) const
  {
    cv::Mat result, resultHelper;
    cv::Point2f center;
    if (aug.rotation || aug.scale)
    {
      center = cv::Point2f(input.cols/2.0f, input.rows/2.0f);
      cv::Mat_<Dtype> A = cv::getRotationMatrix2D(center, aug.degree, aug.scaleAmount);
      const auto bbox = transformBoundingBox<Dtype>(input.size(), A);
      A(0, 2) -= bbox.x;
      A(1, 2) -= bbox.y;

      cv::warpAffine(input, result, A, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, mean_values);

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
      center += aug.cropOffset;
      //LOG(INFO) << "center: " << center;
      const auto offsetLeft = boost::numeric_cast<int>(center.x - (aug.cropSize.width / 2.0));
      const auto offsetTop = boost::numeric_cast<int>(center.y - (aug.cropSize.height / 2.0));

      resultHelper = cv::Mat::zeros(aug.cropSize.height, aug.cropSize.width, result.type()) + mean_values;
      const auto inputRoi = cv::Rect(
              std::max(0, offsetLeft),
              std::max(0, offsetTop),
              std::min(result.cols, offsetLeft + aug.cropSize.width) - std::max(0, offsetLeft),
              std::min(result.rows, offsetTop + aug.cropSize.height) - std::max(0, offsetTop)
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

    return result;
  }

  // TODO(zssanta): refactor this (move to another class with all relevant methods)
  template<typename Dtype>
  void CPMDataTransformer<Dtype>::ConvertMetaJoints(MetaData &meta) const
  {
    ConvertJoints(meta.joint_self);
    for(int i=0;i<meta.joint_others.size();i++)
    {
      ConvertJoints(meta.joint_others[i]);
    }
  }

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::ConvertJoints(Joints &j) const
  {
    std::array<std::array<std::function<Joints(const Joints&)>, CPMTransformationParameter::PoseModel_ARRAYSIZE>, CPMTransformationParameter::PoseModel_ARRAYSIZE> transform{};
    transform[CPMTransformationParameter::COCO][CPMTransformationParameter::COCO] = [](const Joints& _j) { return _j; };
    transform[CPMTransformationParameter::COCO][CPMTransformationParameter::COCO_WITH_NECK] = [this](const Joints& _j)
    {
      Joints jo = _j;
      const auto numberOfKeypoints = this->getNumberOfKeypoints(CPMTransformationParameter::COCO_WITH_NECK);
      int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
      int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
      jo.joints.resize(numberOfKeypoints);
      jo.isVisible.resize(numberOfKeypoints);
      for(int i=0;i<18;i++){
        jo.joints[i] = (_j.joints[COCO_to_ours_1[i]-1] + _j.joints[COCO_to_ours_2[i]-1]) * 0.5;
        if(_j.isVisible[COCO_to_ours_1[i]-1]==2 || _j.isVisible[COCO_to_ours_2[i]-1]==2){
          jo.isVisible[i] = Joints::OUTSIDE_IMAGE;
        }
        else if(_j.isVisible[COCO_to_ours_1[i]-1]==3 || _j.isVisible[COCO_to_ours_2[i]-1]==3){
          jo.isVisible[i] = Joints::NOT_AVAILABLE;
        }
        else {
          jo.isVisible[i] = (_j.isVisible[COCO_to_ours_1[i]-1] == _j.isVisible[COCO_to_ours_2[i]-1] && _j.isVisible[COCO_to_ours_1[i]-1] == Joints::VISIBLE)? Joints::VISIBLE : Joints::OCCLUDED;
        }
      }
      return jo;
    };

    const auto fromModel = param_.input_model();
    const auto toModel = param_.output_model();
    if (transform[fromModel][toModel])
    {
      j = transform[fromModel][toModel](j);
    }
    else
    {
      LOG(FATAL) << "Invalid model transformation";
    }
  }

  template<typename Dtype>
  void CPMDataTransformer<Dtype>::ReadMetaData(MetaData& meta, const Datum& datum) const
  {
    const auto datum_height = datum.height();
    const auto datum_width = datum.width();

    const auto offset_ch3 = boost::numeric_cast<std::size_t>(datum_height * datum_width * 3);
    const auto offset_ch1 = boost::numeric_cast<std::size_t>(datum_width);

    const auto& data = datum.data();
    const auto np_in_lmdb = getNumberOfKeypoints(param_.input_model());

    // ------------------- Dataset name ----------------------
    meta.dataset = decode<std::string>(data, offset_ch3);
    // ------------------- Image Dimension -------------------
    const auto height = decode<float>(data, offset_ch3+offset_ch1);
    const auto width = decode<float>(data, offset_ch3+offset_ch1+4);
    meta.img_size = cv::Size(boost::numeric_cast<int>(width), boost::numeric_cast<int>(height));
    // ----------- Validation, nop, counters -----------------
    meta.isValidation = data[offset_ch3+2*offset_ch1] != 0;
    meta.numOtherPeople = static_cast<int>(data[offset_ch3+2*offset_ch1+1]);
    meta.people_index = static_cast<int>(data[offset_ch3+2*offset_ch1+2]);
    meta.annolist_index = static_cast<int>(decode<float>(data, offset_ch3+2*offset_ch1+3));
    meta.write_number = static_cast<int>(decode<float>(data, offset_ch3+2*offset_ch1+7));
    meta.total_write_number = static_cast<int>(decode<float>(data, offset_ch3+2*offset_ch1+11));

    /*
     TODO(zssanta): Refactor this (doesn't depend on data)
    // count epochs according to counters
    static int cur_epoch = -1;
    if(meta.write_number == 0){
      cur_epoch++;
    }
    meta.epoch = cur_epoch;
    if(meta.write_number % 1000 == 0){
      LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
                << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number
                << "; meta.total_write_number: " << meta.total_write_number << "; meta.epoch: " << meta.epoch;
    }
     */

    /*
     * TODO(zssanta): Refactor this (doesn't depend on data)
    if(param_.aug_way() == "table" && !is_table_set){
      SetAugTable(meta.total_write_number);
      is_table_set = true;
    }
     */

    auto visibilityConvert = [](const float _visibility, const cv::Point2f& _joint, const cv::Size& _img_size)
    {
      if (_visibility == 3)
      {
        return Joints::NOT_AVAILABLE;
      }
      else if (_joint.x < 0 || _joint.y < 0 || _joint.x >= _img_size.width || _joint.y >= _img_size.height)
      {
        return Joints::OUTSIDE_IMAGE;
      } else {
        return (_visibility == 0)? Joints::OCCLUDED : Joints::VISIBLE;
      }
    };

    // ------------------- objpos -----------------------
    meta.objpos.x = decode<float>(data, offset_ch3+3*offset_ch1);
    meta.objpos.y = decode<float>(data, offset_ch3+3*offset_ch1+4);
    meta.objpos -= cv::Point2f(1,1); // from matlab 1-index to c++ 0-index
    // ------------ scale_self, joint_self --------------
    meta.scale_self = decode<float>(data, offset_ch3+4*offset_ch1);
    meta.joint_self.joints.resize(np_in_lmdb);
    meta.joint_self.isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++)
    {
      meta.joint_self.joints[i].x = decode<float>(data, offset_ch3+5*offset_ch1+4*i);
      meta.joint_self.joints[i].y = decode<float>(data, offset_ch3+6*offset_ch1+4*i);
      meta.joint_self.joints[i] -= cv::Point2f(1,1); // from matlab 1-index to c++ 0-index
      const auto isVisible = decode<float>(data, offset_ch3+7*offset_ch1+4*i);
      meta.joint_self.isVisible[i] = visibilityConvert(isVisible, meta.joint_self.joints[i], meta.img_size);
    }

    //others (7 lines loaded)
    meta.objpos_other.resize(boost::numeric_cast<std::size_t>(meta.numOtherPeople));
    meta.scale_other.resize(boost::numeric_cast<std::size_t>(meta.numOtherPeople));
    meta.joint_others.resize(boost::numeric_cast<std::size_t>(meta.numOtherPeople));
    for(int p=0; p<meta.numOtherPeople; p++)
    {
      meta.objpos_other[p].x = decode<float>(data, offset_ch3+(8+p)*offset_ch1);
      meta.objpos_other[p].y = decode<float>(data, offset_ch3+(8+p)*offset_ch1+4);
      meta.objpos_other[p] -= cv::Point2f(1,1); // from matlab 1-index to c++ 0-index
      meta.scale_other[p] = decode<float>(data, offset_ch3+(8+meta.numOtherPeople)*offset_ch1+4*p);
    }
    //8 + numOtherPeople lines loaded
    for(int p=0; p<meta.numOtherPeople; p++)
    {
      meta.joint_others[p].joints.resize(np_in_lmdb);
      meta.joint_others[p].isVisible.resize(np_in_lmdb);
      for(int i=0; i<np_in_lmdb; i++)
      {
        meta.joint_others[p].joints[i].x = decode<float>(data, offset_ch3+(9+meta.numOtherPeople+3*p)*offset_ch1+4*i);
        meta.joint_others[p].joints[i].y = decode<float>(data, offset_ch3+(9+meta.numOtherPeople+3*p+1)*offset_ch1+4*i);
        meta.joint_others[p].joints[i] -= cv::Point2f(1,1); // from matlab 1-index to c++ 0-index
        const auto isVisible = decode<float>(data, offset_ch3+(9+meta.numOtherPeople+3*p+2)*offset_ch1+4*i);
        meta.joint_others[p].isVisible[i] = visibilityConvert(isVisible, meta.joint_others[p].joints[i], meta.img_size);
      }
    }
  }

  template <typename Dtype>
  void CPMDataTransformer<Dtype>::InitRand()
  {
    // TODO(zssanta): Add stuff for needs_rand
    const bool needs_rand = phase_ == TRAIN;
    if (needs_rand) {
      const unsigned int rng_seed = caffe_rng_rand();
      rng_.reset(new Caffe::RNG(rng_seed));
    } else {
      rng_.reset();
    }
  }

  INSTANTIATE_CLASS(CPMDataTransformer);
} // namespace ultinous
} // namespace caffe
