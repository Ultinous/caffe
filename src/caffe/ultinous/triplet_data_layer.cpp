#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/ultinous/triplet_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
TripletDataLayer<Dtype>::~TripletDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  const string& sourceFile = this->layer_param_.image_data_param().source();

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  m_serialize = this->layer_param_.triplet_data_param().serialize();

  // Init Classification Model
  read( sourceFile, m_imageClassificationModel );

  tripletBatchGenerator = TripletBatchGeneratorPtr(
    new TripletBatchGenerator<Dtype>(
      batch_size
      , m_imageClassificationModel
      , this->layer_param_.triplet_data_param() )
  );

  if( this->layer_param_.triplet_data_param().inputfeatures().size() > 0 )
  {
    std::cout << "TripletDataLayer: Loading input features!" << std::endl;
    const string& inputFeaturesFile = this->layer_param_.triplet_data_param().inputfeatures();
    std::ifstream infile(inputFeaturesFile.c_str(), std::ifstream::binary);

    infile.read(reinterpret_cast<char *>(&m_inputFeatureLength), sizeof(m_inputFeatureLength));

    m_inputFeatures = vector<FeatureVector>(
      m_imageClassificationModel.getImageNum()
      , FeatureVector(m_inputFeatureLength)
    );

    for( size_t i = 0; i < m_imageClassificationModel.getImageNum(); ++i )
      infile.read(
        reinterpret_cast<char *>(&(m_inputFeatures[i].at(0)))
        , m_inputFeatureLength*sizeof(Dtype)
      );

    m_top_shape = vector<int>(2);
    if( m_serialize )
    {
      m_top_shape[0] = 3*batch_size;
      m_top_shape[1] = m_inputFeatureLength;
    }
    else
    {
      m_top_shape[0] = batch_size;
      m_top_shape[1] = 3*m_inputFeatureLength;
    }
  }
  else
  {
    std::ifstream infile(sourceFile.c_str());
    string filename;
    infile >> filename;
    infile.close();

    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder+filename, new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << filename;

    // Use data_transformer to infer the expected blob shape from a cv_image.
    m_top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(m_top_shape);

    // Reshape prefetch_data and top[0] according to the batch_size.
    if( m_serialize )
    {
      m_top_shape[0] = 3*batch_size;
    }
    else
    {
      m_top_shape[0] = batch_size;
      m_top_shape[1] *= 3;
    }
  }

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(m_top_shape);
  }
  top[0]->Reshape(m_top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  // label
  m_label_shape = vector<int>(2);
  m_label_shape[0] = batch_size;
  m_label_shape[1] = 3;

  top[1]->Reshape(m_label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(m_label_shape);
  }

}

// This function is called on prefetch thread
template <typename Dtype>
void TripletDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  const bool inputFeatures = this->layer_param_.triplet_data_param().inputfeatures().size() > 0;

  CHECK(batch->data_.count());
  CHECK(inputFeatures || this->transformed_data_.count());

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
 /* cv::Mat cv_img = ReadImageToCVMat(root_folder + imageClassificationModel.getImageName(0),
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imageClassificationModel.getImageName(0);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] *= 3;*/

  batch->data_.Reshape(m_top_shape);
  batch->label_.Reshape(m_label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  typename TripletBatchGenerator<Dtype>::TripletBatch tripletBatch = tripletBatchGenerator->nextTripletBatch();

  CHECK_EQ(batch_size, tripletBatch.size());

  std::vector< size_t > indices(3);
  std::vector< int > classes(3);

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_EQ(3, tripletBatch[item_id].size());

    for( int i = 0; i < 3; ++i )
    {
      indices[i] = tripletBatch[item_id][i];
      classes[i] = m_imageClassificationModel.getImageClass(indices[i]);
    }
    CHECK_EQ(classes[0], classes[1]);
    CHECK_NE(classes[1], classes[2]);

    // get a blob
    for( int i = 0; i < 3; ++ i ) {
      if( inputFeatures )
      {
        int offset = m_serialize
          ? batch->data_.offset( i*batch_size+item_id )
          : batch->data_.offset(item_id, i*m_inputFeatureLength );
        memcpy( prefetch_data+offset, &(m_inputFeatures[indices[i]].at(0)), m_inputFeatureLength*sizeof(Dtype) );
      }
      else
      {
        std::string fileName = m_imageClassificationModel.getImageName(indices[i]);

        cv::Mat cv_img = ReadImageToCVMat(root_folder + fileName,
            new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << fileName;


        if( this->phase_ == TRAIN )
        {
          // Apply color transformation
          float satCoef = 0.8f + 0.4f*static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
          float lumCoef = 0.8f + 0.4f*static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
          cv::cvtColor( cv_img, cv_img, CV_BGR2HLS );
          for( size_t x = 0; x < cv_img.rows; ++x )
          {
            for( size_t y = 0; y < cv_img.cols; ++y )
            {
              cv::Vec3b &pixel = cv_img.at<cv::Vec3b>( x, y );

              pixel[1] = static_cast<uint8_t>( std::min(255.0f, float(pixel[1])*lumCoef) );
              pixel[2] = static_cast<uint8_t>( std::min(255.0f, float(pixel[2])*satCoef) );

              cv_img.at<cv::Vec3b>( x, y ) = pixel;
            }
          }
          cv::cvtColor( cv_img, cv_img, CV_HLS2BGR );

          // Apply random scale
          float xScale = 0.9+0.2*static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
          float yScale = 0.9+0.2*static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
          cv::Mat cv_scaled;
          cv::resize( cv_img, cv_scaled, cv::Size(), xScale, yScale, cv::INTER_LINEAR );
          cv_img = cv_scaled;
        }

        // Apply transformations (mirror, crop...) to the image
        int offset;
        if( m_serialize )
        {
            offset = batch->data_.offset( i*batch_size+item_id );
        }
        else
        {
          if( this->layer_param_.image_data_param().is_color() )
            offset = batch->data_.offset(item_id, 3*i);
          else
            offset = batch->data_.offset(item_id, i);
        }

        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      }

      prefetch_label[3*item_id+i] = indices[i];
    }
  }
}

INSTANTIATE_CLASS(TripletDataLayer);
REGISTER_LAYER_CLASS(TripletData);

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
