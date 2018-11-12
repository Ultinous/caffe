#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/ultinous/demography_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
DemographyDataLayer<Dtype>::~DemographyDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DemographyDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  m_maxAge = this->layer_param_.demography_data_param().max_age();
  m_intervalLength = this->layer_param_.demography_data_param().interval_length();
  m_additionalIntervals = this->layer_param_.demography_data_param().additional_intervals();
  m_numIntervals = int(2*m_additionalIntervals + float(m_maxAge) / m_intervalLength);
  LOG(INFO) << "demography_data_layer numIntervals:" << m_numIntervals;

  //CHECK( 0 == (m_maxAge) % m_intervalLength );

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string imageName;
  int age, gender;
  while(infile >> imageName >> age >> gender)
  {
    if( age >= m_maxAge ) age = m_maxAge;
    m_files[age][gender].push_back( imageName );
    m_indices[age][gender] = 0;
  }
  CHECK(!m_files.empty()) << "File is empty";

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imageName, new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imageName;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();


  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back( m_numIntervals + 1 ); // age+gender

  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

static double ncdf( double x, double mu, double sigma )
{
  return 0.5 + 0.5 * erf( (x-mu)/(sigma*std::sqrt(2.0)) );
}

// This function is called on prefetch thread
template <typename Dtype>
void DemographyDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();

    int age = rand() % m_maxAge;
    int gender = rand() % 2;
    if( m_files[age][gender].empty() )
    {
      int age1 = age-1;
      while( age1 > 0 && m_files[age1][gender].empty() ) --age1;

      int age2 = age+1;
      while( age2 < m_maxAge && m_files[age2][gender].empty() ) ++age2;

      CHECK( age1 >= 0 || age2 < m_maxAge );

      if( age1 < 0 ) // if only age2 is valid
        age = age2;
      else if( age2>=m_maxAge ) // if only age1 is valid
        age = age1;
      else // if boath age1 and age2 are valid
      {
        float prob = float(age - age1) / float(age2-age1);
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if( r < prob )
          age = age1;
        else
          age = age2;
      }
    }

    size_t imageIx = m_indices[age][gender]++;
    m_indices[age][gender] %= m_files[age][gender].size();

    std::string imageFile = m_files[age][gender][imageIx];

    cv::Mat cv_img = ReadImageToCVMat(root_folder + imageFile,
        new_height, new_width, is_color);

    CHECK(cv_img.data) << "Could not load " << imageFile;

    m_unTransformer.transform( cv_img );

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    int labelOffset = batch->label_.offset(item_id);  // item_id * (m_numIntervals+1)

    //std::cout << age;

    for( int intervalIx = 0; intervalIx < m_numIntervals; ++intervalIx )
    {
      double x1 = (intervalIx-m_additionalIntervals)*m_intervalLength;
      double x2 = x1 + m_intervalLength;
      double sigma = this->layer_param_.demography_data_param().age_stddev();

      prefetch_label[ labelOffset + intervalIx ] = ncdf( x2, age, sigma ) - ncdf( x1, age, sigma );
      //std::cout << " " << prefetch_label[ labelOffset + intervalIx ];
    }

    prefetch_label[ labelOffset + m_numIntervals ] = gender;
    //std::cout << std::endl << std::endl;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DemographyDataLayer);
REGISTER_LAYER_CLASS(DemographyData);

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
