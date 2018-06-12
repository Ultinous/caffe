#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <boost/make_shared.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/ultinous/adv_image_data_layer.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
AdvImageDataLayer<Dtype>::~AdvImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void AdvImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = boost::lexical_cast<int>(line.substr(pos + 1).c_str());
    lines_.emplace_back(line.substr(0, pos), label);
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_ = boost::make_shared<Caffe::RNG>(prefetch_rng_seed);
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
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
  vector<int> label_shape(2);
  label_shape[0] = batch_size;
  label_shape[1] = 2;                      // label and isAdversarial for each sample
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void AdvImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void AdvImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  string const &root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  static int iterations = 0;

  // datum scales
  int const lines_size = boost::numeric_cast<int>(lines_.size());
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob


//    std::cout << batch_index << m_advImages.empty() << std::endl;
    if( iterations >= m_adv_params.skip_iterations()
        && !m_advImages.empty()
        && (static_cast<float>(rand())/RAND_MAX) < m_adv_params.adversarial_ratio()
      )
    {
      AdversarialImageForClass pair = m_advImages.front();
      m_advImages.pop();

      prefetch_label[2*item_id] = pair.first;
      prefetch_label[2*item_id+1] = 1;

      int offset = batch->data_.offset(item_id);
      Dtype * data = prefetch_data + offset;

      memcpy(data, pair.second.data(), pair.second.size()*sizeof(Dtype));

      /*{
        int channels = batch->data_.channels();
        int height = batch->data_.height();
        int width = batch->data_.width();
        std::cout << offset << " " << channels << " " << height << " " << width << std::endl;
        cv::Mat im( height, width, CV_8UC3, cv::Scalar(0) );
        for( size_t c = 0; c < channels; ++c )
          for( size_t x = 0; x < height; ++x )
            for( size_t y = 0; y < width; ++y )
            {
              int value = 128 + data[c*height*width+x*width+y];
              value = std::max(0, std::min(255, value ) );
              im.data[ channels*(x*width+y) + c ] = value;
            }
        {
          static int CCC = 0; ++CCC;
          std::stringstream ss; ss << "img" << CCC << "_adv.png";
          std::string filename(ss.str());
          cv::imwrite( filename, im );
        }
      }*/

    }
    else
    {
      int label = lines_[lines_id_].second;

      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

      m_unTransformer.transform( cv_img );

      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();



      prefetch_label[2*item_id] = label;
      prefetch_label[2*item_id+1] = 0;

      // go to the next iter
      lines_id_++;
      if (lines_id_ >= lines_size) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.image_data_param().shuffle()) {
          ShuffleImages();
        }
      }
    }


  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

  ++iterations;

}


template <typename Dtype>
void AdvImageDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  int num = top[0]->num();
  int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
  int spatialSize = channels*height*width;


  for( size_t n = 0; n < num; n++ )
  {
    int isAdversarial = top[1]->cpu_data()[2*n + 1];

    if( 0!=isAdversarial ) continue;

    if( m_advImages.size() > m_adv_params.pool_size() )
      break;


    int label = top[1]->cpu_data()[2*n];

    Dtype const* data = top[0]->cpu_data() + n*spatialSize;
    Dtype const* diff = top[0]->cpu_diff() + n*spatialSize;


    Dtype dataLength=0, diffLength = 0, maxDiff = 0;

    for( size_t i = 0; i < spatialSize; ++i )
    {
      diffLength += diff[i]*diff[i];
      dataLength += data[i]*data[i];
      maxDiff = std::max( maxDiff, std::abs(diff[i]) );
    }

    if( maxDiff == 0 ) continue;

//    diffLength /= spatialSize;
    diffLength = sqrt(diffLength);
//    dataLength /= spatialSize;
    dataLength = sqrt(dataLength);

    Dtype maxChange = Dtype(std::abs(m_adv_params.max_pixel_change()));

    AdversarialImage im(spatialSize);
    for( size_t i = 0; i < spatialSize; ++i )
      im[i] = data[i] + std::max(-maxChange, std::min( maxChange, Dtype(m_adv_params.diff_strength()*dataLength*diff[i] / diffLength )) );
      //im[i] = data[i] + 10.0*((diff[i]>0)?1:((diff[i]<0)?-1:0));
      //im[i] = data[i] + 10.0*diff[i]/maxDiff;


    m_advImages.push( AdversarialImageForClass(label, im) );

      /*{
        std::cout << channels << " " << height << " " << width << std::endl;
        cv::Mat cvim( height, width, CV_8UC3, cv::Scalar(0) );
        for( size_t c = 0; c < channels; ++c )
          for( size_t x = 0; x < height; ++x )
            for( size_t y = 0; y < width; ++y )
            {
              cvim.data[ channels*(x*width+y) + c ] = 128+im[c*height*width+x*width+y];
            }
        {
          static int CCC = 0; ++CCC;
          std::stringstream ss; ss << "img" << CCC << "_adv.png";
          std::string filename(ss.str());
          cv::imwrite( filename, cvim );
        }
      }*/
    /*
    static int counter = 0;
    ++counter;

    cv::Mat oriImg( height, width, CV_8UC3, cv::Scalar(0) );
    for( size_t c = 0; c < channels; ++c )
      for( size_t x = 0; x < height; ++x )
        for( size_t y = 0; y < width; ++y )
        {
          int value = 128 + data[c*height*width+x*width+y];
          value = std::max(0, std::min(255, value ) );
          oriImg.data[ channels*(x*width+y) + c ] = value;
        }
    {
      std::stringstream ss; ss << "img" << counter << "_ori.png";
      std::string filename(ss.str());
      cv::imwrite( filename, oriImg );
    }

    cv::Mat advImg( height, width, CV_8UC3, cv::Scalar(0) );
    for( size_t c = 0; c < channels; ++c )
      for( size_t x = 0; x < height; ++x )
        for( size_t y = 0; y < width; ++y )
        {
          int value = 128 + data[c*height*width+x*width+y] + 5.0*diff[c*height*width+x*width+y] / diffLength;
          value = std::max(0, std::min(255, value ) );
          advImg.data[ channels*(x*width+y) + c ] = value;
        }

    {
      std::stringstream ss; ss << "img" << counter << "_diff.png";
      std::string filename(ss.str());
      cv::imwrite( filename, advImg );
    }*/

  }

}


template <typename Dtype>
void AdvImageDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  Backward_cpu( top, propagate_down, bottom );
}


INSTANTIATE_CLASS(AdvImageDataLayer);
REGISTER_LAYER_CLASS(AdvImageData);

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
