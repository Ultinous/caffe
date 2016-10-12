#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/ultinous/image_roi_data_layer.hpp"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
ImageROIDataLayer<Dtype>::~ImageROIDataLayer<Dtype>()
{
  this->StopInternalThread();
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
        (new_height > 0 && new_width > 0)) << "Current implementation requires "
            "new_height and new_width to be set at the same time.";

  // Read the annotation file
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "Can not open source file";

  std::string line;
  while (std::getline(infile, line))
  {
    Sample sample;
    std::istringstream iss(line);

    iss >> sample.image_file;
    BBox bbox;
    while( iss >> bbox.x1 >> bbox.y1 >> bbox.x2 >> bbox.y2 >> bbox.cls )
      sample.bboxes.push_back(bbox);

    if( !sample.bboxes.empty() )
      samples.push_back( sample );
  }

  CHECK(!samples.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle())
  {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << samples.size() << " images.";

  // Check if we would need to randomly skip a few data points
  sample_id_ = 0;
  if (this->layer_param_.image_data_param().rand_skip())
  {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(samples.size(), skip) << "Not enough points to skip";
    sample_id_ = skip;
  }
  
  
  cv::Mat cv_img = ReadImageToCVMat(root_folder + samples[0].image_file,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << samples[0].image_file;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  // Set im_info shape
  vector<int> info_shape(2);
  info_shape[0] = 1;
  info_shape[1] = 3;
  top[1]->Reshape(info_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
  {
    this->prefetch_[i].info_.Reshape(info_shape);
  }
  
  vector<int> bboxes_shape(2);
  bboxes_shape[0] = 1;
  bboxes_shape[1] = 5;
  top[2]->Reshape(bboxes_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
  {
    this->prefetch_[i].bboxes_.Reshape(bboxes_shape);
  }
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::ShuffleImages()
{
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(samples.begin(), samples.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageROIDataLayer<Dtype>::load_batch(Batch* batch)
{
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

  ImageROIDataParameter image_roi_data_param = this->layer_param_.image_roi_data_param();
  uint32_t smallerDimensionSize = image_roi_data_param.smallerdimensionsize();
  uint32_t maxSize = image_roi_data_param.maxsize();

  CHECK( batch_size==1 ) << "Currently implemented only for batch_size=1";

  const int samples_size = samples.size();


  // Load image
  timer.Start();
  CHECK_GT(samples_size, sample_id_);
  cv::Mat cv_img = ReadImageToCVMat(root_folder + samples[sample_id_].image_file,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << samples[sample_id_].image_file;
  read_time += timer.MicroSeconds();

  //std::cout << "---- cv_img size: " << cv_img.rows << " " << cv_img.cols << std::endl;

  float scale = 1.0f;

  // resize
  if( smallerDimensionSize > 0 )
  {
    float scale1 = static_cast<float>(smallerDimensionSize)
          / std::min(cv_img.rows, cv_img.cols);
    float scale2 = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));
    scale = std::min(scale1, scale2 );

  } else if( std::max(cv_img.rows, cv_img.cols) > maxSize )
  {
    scale = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));
  }

  if( scale != 1.0f)
  {
    cv::Mat cv_resized;
    cv::resize( cv_img, cv_resized, cv::Size(0,0), scale, scale, cv::INTER_LINEAR );

    cv_img = cv_resized;
  }

  bool mirror = image_roi_data_param.mirror() && ((rand()%2)==1);

  if( mirror )
  {
    cv::Mat cv_flipped;
    cv::flip(cv_img, cv_flipped, 1);
    cv_img = cv_flipped;
  }

  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  timer.Start();
  // Apply transformations (mirror, crop...) to the image
  int offset = batch->data_.offset(0); // (item_id)
  this->transformed_data_.set_cpu_data(prefetch_data + offset);
  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
  trans_time += timer.MicroSeconds();

  // info
  Dtype* prefetch_info = batch->info_.mutable_cpu_data();
  prefetch_info[0] = static_cast<Dtype>(cv_img.rows);
  prefetch_info[1] = static_cast<Dtype>(cv_img.cols);
  prefetch_info[2] = static_cast<Dtype>(scale);

  // bboxes
  vector<int> bboxes_shape(2);
  bboxes_shape[0] = samples[sample_id_].bboxes.size();
  bboxes_shape[1] = 5;
  batch->bboxes_.Reshape(bboxes_shape);

  Dtype* prefetch_bboxes = batch->bboxes_.mutable_cpu_data();

  for( int bboxIx = 0; bboxIx < samples[sample_id_].bboxes.size(); ++bboxIx )
  {
    BBox bbox = samples[sample_id_].bboxes[bboxIx];
    Dtype x1 = static_cast<Dtype>(bbox.x1) * scale;
    Dtype y1 = static_cast<Dtype>(bbox.y1) * scale;
    Dtype x2 = static_cast<Dtype>(bbox.x2) * scale;
    Dtype y2 = static_cast<Dtype>(bbox.y2) * scale;
    Dtype cls = static_cast<Dtype>(bbox.cls);

    if( mirror )
    {
      Dtype temp = x1;
      x1 = cv_img.cols - x2;
      x2 = cv_img.cols - temp;
    }

    //std::cout << "---- x1:" << x1 << " y1:" << y1<< " x2:" << x2 << " y2:" << y1 << " cls:" << cls << std::endl;

    prefetch_bboxes[ 5*bboxIx ] = x1;
    prefetch_bboxes[ 5*bboxIx + 1 ] = y1;
    prefetch_bboxes[ 5*bboxIx + 2 ] = x2;
    prefetch_bboxes[ 5*bboxIx + 3 ] = y2;
    prefetch_bboxes[ 5*bboxIx + 4 ] = cls;
  }

  // go to the next iter
  sample_id_++;
  if (sample_id_ >= samples_size)
  {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    sample_id_ = 0;
    if (this->layer_param_.image_data_param().shuffle())
    {
      ShuffleImages();
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


template <typename Dtype>
ImageROIDataLayer<Dtype>::ImageROIDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    prefetch_[i].info_.mutable_cpu_data();
    prefetch_[i].bboxes_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      prefetch_[i].info_.mutable_gpu_data();
      prefetch_[i].bboxes_.mutable_gpu_data();
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  //std::cout << "---- ImageROIDataLayer<Dtype>::Forward_cpu" << std::endl;

  // Reshape to loaded image.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  //std::cout << "---- batch->data_.count()" << batch->data_.count() << std::endl;

  // Reshape to image info.
  top[1]->ReshapeLike(batch->info_);
  // Copy info.
  caffe_copy(batch->info_.count(), batch->info_.cpu_data(),
      top[1]->mutable_cpu_data());
  //std::cout << "---- batch->info_.count()" << batch->info_.count() << std::endl;

  // Reshape to image info.
  top[2]->ReshapeLike(batch->bboxes_);
  // Copy bbox.
  caffe_copy(batch->bboxes_.count(), batch->bboxes_.cpu_data(),
      top[2]->mutable_cpu_data());
  //std::cout << "---- batch->bboxes_.count()" << batch->bboxes_.count() << std::endl;

  //std::cout << "--- INFO: ";
  //for( int i = 0; i < 3; ++i ) std::cout << batch->info_.cpu_data()[i]<< " " ;
  //std::cout << std::endl;

  //std::cout << "--- BBOXES: ";
  //for( int i = 0; i < batch->bboxes_.count(); ++i ) std::cout << batch->bboxes_.cpu_data()[i]<< " " ;
  //std::cout << std::endl;

  prefetch_free_.push(batch);
}



#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageROIDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageROIDataLayer);
REGISTER_LAYER_CLASS(ImageROIData);

}  // namespace ultinous
}  // namespace caffe
