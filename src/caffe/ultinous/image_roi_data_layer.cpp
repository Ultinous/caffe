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
#include <iostream>

#include <chrono>

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
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  int new_height = image_data_param.new_height();
  int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const string root_folder = image_data_param.root_folder();
  m_batch_size = image_data_param.batch_size();

  ImageROIDataParameter image_roi_data_param = this->layer_param_.image_roi_data_param();
  const uint32_t inImgNum = image_roi_data_param.in_img_num();
  bool randomScale = (image_roi_data_param.has_rnd_scale_min() || image_roi_data_param.has_rnd_scale_max());
  bool randomCrop = ( image_roi_data_param.has_crop_height() && image_roi_data_param.has_crop_width() );
  int crop_height=0, crop_width=0;
  if (randomCrop)
  {
    crop_height = image_roi_data_param.crop_height();
    crop_width = image_roi_data_param.crop_width();
  }

  CHECK( !(randomCrop && randomScale) )
    << "Can not randomly crop and resize at the same time.";
  CHECK( (new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0) )
    << "Current implementation requires "
       "new_height and new_width to be set at the same time.";

  // Read the annotation file
  const string& source = image_data_param.source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "Can not open source file";

  std::string line, image_file;
  while (std::getline(infile, line))
  {
    Sample sample;
    std::istringstream iss(line);

    for (int i=0; i<inImgNum; ++i)
    {
      iss >> image_file;
      sample.image_files.push_back(image_file);
    }

    BBox bbox;
    int tmp;
    while( iss >> bbox.x1 >> bbox.y1 >> bbox.x2 >> bbox.y2 >> tmp )
      sample.bboxes.push_back(bbox);
    if( !sample.bboxes.empty() )
      samples.push_back( sample );
  }

  CHECK(!samples.empty()) << "File is empty";

  if (image_data_param.shuffle())
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
  if (image_data_param.rand_skip())
  {
    unsigned int skip = caffe_rng_rand() %
        image_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(samples.size(), skip) << "Not enough points to skip";
    sample_id_ = skip;
  }

  if (randomCrop)
  {
    new_height = crop_height;
    new_width = crop_width;

    this->m_gen = std::mt19937(this->m_rd());
    srand(time(0));
  }

  cv::Mat cv_img = ReadImageToCVMat(root_folder + samples[0].image_files[0],
                                    new_height, new_width, is_color);

  CHECK(image_roi_data_param.pad()>=1);
  if(image_roi_data_param.pad()>1)
  {
    int pad_h = ( image_roi_data_param.pad() - ( cv_img.rows % image_roi_data_param.pad() ) ) % image_roi_data_param.pad();
    int pad_w = ( image_roi_data_param.pad() - ( cv_img.cols % image_roi_data_param.pad() ) ) % image_roi_data_param.pad();
    if( pad_h != 0 || pad_w != 0 )
      cv::copyMakeBorder(cv_img, cv_img, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
  }

  CHECK(cv_img.data) << "Could not load " << samples[0].image_files[0];
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);

  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0].
  top_shape[0] = m_batch_size;
  top_shape[1] = cv_img.channels() * inImgNum;
  top[0]->Reshape(top_shape);

  top_shape[0] = 1;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(top_shape);


  // Set im_info shape
  vector<int> info_shape(2);
  info_shape[0] = m_batch_size;
  info_shape[1] = 8;
  top[1]->Reshape(info_shape);

  info_shape[0] = 1;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].info_.Reshape(info_shape);

  vector<int> bboxes_shape(2);
  bboxes_shape[0] = 1;
  bboxes_shape[1] = 5;
  top[2]->Reshape(bboxes_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].bboxes_.Reshape(bboxes_shape);
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
//  TODO
//  std::string name;
//  {
//    using namespace std::chrono;
//    milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
//    std::stringstream ss;
//    ss << ms.count();
//    name = ss.str();
//  }

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  ImageROIDataParameter image_roi_data_param = this->layer_param_.image_roi_data_param();
  bool randomScale = (image_roi_data_param.has_rnd_scale_min() || image_roi_data_param.has_rnd_scale_max());
  bool randomCrop = (image_roi_data_param.has_crop_height() && image_roi_data_param.has_crop_width());
  uint32_t smallerDimensionSize = image_roi_data_param.smallerdimensionsize();
  uint32_t maxSize = image_roi_data_param.maxsize();
  uint32_t inImgNum = image_roi_data_param.in_img_num();
  int crop_height=0, crop_width=0;
  if (randomCrop)
  {
    crop_height = image_roi_data_param.crop_height();
    crop_width = image_roi_data_param.crop_width();
  }

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;

  const int samples_size = samples.size();
  bool done = false;
  while ( !done )
  {
    CPUTimer timer;

    // Copy bounding boxes
    BBoxes boxes = samples[sample_id_].bboxes;

    // Load image
    timer.Start();
    CHECK_GT(samples_size, sample_id_);

    cv::Mat cv_img = readMultiChannelImage(inImgNum, root_folder, new_height, new_width, is_color);

//    TODO
//    cv::Mat cv_tmp = cv_img.clone();
//    for (auto it=samples[sample_id_].bboxes.begin(); it!=samples[sample_id_].bboxes.end(); ++it)
//      cv::rectangle(cv_tmp, cv::Point(it->x1, it->y1), cv::Point(it->x2, it->y2), cv::Scalar(0, 255, 0));

    read_time += timer.MicroSeconds();

    bool skip = false;
    int crop_x=0, crop_y=0, pad_x=0, pad_y=0;
    int source_width = cv_img.cols, source_height = cv_img.rows;
    int source_x1=0, source_y1=0;
    int source_x2=(randomCrop ? crop_width-1 : source_width-1);
    int source_y2=(randomCrop ? crop_height-1 : source_height - 1);
    if (randomCrop)
      skip = doRandomCrop(
        boxes, crop_height, crop_width, crop_x, crop_y, cv_img, pad_x, pad_y,
        source_height, source_width, source_x1, source_x2, source_y1, source_y2
      );

    if (!skip)
    {
      float scale = 1.0f;

      // resize
      if (randomScale)
      {
        float scale_min = image_roi_data_param.has_rnd_scale_min() ? image_roi_data_param.rnd_scale_min() : 1;
        float scale_max = image_roi_data_param.has_rnd_scale_max() ? image_roi_data_param.rnd_scale_max() : 1;
        CHECK(scale_max - scale_min > 0);
        if ((rand() % 2) == 1)
          scale = scale_min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (scale_max - scale_min)));
      }

      if (smallerDimensionSize > 0)
      {
        float scale1 = static_cast<float>(smallerDimensionSize)
                       / std::min(cv_img.rows, cv_img.cols);
        float scale2 = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));
        scale = std::min(scale1, scale2);

      }
      else if (std::max(cv_img.rows, cv_img.cols) > maxSize)
        scale = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));

      if (scale != 1.0f)
      {
        cv::Mat cv_resized;
        cv::resize(cv_img, cv_resized, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);

        cv_img = cv_resized;
      }

      CHECK(image_roi_data_param.pad() >= 1);
      if (image_roi_data_param.pad() > 1)
      {
        int pad_h =
          (image_roi_data_param.pad() - (cv_img.rows % image_roi_data_param.pad())) % image_roi_data_param.pad();
        int pad_w =
          (image_roi_data_param.pad() - (cv_img.cols % image_roi_data_param.pad())) % image_roi_data_param.pad();
        if (pad_h != 0 || pad_w != 0)
          cv::copyMakeBorder(cv_img, cv_img, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
      }

      bool mirror = image_roi_data_param.mirror() && ((rand() % 2) == 1);

      if (mirror)
      {
        cv::Mat cv_flipped;
        cv::flip(cv_img, cv_flipped, 1);
        cv_img = cv_flipped;
      }

      vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch.
      top_shape[0] = 1;
      batch->data_.Reshape(top_shape);

      Dtype *prefetch_data = batch->data_.mutable_cpu_data();

      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(0); // (item_id)
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      // info
      if (mirror)
      {
        Dtype temp = source_x1;
        source_x1 = cv_img.cols - source_x2 - 1;
        source_x2 = cv_img.cols - temp - 1;
      }

      Dtype *prefetch_info = batch->info_.mutable_cpu_data();
      prefetch_info[0] = static_cast<Dtype>(cv_img.rows);
      prefetch_info[1] = static_cast<Dtype>(cv_img.cols);
      prefetch_info[2] = static_cast<Dtype>(scale);
      prefetch_info[3] = static_cast<Dtype>(source_x1);
      prefetch_info[4] = static_cast<Dtype>(source_y1);
      prefetch_info[5] = static_cast<Dtype>(source_x2);
      prefetch_info[6] = static_cast<Dtype>(source_y2);

      // bboxes
      BBoxes finalBoxes;
      for (BBox bbox : boxes)
      {
        if (randomCrop)
        {
          if (bbox.x1 >= bbox.x2 || bbox.y1 >= bbox.y2)
            continue;

//          if (bbox.x2-bbox.x1+1 > 512 || bbox.y2-bbox.y1+1 > 512)
//            continue;

          if (mirror)
          {
            Dtype temp = bbox.x1;
            bbox.x1 = cv_img.cols - bbox.x2 - 1;
            bbox.x2 = cv_img.cols - temp - 1;
          }

//          int bw=std::min( (int)ceil((bbox.x2-bbox.x1+1)*0.6), 128 ), bh=std::min( (int)ceil((bbox.y2-bbox.y1+1)*0.6), 128 );
          int bw=ceil((bbox.x2-bbox.x1+1)*0.6), bh=ceil((bbox.y2-bbox.y1+1)*0.6);

          if (bbox.x1 < -bw + source_x1)
            continue;
          if (bbox.y1 < -bh + source_y1)
            continue;
          if (bbox.x2 > source_x2 + bw)
            continue;
          if (bbox.y2 > source_y2 + bh)
            continue;
        }

        bbox.x1 *= scale;
        bbox.y1 *= scale;
        bbox.x2 *= scale;
        bbox.y2 *= scale;

        finalBoxes.push_back(bbox);
      }

      if (finalBoxes.size() > 0)
      {
        prefetch_info[7] = finalBoxes.size();

        vector<int> bboxes_shape(2);
        bboxes_shape[0] = finalBoxes.size();
        bboxes_shape[1] = 5;
        batch->bboxes_.Reshape(bboxes_shape);

        Dtype *prefetch_bboxes = batch->bboxes_.mutable_cpu_data();

        for (int bboxIx = 0; bboxIx < finalBoxes.size(); ++bboxIx)
        {
          BBox bbox = finalBoxes[bboxIx];
          Dtype x1 = static_cast<Dtype>(bbox.x1);
          Dtype y1 = static_cast<Dtype>(bbox.y1);
          Dtype x2 = static_cast<Dtype>(bbox.x2);
          Dtype y2 = static_cast<Dtype>(bbox.y2);

//          cv::rectangle(cv_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));//TODO

          prefetch_bboxes[5 * bboxIx] = x1;
          prefetch_bboxes[5 * bboxIx + 1] = y1;
          prefetch_bboxes[5 * bboxIx + 2] = x2;
          prefetch_bboxes[5 * bboxIx + 3] = y2;
          prefetch_bboxes[5 * bboxIx + 4] = 1;
        }

//        cv::imwrite("debug/"+name+"_crop.jpg", cv_img);//TODO
//        cv::imwrite("debug/"+name+".jpg", cv_tmp);//TODO

        done = true;
      }
//      else
//        std::cout << "image skip reason 2\n";//TODO
    }
//    else
//      std::cout << "image skip reason 1\n";//TODO

    // go to the next iter
    sample_id_++;
    if (sample_id_ >= samples_size)
    {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      sample_id_ = 0;
      if (image_data_param.shuffle())
        ShuffleImages();
    }
  } // loop: while not done

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


template <typename Dtype>
ImageROIDataLayer<Dtype>::ImageROIDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param)
    , prefetch_free_(), prefetch_full_()
    , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
{
  for (int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);

  auto transform_param = this->layer_param_.transform_param();
  if ( transform_param.mean_value_size() > 0 )
    for (int c = 0; c < transform_param.mean_value_size(); ++c)
      m_mean_values.push_back(transform_param.mean_value(c));
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::LayerSetUp
(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i)
  {
    prefetch_[i].data_.mutable_cpu_data();
    prefetch_[i].info_.mutable_cpu_data();
    prefetch_[i].bboxes_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU)
  {
    for (int i = 0; i < PREFETCH_COUNT; ++i)
    {
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
void ImageROIDataLayer<Dtype>::InternalThreadEntry()
{
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU)
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif

  try {
    while (!must_stop())
    {
      Batch* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU)
      {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  }
  catch (boost::thread_interrupted&)
  {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU)
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif
}

template <typename Dtype>
inline cv::Mat ImageROIDataLayer<Dtype>::readMultiChannelImage(uint32_t inImgNum, const string& root_folder, int new_height, int new_width, bool is_color)
{
  vector<cv::Mat> slices;
  slices.reserve(inImgNum);
  for (int i = 0; i < inImgNum; ++i) {
    if (i == 0) {
      slices.push_back(
          ReadImageToCVMat(root_folder + samples[sample_id_].image_files[0], new_height, new_width, is_color));
      CHECK(slices.back().data && slices.back().rows != 0 && slices.back().cols != 0) << "Could not load "
                                                                                      << samples[sample_id_].image_files[0];
    } else {
      slices.push_back(
          ReadImageToCVMat(root_folder + samples[sample_id_].image_files[i], new_height, new_width, is_color));

      CHECK(slices.back().data) << "Could not load " << samples[sample_id_].image_files[i];
      CHECK(slices[0].cols == slices.back().cols && slices[0].rows == slices.back().rows)
      << "Resolution mismatch, expected " << slices[0].cols << "x" << slices[0].rows
      << " got " << slices.back().cols << "x" << slices.back().rows
      << " in " << samples[sample_id_].image_files[i];
    }
    m_unTransformer.transform(slices.back());
  }

  cv::Mat cv_img;
  cv::merge(slices, cv_img);

  return cv_img;
}

template <typename Dtype>
inline bool ImageROIDataLayer<Dtype>::doRandomCrop(
  BBoxes& boxes, const int crop_height, const int crop_width, int& crop_x, int& crop_y, cv::Mat& cv_img,
  int& pad_x, int& pad_y, int& source_height, int& source_width,
  int& source_x1, int& source_x2, int& source_y1, int& source_y2
){
  std::uniform_int_distribution<> dis;
  bool skip = true;

  int dy = crop_height - source_height;
  int dx = crop_width - source_width;
  if (dy >= 0 && dx >= 0)
  {
    if (dx == 0)
      pad_x = 0;
    else
    {
      dis = std::uniform_int_distribution<>(0, dx);
      pad_x = dis(this->m_gen);
    }

    if (dy == 0)
      pad_y = 0;
    else
    {
      dis = std::uniform_int_distribution<>(0, dy);
      pad_y = dis(this->m_gen);
    }

    source_x1 = pad_x;
    source_y1 = pad_y;
    source_x2 = pad_x + source_width  - 1;
    source_y2 = pad_y + source_height - 1;

    if (dx != 0 || dy != 0)
    {
      cv::copyMakeBorder(cv_img, cv_img, pad_y, dy - pad_y, pad_x, dx - pad_x, cv::BORDER_CONSTANT,
                         cv::Scalar(127,127,127));

      for (auto it=boxes.begin(); it!=boxes.end(); ++it)
      {
        it->x1 += pad_x;
        it->x2 += pad_x;
        it->y1 += pad_y;
        it->y2 += pad_y;
      }
    }

    skip = false;

  }
  else
  {
    if (dy > 0)
    {
      pad_y = dy;
      cv::copyMakeBorder(cv_img, cv_img, pad_y, pad_y, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    }
    else if (dx > 0)
    {
      pad_x = dx;
      cv::copyMakeBorder(cv_img, cv_img, 0, 0, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    }
    for (auto it=boxes.begin(); it!=boxes.end(); ++it)
    {
      it->x1 += pad_x;
      it->x2 += pad_x;
      it->y1 += pad_y;
      it->y2 += pad_y;
    }

    std::vector<int> vx1,vx2,vy1,vy2;
    for (BBox bb : boxes)
    {
      vx1.push_back( std::min(std::max(0,bb.x1), cv_img.cols-1) );
      vy1.push_back( std::min(std::max(0,bb.y1), cv_img.rows-1) );
      vx2.push_back( std::min(std::max(0,bb.x2), cv_img.cols-1) );
      vy2.push_back( std::min(std::max(0,bb.y2), cv_img.rows-1) );
    }

    int x1,x2,y1,y2;
    while (true)
    {
      if (vx1.size() == 0)
        break;

      x1 = *std::min_element(vx1.begin(),vx1.end());
      y1 = *std::min_element(vy1.begin(),vy1.end());
      x2 = *std::max_element(vx2.begin(),vx2.end());
      y2 = *std::max_element(vy2.begin(),vy2.end());

      if (x1 >= x2 || y1 >= y2)
        break;

      int h = y2-y1+1, w = x2-x1+1;

      bool vertical = h > crop_height;
      bool horizontal = w > crop_width;

//          TODO
//          std::cout << '\n';
//          std::cout << root_folder + samples[sample_id_].image_files.back() << '\n';
//          std::cout << name << '\n';
//          std::cout << "orig: " << cv_img.cols << 'x' << cv_img.rows << '\n';
//          std::cout << "new:  " << crop_width << 'x' << crop_height << '\n';
//          std::cout << "bb:   " << x1 << ' ' << x2 << ' ' << y1 << ' ' << y2 << '\n';
//          std::cout << "v: " << vertical << " h: " << horizontal << '\n';

      if (vertical || horizontal)
      {
        std::vector<size_t> n,e,s,w, indicesToRemove;
        for (int i = 0; i < vx1.size(); ++i)
        {
          if (vx1[i] == x1)
            w.push_back(i);
          if (vy1[i] == y1)
            n.push_back(i);
          if (vx2[i] == x2)
            e.push_back(i);
          if (vy2[i] == y2)
            s.push_back(i);
        }

//            TODO
//            std::cout << "n: ";
//            for (auto el : n)
//              std::cout << el << ' ';
//            std::cout << "\ne: ";
//            for (auto el : e)
//              std::cout << el << ' ';
//            std::cout << "\ns: ";
//            for (auto el : s)
//              std::cout << el << ' ';
//            std::cout << "\nw: ";
//            for (auto el : w)
//              std::cout << el << ' ';
//            std::cout << '\n';

        if (vertical && horizontal)
        {
          std::sort(n.begin(), n.end());
          std::sort(e.begin(), e.end());
          std::sort(s.begin(), s.end());
          std::sort(w.begin(), w.end());

          std::vector<size_t> u;
          std::set_union(n.begin(), n.end(), e.begin(), e.end(), std::back_inserter(u));
          std::vector<size_t> ne = u;

          u.clear();
          std::set_union(s.begin(), s.end(), e.begin(), e.end(), std::back_inserter(u));
          std::vector<size_t> se = u;

          u.clear();
          std::set_union(s.begin(), s.end(), w.begin(), w.end(), std::back_inserter(u));
          std::vector<size_t> sw = u;

          u.clear();
          std::set_union(n.begin(), n.end(), w.begin(), w.end(), std::back_inserter(u));
          std::vector<size_t> nw = u;

          int d = rand() % 4;

          if (d == 0)
            indicesToRemove.insert( indicesToRemove.end(), ne.begin(), ne.end() );
          else if (d == 1)
            indicesToRemove.insert( indicesToRemove.end(), se.begin(), se.end() );
          else if (d == 2)
            indicesToRemove.insert( indicesToRemove.end(), sw.begin(), sw.end() );
          else if (d == 3)
            indicesToRemove.insert( indicesToRemove.end(), nw.begin(), nw.end() );
        }
        else if (vertical)
        {
          if ( s.size()==0 || (n.size()!=0 && rand()%2) )
            indicesToRemove.insert( indicesToRemove.end(), n.begin(), n.end() );
          else
            indicesToRemove.insert( indicesToRemove.end(), s.begin(), s.end() );
        }
        else if (horizontal)
        {
          if ( w.size()==0 || (e.size()!=0 && rand()%2) )
            indicesToRemove.insert( indicesToRemove.end(), e.begin(), e.end() );
          else
            indicesToRemove.insert( indicesToRemove.end(), w.begin(), w.end() );
        }

        std::sort( indicesToRemove.begin(), indicesToRemove.end(), std::greater<size_t>() );

//            TODO
//            std::cout << "to remove from " << vx1.size() << ": ";
//            for (auto e : indicesToRemove)
//            {
//              std::cout << e << ' ';
//            }
//            std::cout << '\n';

        for (size_t index : indicesToRemove)
        {
          vx1.erase(vx1.begin() + index);
          vx2.erase(vx2.begin() + index);
          vy1.erase(vy1.begin() + index);
          vy2.erase(vy2.begin() + index);
        }
      }
      else
      {
        skip = false;
        break;
      }
    }
    if (!skip)
    {
      int crop_min_x = std::max(0, x2 - crop_width + 1);
      int crop_min_y = std::max(0, y2 - crop_height + 1);
      int crop_max_x = std::min(x1, cv_img.cols - crop_width);
      int crop_max_y = std::min(y1, cv_img.rows - crop_height);

      if (crop_min_y == crop_max_y)
        crop_y = crop_min_y;
      else
      {
        dis = std::uniform_int_distribution<>(crop_min_y, crop_max_y);
        crop_y = dis(this->m_gen);
//            std::cout << "crop0 y: " << crop_min_y << ' ' << crop_max_y << '\n';//TODO
      }

      if (crop_min_x == crop_max_x)
        crop_x = crop_min_x;
      else
      {
        dis = std::uniform_int_distribution<>(crop_min_x, crop_max_x);
        crop_x = dis(this->m_gen);
//            std::cout << "crop0 x: " << crop_min_x << ' ' << crop_max_x << '\n';//TODO
      }

//          std::cout << "crop1: " << crop_x << ' ' << crop_x+crop_width << ' ' << crop_y << ' ' << crop_y+crop_height << '\n';//TODO


      if (dy > 0)
      {
        source_y1 = dy-crop_y;
        source_y2 = source_y1 + source_height - 1;
      }
      else if (dx > 0)
      {
        source_x1 = dx-crop_x;
        source_x2 = source_x1 + source_width - 1;
      }

      cv::Rect roi(crop_x, crop_y, crop_width, crop_height);
      cv_img = cv_img(roi);

      for (auto it=boxes.begin(); it!=boxes.end(); ++it)
      {
        it->x1 -= crop_x;
        it->x2 -= crop_x;
        it->y1 -= crop_y;
        it->y2 -= crop_y;
      }

//          std::cout << "crop2: " << cv_img.cols << 'x' << cv_img.rows << '\n';//TODO
    }
  }

  return skip;
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::Forward_cpu
(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  int batch_size = m_batch_size;

  Batch * batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded image.
  top[0]->Reshape(std::vector<int>({ batch_size, batch->data_.shape(1), batch->data_.shape(2), batch->data_.shape(3) }));

  // Reshape to image info.
  top[1]->Reshape(std::vector<int>({ batch_size, batch->info_.shape(1) }));

  int dData=0, dInfo=0;
  int cData=0, cInfo=0;

  std::vector<Dtype> boxes;

  for (int i=0; i<batch_size; ++i)
  {
    if (dData != 0)
      batch = prefetch_full_.pop("Data layer prefetch queue empty");

    cData = batch->data_.count();
    cInfo = batch->info_.count();

    // Copy the data
    caffe_copy( cData, batch->data_.cpu_data(), top[0]->mutable_cpu_data() + dData );
    DLOG(INFO) << "Prefetch copied";

    // Copy info.
    caffe_copy( cInfo, batch->info_.cpu_data(), top[1]->mutable_cpu_data() + dInfo );

    for (int j=0; j < batch->bboxes_.count(); j++)
      boxes.push_back( batch->bboxes_.cpu_data()[j] );

    if (i == batch_size-1)
    {
      // Reshape to bounding boxes.
      top[2]->Reshape(std::vector<int>({ (int)(boxes.size() / batch->bboxes_.shape(1)), batch->bboxes_.shape(1) }));
      // Copy bbox.
      caffe_copy( boxes.size(), boxes.data(), top[2]->mutable_cpu_data() );
    }
    else
    {
      dData += cData;
      dInfo += cInfo;
    }

    prefetch_free_.push(batch);

  }

}



#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageROIDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageROIDataLayer);
REGISTER_LAYER_CLASS(ImageROIData);

}  // namespace ultinous
}  // namespace caffe