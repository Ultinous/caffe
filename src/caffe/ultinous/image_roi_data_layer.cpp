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
  const auto& image_data_param = this->layer_param_.image_data_param();

  int new_height = image_data_param.new_height();
  int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const string &root_folder = image_data_param.root_folder();
  const string& source = image_data_param.source();
  this->batch_size_ = image_data_param.batch_size();

  const auto& transform_param = this->layer_param_.transform_param();
  if ( transform_param.mean_value_size() > 0 )
    for (int c = 0; c < transform_param.mean_value_size(); ++c)
      this->mean_values_.push_back(static_cast<double>(transform_param.mean_value(c)));

  const auto& image_roi_data_param = this->layer_param_.image_roi_data_param();

  const int inImgNum = image_roi_data_param.in_img_num();
  bool randomScale = (image_roi_data_param.has_rnd_scale_min() || image_roi_data_param.has_rnd_scale_max());
  bool randomCrop  = ( image_roi_data_param.has_crop_height() && image_roi_data_param.has_crop_width() );
  if (randomCrop)
  {
    new_height = image_roi_data_param.crop_height();
    new_width = image_roi_data_param.crop_width();
  }

  CHECK( !(batch_size_>1 && !randomCrop) )
    << "Batch size > 1 requires random cropping.";

  CHECK( !(randomCrop && randomScale) )
    << "Can not randomly crop and resize at the same time.";

  CHECK( (new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0) )
    << "Current implementation requires new_height and new_width to be set at the same time.";

  // Read the annotation file
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
    RotationMatrix rot_m;
    //LOG(INFO) << "Reading source file ==================================================";
    while( iss >> bbox.x1 >> bbox.y1 >> bbox.x2 >> bbox.y2 >> tmp >> rot_m.x11 >> rot_m.x12 >> rot_m.x13 >> rot_m.x21 >> rot_m.x22 >> rot_m.x23 >> rot_m.x31 >> rot_m.x32 >> rot_m.x33)
    {
      sample.bboxes.push_back(bbox);
      if(tmp == 1){
          //There is a rotation available for this bbox

          //iss >> rot_m.x11 >> rot_m.x12 >> rot_m.x13 >> rot_m.x21 >> rot_m.x22 >> rot_m.x23;
          //iss >> rot_m.x31 >> rot_m.x32 >> rot_m.x33;
          sample.rot_matrixes.push_back(rot_m);
	  //LOG(INFO) << "Line: " << line;
	  //LOG(INFO) << "Reading rot mtx: " << rot_m.x11 << " " << rot_m.x12<< " " << rot_m.x13;
      }else{
          //There is no rotation matrix for thib bbox
          sample.rot_matrixes.push_back({});
	  LOG(INFO) << "No rot mtx found";
      }
    if( !sample.bboxes.empty() )
      samples.push_back( sample );
  }
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
    unsigned int skip = caffe_rng_rand() % image_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(samples.size(), skip) << "Not enough points to skip";
    sample_id_ = skip;
  }

  cv::Mat cv_img = ReadImageToCVMat(root_folder + samples[0].image_files[0], new_height, new_width, is_color);

  CHECK(image_roi_data_param.pad()>=1);
  if(image_roi_data_param.pad()>1)
  {
    int pad_h = ( image_roi_data_param.pad() - ( cv_img.rows % image_roi_data_param.pad() ) ) % image_roi_data_param.pad();
    int pad_w = ( image_roi_data_param.pad() - ( cv_img.cols % image_roi_data_param.pad() ) ) % image_roi_data_param.pad();
    if( pad_h != 0 || pad_w != 0 )
      copyMakeBorderWrapper(cv_img, cv_img, 0, pad_h, 0, pad_w, {0,0,0});
  }

  CHECK(cv_img.data) << "Could not load " << samples[0].image_files[0];

  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size_;
  top_shape[1] = cv_img.channels() * inImgNum;
  top[0]->Reshape(top_shape);

  vector<int> info_shape(2);
  info_shape[0] = batch_size_;
  info_shape[1] = 7;
  top[1]->Reshape(info_shape);

  vector<int> bboxes_shape(2);
  bboxes_shape[0] = 1;
  bboxes_shape[1] = 5;
  top[2]->Reshape(bboxes_shape);

  vector<int> rot_m_shape(2);
  rot_m_shape[0] = 1;
  rot_m_shape[1] = 9;
  top[3]->Reshape(rot_m_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
  {
    this->prefetch_[i].data_.Reshape(top_shape);
    this->prefetch_[i].info_.Reshape(info_shape);
    this->prefetch_[i].bboxes_.Reshape(bboxes_shape);
    this->prefetch_[i].rot_mtxs_.Reshape(rot_m_shape);
  }
}

template <typename Dtype>
void ImageROIDataLayer<Dtype>::ShuffleImages()
{
  auto prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
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

  const auto& image_data_param = this->layer_param_.image_data_param();

  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const string &root_folder = image_data_param.root_folder();

  const auto& image_roi_data_param = this->layer_param_.image_roi_data_param();

  bool randomScale = (image_roi_data_param.has_rnd_scale_min() || image_roi_data_param.has_rnd_scale_max());
  bool randomCrop = (image_roi_data_param.has_crop_height() && image_roi_data_param.has_crop_width());
  int smallerDimensionSize = image_roi_data_param.smallerdimensionsize();
  int maxSize = image_roi_data_param.maxsize();
  int inImgNum = image_roi_data_param.in_img_num();
  int crop_height=0, crop_width=0;
  if (randomCrop)
  {
    crop_height = image_roi_data_param.crop_height();
    crop_width  = image_roi_data_param.crop_width();
  }

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;

  const int samples_size = samples.size();
  int batch_index = 0;
  BBoxes accumulatedBoxes;
  RotationMatrixList finalRotMatrixes;
  while ( batch_index < batch_size_ )
  {
    CPUTimer timer;

    // Copy bounding boxes
    BBoxes boxes = samples[sample_id_].bboxes;

    // Copy rotation matrixes
    RotationMatrixList rot_matrixes = samples[sample_id_].rot_matrixes;
    //LOG(INFO) << "Sample id: " << sample_id_ <<  " " << rot_matrixes[0].x11 << " " << rot_matrixes[1].x11 << " " << rot_matrixes[2].x11;

    // Load image
    timer.Start();
    CHECK_GT(samples_size, sample_id_);

    cv::Mat cv_img = readMultiChannelImage(inImgNum, new_height, new_width, is_color, root_folder);

//    TODO
//    cv::Mat cv_tmp = cv_img.clone();
//    for (auto it=samples[sample_id_].bboxes.begin(); it!=samples[sample_id_].bboxes.end(); ++it)
//      cv::rectangle(cv_tmp, cv::Point(it->x1, it->y1), cv::Point(it->x2, it->y2), cv::Scalar(0, 255, 0));

    read_time += timer.MicroSeconds();

    bool skip = false;
    int source_x1=0, source_y1=0, source_x2=0, source_y2=0;
    if (randomCrop)
      skip = doRandomCrop(
        cv_img, boxes,
        source_x1, source_x2, source_y1, source_y2,
        crop_height, crop_width
      );

    if (!skip)
    {
      float scale = 1.0f;

      if ( !randomCrop ) {
        // resize
        if (randomScale) {
          float scale_min = image_roi_data_param.has_rnd_scale_min() ? image_roi_data_param.rnd_scale_min() : 1;
          float scale_max = image_roi_data_param.has_rnd_scale_max() ? image_roi_data_param.rnd_scale_max() : 1;
          CHECK(scale_max - scale_min > 0);

          if ((caffe_rng_rand() % 2) == 1)
            caffe_rng_uniform(1, scale_min, scale_max, &scale);
        }

        if (smallerDimensionSize > 0) {
          float scale1 = static_cast<float>(smallerDimensionSize)
                         / std::min(cv_img.rows, cv_img.cols);
          float scale2 = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));
          scale = std::min(scale1, scale2);

        } else if (std::max(cv_img.rows, cv_img.cols) > maxSize)
          scale = static_cast<float>(maxSize) / static_cast<float>(std::max(cv_img.rows, cv_img.cols));

        if (scale != 1.0f) {
          cv::Mat cv_resized;
          cv::resize(cv_img, cv_resized, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);

          cv_img = cv_resized;
        }

        CHECK(image_roi_data_param.pad() >= 1);
        if (image_roi_data_param.pad() > 1) {
          int pad_h =
              (image_roi_data_param.pad() - (cv_img.rows % image_roi_data_param.pad())) % image_roi_data_param.pad();
          int pad_w =
              (image_roi_data_param.pad() - (cv_img.cols % image_roi_data_param.pad())) % image_roi_data_param.pad();
          if (pad_h != 0 || pad_w != 0)
            copyMakeBorderWrapper(cv_img, cv_img, 0, pad_h, 0, pad_w, mean_values_);
        }

        source_x2 = cv_img.cols - 1;
        source_y2 = cv_img.rows - 1;
      }

      bool mirror = image_roi_data_param.mirror() && ((caffe_rng_rand() % 2) == 1);

      if (mirror)
      {
        cv::Mat cv_flipped;
        cv::flip(cv_img, cv_flipped, 1);
        cv_img = cv_flipped;
        Dtype temp = source_x1;
        source_x1 = cv_img.cols - source_x2 - 1;
        source_x2 = cv_img.cols - temp - 1;
      }

      BBoxes finalBoxes;

      for (size_t i = 0; i < boxes.size(); i++)
      {
        BBox bbox(boxes[i]);
        bbox.x1 *= scale;
        bbox.y1 *= scale;
        bbox.x2 *= scale;
        bbox.y2 *= scale;

        if (bbox.x1 >= bbox.x2 || bbox.y1 >= bbox.y2)
          continue;

        int bw=ceil((bbox.x2-bbox.x1+1)*0.6), bh=ceil((bbox.y2-bbox.y1+1)*0.6);
        if (bbox.x1 < -bw + source_x1)
          continue;
        if (bbox.y1 < -bh + source_y1)
          continue;
        if (bbox.x2 > source_x2 + bw)
          continue;
        if (bbox.y2 > source_y2 + bh)
          continue;

        if (mirror)
        {
          Dtype temp = bbox.x1;
          bbox.x1 = cv_img.cols - bbox.x2 - 1;
          bbox.x2 = cv_img.cols - temp - 1;
        }


          if (mirror) {

                  rot_matrixes[i].x12 =  rot_matrixes[i].x12 *-1.0;
                  rot_matrixes[i].x13  = rot_matrixes[i].x13* -1.0;

                  rot_matrixes[i].x21 = rot_matrixes[i].x21* -1.0;

                  rot_matrixes[i].x31 = rot_matrixes[i].x31* -1.0;


          }

        finalBoxes.push_back(bbox);
	//LOG(INFO) << "Index insert: " << i;
        finalRotMatrixes.push_back(rot_matrixes[i]);
	//LOG(INFO) << "Push mtx to finalRotMtces " << rot_matrixes[i].x11<< " " << rot_matrixes[i].x12 << " "<< rot_matrixes[i].x13;
      }

      if ( !finalBoxes.empty() )
      {
        vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
        if (batch_index == 0)
        {
          // Reshape batch.
          top_shape[0] = batch_size_;
          batch->data_.Reshape(top_shape);
          top_shape[0] = 1;
        }
        this->transformed_data_.Reshape(top_shape);

        Dtype *prefetch_data = batch->data_.mutable_cpu_data();

        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset = batch->data_.offset(batch_index);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        trans_time += timer.MicroSeconds();

        Dtype *prefetch_info = batch->info_.mutable_cpu_data();
        prefetch_info[batch->info_.offset( batch_index, 0 )] = static_cast<Dtype>(cv_img.rows);
        prefetch_info[batch->info_.offset( batch_index, 1 )] = static_cast<Dtype>(cv_img.cols);
        prefetch_info[batch->info_.offset( batch_index, 2 )] = static_cast<Dtype>(source_x1);
        prefetch_info[batch->info_.offset( batch_index, 3 )] = static_cast<Dtype>(source_y1);
        prefetch_info[batch->info_.offset( batch_index, 4 )] = static_cast<Dtype>(source_x2);
        prefetch_info[batch->info_.offset( batch_index, 5 )] = static_cast<Dtype>(source_y2);
        prefetch_info[batch->info_.offset( batch_index, 6 )] = static_cast<Dtype>(finalBoxes.size());
        prefetch_info[batch->info_.offset( batch_index, 7)] = static_cast<Dtype>(finalRotMatrixes.size());

        std::copy(finalBoxes.begin(), finalBoxes.end(), std::back_inserter(accumulatedBoxes));


//        TODO
//        for (int bboxIx = 0; bboxIx < finalBoxes.size(); ++bboxIx)
//        {
//          BBox bbox = finalBoxes[bboxIx];
//          Dtype x1 = static_cast<Dtype>(bbox.x1);
//          Dtype y1 = static_cast<Dtype>(bbox.y1);
//          Dtype x2 = static_cast<Dtype>(bbox.x2);
//          Dtype y2 = static_cast<Dtype>(bbox.y2);
//
//          cv::rectangle(cv_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));//TODO
//        }
//        cv::imwrite("debug/"+name+"_"+batch_index+"_crop.jpg", cv_img);//TODO
//        cv::imwrite("debug/"+name+_+batch_index+".jpg", cv_tmp);//TODO

        batch_index += 1;
      }
    } // if (!skip)

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
  } // loop: while ( batch_index < batch_size_ )

  vector<int> bboxes_shape(2);
  bboxes_shape[0] = accumulatedBoxes.size();
  bboxes_shape[1] = 5;
  batch->bboxes_.Reshape(bboxes_shape);
  Dtype *prefetch_bboxes = batch->bboxes_.mutable_cpu_data();
  for (int bboxIx = 0; bboxIx < accumulatedBoxes.size(); ++bboxIx)
  {
    BBox bbox = accumulatedBoxes[bboxIx];
    auto x1 = static_cast<Dtype>(bbox.x1);
    auto y1 = static_cast<Dtype>(bbox.y1);
    auto x2 = static_cast<Dtype>(bbox.x2);
    auto y2 = static_cast<Dtype>(bbox.y2);

    prefetch_bboxes[batch->bboxes_.offset( bboxIx, 0 )] = x1;
    prefetch_bboxes[batch->bboxes_.offset( bboxIx, 1 )] = y1;
    prefetch_bboxes[batch->bboxes_.offset( bboxIx, 2 )] = x2;
    prefetch_bboxes[batch->bboxes_.offset( bboxIx, 3 )] = y2;
    prefetch_bboxes[batch->bboxes_.offset( bboxIx, 4 )] = 1;
  }

  vector<int> rot_mtx_shape(2);
  rot_mtx_shape[0] = finalRotMatrixes.size();
  rot_mtx_shape[1] = 9;
  batch->rot_mtxs_.Reshape(rot_mtx_shape);
  Dtype *prefetched_rot_matrixes = batch->rot_mtxs_.mutable_cpu_data();
  for(int rotMtxIx = 0; rotMtxIx < finalRotMatrixes.size(); ++rotMtxIx)
  {
      RotationMatrix r_mtx = finalRotMatrixes[rotMtxIx];
      auto x11 = static_cast<Dtype>(r_mtx.x11);
      auto x12 = static_cast<Dtype>(r_mtx.x12);
      auto x13 = static_cast<Dtype>(r_mtx.x13);
      auto x21 = static_cast<Dtype>(r_mtx.x21);
      auto x22 = static_cast<Dtype>(r_mtx.x22);
      auto x23 = static_cast<Dtype>(r_mtx.x23);
      auto x31 = static_cast<Dtype>(r_mtx.x31);
      auto x32 = static_cast<Dtype>(r_mtx.x32);
      auto x33 = static_cast<Dtype>(r_mtx.x33);

      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 0)] = x11;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 1)] = x12;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 2)] = x13;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 3)] = x21;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 4)] = x22;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 5)] = x23;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 6)] = x31;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 7)] = x32;
      prefetched_rot_matrixes[batch->rot_mtxs_.offset(rotMtxIx, 8)] = x33;


      //LOG(INFO)<< "rotMtxIx: " << rotMtxIx << "first col:  " << x11 << " " << x12 <<  " " << x13;

  }

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
{
  for (int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
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
    prefetch_[i].rot_mtxs_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU)
  {
    for (int i = 0; i < PREFETCH_COUNT; ++i)
    {
      prefetch_[i].data_.mutable_gpu_data();
      prefetch_[i].info_.mutable_gpu_data();
      prefetch_[i].bboxes_.mutable_gpu_data();
      prefetch_[i].rot_mtxs_.mutable_gpu_data();
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

void copyMakeBorderWrapper(const cv::Mat &src, cv::Mat &dst,
                           int top, int bottom, int left, int right,
                           const std::vector<double> &color)
{
  const int height=src.rows+bottom+top, width=src.cols+left+right, channels=src.channels(),
                   colorSize=static_cast<int>(color.size());

  CHECK(channels%colorSize == 0) << "Supplied image is incompatible with supplied scalar.";
  if (channels <= 4 && channels == colorSize)
  {
    std::vector<double> paddedColor(4,0);
    std::copy(color.begin(), color.end(), paddedColor.begin());
    const cv::Scalar scalar(cv::Vec<double,4>(paddedColor.data()));

    cv::copyMakeBorder(src, dst, top, bottom, left, right, cv::BORDER_CONSTANT, scalar);
  }
  else
  {
    cv::Mat tmp = cv::Mat(height,width,CV_8UC(channels));
    size_t colorIndex = 0;
    for (int channel=0; channel<channels; ++channel)
    {
      for (int y=0; y<height; y++)
      {
        // get pointer to the first byte to be changed in this row
        unsigned char *p_row = tmp.ptr(y) + channel;
        unsigned char *row_end = p_row + width*channels;
        for (; p_row != row_end; p_row += channels)
          *p_row = color[colorIndex];
      }

      colorIndex = (colorIndex==colorSize ? 0 : colorIndex+1);
    }

    src.copyTo(tmp(cv::Rect(left,top,src.cols,src.rows)));
    dst = tmp;
  }
}

template <typename Dtype>
inline cv::Mat ImageROIDataLayer<Dtype>::readMultiChannelImage(int inImgNum, int new_height, int new_width, bool is_color, const string& root_folder)
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
  }

  cv::Mat cv_img;
  cv::merge(slices, cv_img);

  return cv_img;
}

template <typename Dtype>
inline std::map<Gain, int> ImageROIDataLayer<Dtype>::getGain(
  std::vector<size_t>& excludeIndices,
  std::vector<int> vx1, std::vector<int> vy1, std::vector<int> vx2, std::vector<int> vy2
){
  for (auto it=excludeIndices.rbegin(); excludeIndices.rend()!=it; ++it)
  {
    vx1.erase(vx1.begin() + *it);
    vx2.erase(vx2.begin() + *it);
    vy1.erase(vy1.begin() + *it);
    vy2.erase(vy2.begin() + *it);
  }

  int x1 = *std::min_element(vx1.begin(),vx1.end());
  int y1 = *std::min_element(vy1.begin(),vy1.end());
  int x2 = *std::max_element(vx2.begin(),vx2.end());
  int y2 = *std::max_element(vy2.begin(),vy2.end());

  std::map<Gain, int> gain = {{Gain::vertical, y2-y1+1}, {Gain::horizontal, x2-x1+1}};

  return gain;
}

template <typename Dtype>
inline bool ImageROIDataLayer<Dtype>::doRandomCrop(
  cv::Mat& cv_img, BBoxes& boxes,
  int& source_x1, int& source_x2, int& source_y1, int& source_y2,
  const int crop_height, const int crop_width
){
  bool skip = true;

  int crop_x,crop_y,pad_x=0,pad_y=0;

  int source_height = cv_img.rows;
  int source_width = cv_img.cols;

  int dy = crop_height - source_height;
  int dx = crop_width - source_width;
  if (dy >= 0 && dx >= 0)
  {
    pad_x = ( dx==0 ? 0 : caffe_rng_rand()%(dx + 1) );

    pad_y = ( dy==0 ? 0 : caffe_rng_rand()%(dy + 1) );

    source_x1 = pad_x;
    source_y1 = pad_y;
    source_x2 = pad_x + source_width  - 1;
    source_y2 = pad_y + source_height - 1;

    if (dx != 0 || dy != 0)
    {
      copyMakeBorderWrapper(cv_img, cv_img, pad_y, dy - pad_y, pad_x, dx - pad_x, mean_values_);

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
      copyMakeBorderWrapper(cv_img, cv_img, pad_y, pad_y, 0, 0, mean_values_);
    }
    else if (dx > 0)
    {
      pad_x = dx;
      copyMakeBorderWrapper(cv_img, cv_img, 0, 0, pad_x, pad_x, mean_values_);
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
      if (vx1.empty())
        break;

      x1 = *std::min_element(vx1.begin(),vx1.end());
      y1 = *std::min_element(vy1.begin(),vy1.end());
      x2 = *std::max_element(vx2.begin(),vx2.end());
      y2 = *std::max_element(vy2.begin(),vy2.end());

      if (x1 >= x2 || y1 >= y2)
        break;

      int h = y2-y1+1, w = x2-x1+1;

      bool vertical   = h > crop_height;
      bool horizontal = w > crop_width;

      if (vertical || horizontal)
      {
        std::set<size_t> indicesToRemove;
        std::vector<Dir> satisfiers;
        std::map<Dir, std::vector<size_t>> directions;
        std::map<Dir, int> gains;

        for (int i=0; i<4; ++i) // iterate n,e,s,w
          directions.emplace(std::make_pair(Dir(i), std::vector<size_t>()));

        for (size_t i = 0; i < vx1.size(); ++i)
        {
          if (vx1[i] == x1)
            directions[Dir::w].push_back(i);
          if (vy1[i] == y1)
            directions[Dir::n].push_back(i);
          if (vx2[i] == x2)
            directions[Dir::e].push_back(i);
          if (vy2[i] == y2)
            directions[Dir::s].push_back(i);
        }

        if (vertical && horizontal)
        {
          for (int i=4; i<8; ++i) // iterate ne,se,sw,nw
            directions.emplace(std::make_pair(Dir(i), std::vector<size_t>()));

          std::vector<size_t> u;
          std::set_union(
            directions[Dir::n].begin(), directions[Dir::n].end(), directions[Dir::e].begin(), directions[Dir::e].end(),
            std::back_inserter(u));
          directions[Dir::ne] = u;

          u.clear();
          std::set_union(
            directions[Dir::s].begin(), directions[Dir::s].end(), directions[Dir::e].begin(), directions[Dir::e].end(),
            std::back_inserter(u));
          directions[Dir::se] = u;

          u.clear();
          std::set_union(
            directions[Dir::s].begin(), directions[Dir::s].end(), directions[Dir::w].begin(), directions[Dir::w].end(),
            std::back_inserter(u));
          directions[Dir::sw] = u;

          u.clear();
          std::set_union(
            directions[Dir::n].begin(), directions[Dir::n].end(), directions[Dir::w].begin(), directions[Dir::w].end(),
            std::back_inserter(u));
          directions[Dir::nw] = u;

          for (int i=0; i<4; ++i) // iterate n,e,s,w
            directions.erase(Dir(i));

          for (auto direction : directions)
          {
            auto gain = getGain(direction.second,vx1,vy1,vx2,vy2);
            gains[direction.first] = gain[Gain::vertical] * gain[Gain::horizontal];
            if ( gain[Gain::vertical] <= crop_height && gain[Gain::horizontal] <= crop_width )
              satisfiers.push_back(direction.first);
          }
        }
        else if (vertical)
        {
          directions.erase(Dir::w);
          directions.erase(Dir::e);
          for (auto direction : directions)
          {
            gains[direction.first] = getGain(direction.second,vx1,vy1,vx2,vy2)[Gain::vertical];
            if ( gains[direction.first] <= crop_height )
              satisfiers.push_back(direction.first);
          }
        }
        else if (horizontal)
        {
          directions.erase(Dir::n);
          directions.erase(Dir::s);
          for (auto direction : directions)
          {
            gains[direction.first] = getGain(direction.second,vx1,vy1,vx2,vy2)[Gain::horizontal];
            if ( gains[direction.first] <= crop_width )
              satisfiers.push_back(direction.first);
          }
        }

        if ( !satisfiers.empty() )
        {
          Dir key;
          if (satisfiers.size() == 1)
            key = satisfiers[0];
          else
            key = satisfiers[ caffe_rng_rand() % satisfiers.size() ];

          indicesToRemove.insert(directions[key].begin(), directions[key].end());
        }
        else
        {
          const int minGain = std::min_element(
              std::begin(gains), std::end(gains),
              [] (const std::pair< Dir, int > & p1, const std::pair< Dir, int > & p2) {
                return p1.second < p2.second;
              })->second;

          std::vector<Dir> minKeys;
          for (auto gain : gains)
            if (gain.second == minGain)
              minKeys.push_back(gain.first);
          Dir minKey = minKeys[ caffe_rng_rand() % minKeys.size() ];

          indicesToRemove.insert(directions[minKey].begin(), directions[minKey].end());
        }

        for (auto it=indicesToRemove.rbegin(); indicesToRemove.rend()!=it; ++it)
        {
          vx1.erase(vx1.begin() + *it);
          vx2.erase(vx2.begin() + *it);
          vy1.erase(vy1.begin() + *it);
          vy2.erase(vy2.begin() + *it);
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
        crop_y = crop_min_y + (caffe_rng_rand() % (crop_max_y-crop_min_y+1));

      if (crop_min_x == crop_max_x)
        crop_x = crop_min_x;
      else
        crop_x = crop_min_x + (caffe_rng_rand() % (crop_max_x-crop_min_x+1));

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
    }
  }

  return skip;
}

  template <typename Dtype>
  void ImageROIDataLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");

    // Reshape to loaded image.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
               top[0]->mutable_cpu_data());
    DLOG(INFO) << "Prefetch copied";

    // Reshape to image info.
    top[1]->ReshapeLike(batch->info_);
    // Copy info.
    caffe_copy(batch->info_.count(), batch->info_.cpu_data(),
               top[1]->mutable_cpu_data());

    // Reshape to image info.
    top[2]->ReshapeLike(batch->bboxes_);
    // Copy bbox.
    caffe_copy(batch->bboxes_.count(), batch->bboxes_.cpu_data(),
               top[2]->mutable_cpu_data());

    //Reshape rot matrixes
    top[3]->ReshapeLike(batch->rot_mtxs_);
    // Copy rot_mtx
    caffe_copy(batch->rot_mtxs_.count(), batch->rot_mtxs_.cpu_data(),
            top[3]->mutable_cpu_data());

    prefetch_free_.push(batch);
  }



#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageROIDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageROIDataLayer);
REGISTER_LAYER_CLASS(ImageROIData);

}  // namespace ultinous
}  // namespace caffe
