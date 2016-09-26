#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
namespace ultinous
{

// data model of samples
struct Crop
{
  int x1, y1, x2, y2, cls;
};
typedef std::vector<Crop> Crops;

struct Sample
{
  std::string image_file;
  Crops crops;
};
typedef std::vector<Sample> Samples;

/**
 * @brief Provides data to the Net from image files and annotations. Main use case: train object detection models.
 *
 * Annotation file is a tab separated file. Each line contains one image with all the crops. 
 * Format of one line:
 *   image_file_name x1 y1 x2 y2 class x1 y1 x2 y2 class x1 y1 x2 y2 class ...
 * 
 * Example:
 *   a.jpg 12 14 90 98 0
 *   b.jpg
 *   c.jpg 130 150 210 230 0 230 250 290 310
 *
 * Output blobls:
 *   - data: full images (as read from the images files)
 *   - image info: TODO
 *   - ground truth bounding boxes: (x1, y1, x2, y2, cls)
 */
template <typename Dtype>
class ImageCropDataLayer : public BasePrefetchingDataLayer<Dtype>
{
public:
  explicit ImageCropDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageCropDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const
  {
    return "ImageCropData";
  }
  virtual inline int ExactNumBottomBlobs() const
  {
    return 0;
  }
  virtual inline int ExactNumTopBlobs() const
  {
    return 3;   // data, im_info, gt_boxes
  }

protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

protected:
  Samples samples;
  int lines_id_;
};


}  // namespace ultinous
}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
