#include <vector>

#include "caffe/ultinous/image_roi_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void ImageROIDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());

  // Reshape to image info.
  top[1]->ReshapeLike(batch->info_);
  // Copy info.
  caffe_copy(batch->info_.count(), batch->info_.gpu_data(),
      top[1]->mutable_gpu_data());

  // Reshape to image info.
  top[2]->ReshapeLike(batch->bboxes_);
  // Copy bbox.
  caffe_copy(batch->bboxes_.count(), batch->bboxes_.gpu_data(),
      top[2]->mutable_gpu_data());

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(ImageROIDataLayer);

}  // namespace caffe
