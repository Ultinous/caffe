#include <vector>

#include "caffe/ultinous/image_roi_data_layer.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void ImageROIDataLayer<Dtype>::Forward_gpu
  ( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top )
{
  int batch_size = this->m_batch_size;

  Batch * batch = prefetch_full_.pop("Data layer prefetch queue empty");

//  TODO
//  std::cout << "\nSTART\n\n";
//  std::cout << batch_size << "\n\n";
//
//  std::cout << "top shape:\n";
//  std::cout << "data:\n";
//  for ( auto e : top[0]->shape() )
//    std::cout << e << ' ';
//  std::cout << '\n';
//
//  std::cout << "info:\n";
//  for ( auto e : top[1]->shape() )
//    std::cout << e << ' ';
//  std::cout << '\n';
//
//  std::cout << "box:\n";
//  for ( auto e : top[2]->shape() )
//    std::cout << e << ' ';
//  std::cout << "\n\n";





  // Reshape to loaded image.
  top[0]->Reshape({ batch_size, batch->data_.shape(1), batch->data_.shape(2), batch->data_.shape(3) });

  // Reshape to image info.
  top[1]->Reshape({ batch_size, batch->info_.shape(1) });

  int dData=0, dInfo=0;
  int cData=0, cInfo=0;

  std::vector<Dtype> boxes;

  for (int i=0; i<batch_size; ++i)
  {
    if (dData != 0)
      batch = prefetch_full_.pop("Data layer prefetch queue empty");

//    TODO
//    std::cout << "batch shape:\n";
//    std::cout << "data:\n";
//    for ( auto e : batch->data_.shape() )
//      std::cout << e << ' ';
//    std::cout << '\n';
//
//    std::cout << "info:\n";
//    for ( auto e : batch->info_.shape() )
//      std::cout << e << ' ';
//    std::cout << '\n';
//
//    std::cout << "box:\n";
//    for ( auto e : batch->bboxes_.shape() )
//      std::cout << e << ' ';
//    std::cout << "\n\n";

    cData =   batch->data_.count();
    cInfo =   batch->info_.count();

    // Copy the data
    caffe_copy( cData, batch->data_.gpu_data(), top[0]->mutable_gpu_data() + dData );
    DLOG(INFO) << "Prefetch copied";

    // Copy info.
    caffe_copy( cInfo, batch->info_.gpu_data(), top[1]->mutable_gpu_data() + dInfo );

    for (int j=0; j < batch->bboxes_.count(); j++)
      boxes.push_back( batch->bboxes_.cpu_data()[j] );

    if (i == batch_size-1)
    {
      // Reshape to bounding boxes.
      top[2]->Reshape({ (int)(boxes.size() / batch->bboxes_.shape(1)), batch->bboxes_.shape(1) });
      // Copy bbox.
      caffe_copy( boxes.size(), boxes.data(), top[2]->mutable_cpu_data() );
    }
    else
    {
      dData += cData;
      dInfo += cInfo;
    }

    // Ensure the copy is synchronous wrt the host, so that the next batch isn't
    // copied in meanwhile.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    prefetch_free_.push(batch);

//    TODO
//    std::cout << "top shape:\n";
//    std::cout << "data:\n";
//    for ( auto e : top[0]->shape() )
//      std::cout << e << ' ';
//    std::cout << '\n';
//
//    std::cout << "info:\n";
//    for ( auto e : top[1]->shape() )
//      std::cout << e << ' ';
//    std::cout << '\n';
//
//    std::cout << "box:\n";
//    for ( auto e : top[2]->shape() )
//      std::cout << e << ' ';
//    std::cout << "\n\n";


  }

//  TODO
//  std::cout << "END\n\n";

//  TODO
//  LOG(INFO) << top[0]->shape_string();
//  LOG(INFO) << top[1]->shape_string();
//  LOG(INFO) << top[2]->shape_string();

}

INSTANTIATE_LAYER_GPU_FORWARD(ImageROIDataLayer);

}  // namespace ultinous
}  // namespace caffe
