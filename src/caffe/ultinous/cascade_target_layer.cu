#include "caffe/ultinous/cascade_target_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe
{
namespace ultinous
{

template <typename Dtype>
__global__
void target_find_kernel(
    int nthreads, Dtype* max_overlaps
    , const int proposal_num ,const Dtype* proposals
    , const int gt_boxes_num, const Dtype* gt_boxes
)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {

    if(index<proposal_num)
    {
      max_overlaps[index*2]=0;
      max_overlaps[index*2+1]= dev_IoU<Dtype>(proposals+(index*5+1),
                                              gt_boxes);
      for(int i=1; i<gt_boxes_num; ++i)
      {
        Dtype iou= dev_IoU<Dtype>(proposals+(index*5+1),
                                  gt_boxes+(i*5));
        if( max_overlaps[index*2+1] < iou )
        {
          max_overlaps[index*2] = i;
          max_overlaps[index*2+1] = iou;
        }
      }
    }
    else
    {
      int new_index = index - proposal_num;
      max_overlaps[index*2] = new_index;
      max_overlaps[index*2+1] = 1;
    }
  }
}


template <typename Dtype>
__global__
void cascade_target_kernel(
    const int nthreads, const int* indexes
    ,const int positive_num, const Dtype* maximum
    , const int proposal_num ,const Dtype* proposals
    , const int gt_boxes_num, const Dtype* gt_boxes,const Dtype* gt_labels
    , Dtype* targets, Dtype* target_labels
)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int i =  indexes[index];
    target_labels[index] = index < positive_num ? gt_labels[static_cast<int>(maximum[i*2])] : -1.0;

    targets[index * 5 ] = 0;
    targets[index * 5 + 1 ] = i < proposal_num ? proposals[i*5 + 1] : gt_boxes[(i-proposal_num)*5];
    targets[index * 5 + 2 ] = i < proposal_num ? proposals[i*5 + 2] : gt_boxes[(i-proposal_num)*5 + 1];
    targets[index * 5 + 3 ] = i < proposal_num ? proposals[i*5 + 3] : gt_boxes[(i-proposal_num)*5 + 2];
    targets[index * 5 + 4 ] = i < proposal_num ? proposals[i*5 + 4] : gt_boxes[(i-proposal_num)*5 + 3];
  }
}



template <typename Dtype>
void CascadeTargetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  const Dtype* proposals =  bottom[0]->gpu_data();
  const Dtype* gt_boxes = bottom[1]->gpu_data();
  const Dtype* gt_labels = bottom[2]->gpu_data();

  int proposals_num = bottom[0]->shape(0);
  int gt_boxes_num = bottom[1]->shape(0);

  Dtype* maximum = m_maximum.mutable_gpu_data();
  target_find_kernel<Dtype><<<CAFFE_GET_BLOCKS(m_maximum.shape(0)), CAFFE_CUDA_NUM_THREADS>>>(
    m_maximum.shape(0), maximum
    , proposals_num, proposals
    , gt_boxes_num, gt_boxes
  );
  CUDA_POST_KERNEL_CHECK;

  std::vector<int> positive_inds;
  std::vector<int> negative_inds;

  for(int i=0; i<m_maximum.shape(0); ++i)
  {
    if(m_maximum.cpu_data()[i*2+1] >= m_fg_treshold)
    {
      positive_inds.push_back(i);
    }
    else if(m_maximum.cpu_data()[i*2+1] < m_bg_treshold_hi &&
        m_maximum.cpu_data()[i*2+1] >= m_bg_treshold_lo)
    {
      negative_inds.push_back(i);
    }
  }

  std::random_shuffle(positive_inds.begin(), positive_inds.end());
  std::random_shuffle(negative_inds.begin(), negative_inds.end());
  std::size_t positive_num = static_cast<std::size_t>(round(m_fg_fraction * (Dtype)m_minibatch_size));

  positive_inds.resize(std::min(positive_num, positive_inds.size()));
  negative_inds.resize(std::min(m_minibatch_size-positive_num, negative_inds.size()));

  std::vector<int> shape;
  shape.push_back(positive_inds.size()+negative_inds.size());
  shape.push_back(1);
  m_keep_inds.Reshape(shape);

  int j = 0;
  for(std::size_t i = 0; i < positive_inds.size(); ++i, ++j )
    m_keep_inds.mutable_cpu_data()[j] = positive_inds[i];
  for(std::size_t i = 0; i < negative_inds.size(); ++i, ++j )
    m_keep_inds.mutable_cpu_data()[j] = negative_inds[i];

  shape[1]=5;
  top[0]->Reshape(shape);
  Dtype* rois = top[0]->mutable_gpu_data();
  shape[1]=1;
  top[1]->Reshape(shape);
  Dtype* labels = top[1]->mutable_gpu_data();

  cascade_target_kernel<Dtype><<<CAFFE_GET_BLOCKS(m_keep_inds.shape(0)), CAFFE_CUDA_NUM_THREADS >>>
  (
    m_keep_inds.shape(0), m_keep_inds.gpu_data()
    , positive_inds.size(), m_maximum.gpu_data()
    , proposals_num, proposals
    , gt_boxes_num, gt_boxes, gt_labels
    , top[0]->mutable_gpu_data(), top[1]->mutable_gpu_data()
  );

}

template <typename Dtype>
void CascadeTargetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
const vector<Blob<Dtype> *> &bottom)
{
  //no backward
}


INSTANTIATE_LAYER_GPU_FUNCS(CascadeTargetLayer);
}//ultinous
}//caffe