#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <stdio.h>

#include "caffe/ultinous/proposal_layer.hpp"
#include "caffe/util/nms.hpp"


namespace caffe
{
namespace ultinous {

template<typename  Dtype>
struct TGreaterC
{
  __device__ bool operator()(const Dtype &a) const {
    return a > c;
  }
  Dtype c;
};

template<typename T, typename Dtype>
struct TGreater : public thrust::binary_function<T,T,bool>
{
  __device__ bool operator()(const T &a, const T &b) const {
    return m_values[a]>=m_values[b];
  }
  thrust::device_ptr<Dtype> m_values;
};

__device__
inline int bottom_offset(const int n, const int c,  const int h, const int w,
                             const int channels, const int height, const int width)
{
  return ((n * channels + c) * height + h) * width + w;
}

template <typename Dtype>
__global__ void proposal_kernel (const int n_threads, const int feat_stride, const int base_anchor_num, const int blob_h, const int blob_w,
                                 const Dtype img_width, const Dtype img_height, const Dtype min_height, const Dtype max_height,
                                 const Dtype* base_anchors, const Dtype* bottom_0, const Dtype* bottom_1,
                                 int* indexes, Dtype* proposals, Dtype* scores )
{
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    int ba_ind = index % base_anchor_num;
    int w_ind = (index / base_anchor_num) % blob_w;
    int h_ind = (index / base_anchor_num) / blob_w;

    Dtype prop_x1 = base_anchors[ba_ind*4+0] + w_ind * feat_stride;
    Dtype prop_y1 = base_anchors[ba_ind*4+1] + h_ind * feat_stride;
    Dtype prop_x2 = base_anchors[ba_ind*4+2] + w_ind * feat_stride;
    Dtype prop_y2 = base_anchors[ba_ind*4+3] + h_ind * feat_stride;

    Dtype prop_w = prop_x2 - prop_x1 + (Dtype)1;
    Dtype prop_h = prop_y2 - prop_y1 + (Dtype)1;
    Dtype prop_ctr_x = prop_x1 + (Dtype)0.5 * prop_w;
    Dtype prop_ctr_y = prop_y1 + (Dtype)0.5 * prop_h;

    Dtype pred_ctr_x = bottom_1[bottom_offset(0,ba_ind*4, h_ind, w_ind, base_anchor_num*4, blob_h, blob_w)] * prop_w + prop_ctr_x;
    Dtype pred_ctr_y = bottom_1[bottom_offset(0,ba_ind*4+1, h_ind, w_ind, base_anchor_num*4, blob_h, blob_w)]* prop_h + prop_ctr_y;
    Dtype pred_w = exp( bottom_1[bottom_offset(0,ba_ind*4+2, h_ind, w_ind, base_anchor_num*4, blob_h, blob_w)] ) * prop_w;
    Dtype pred_h = exp( bottom_1[bottom_offset(0,ba_ind*4+3, h_ind, w_ind, base_anchor_num*4, blob_h, blob_w)] ) * prop_h;

    bool is_pred_in_range = min_height < pred_h && pred_h <= max_height;

    proposals[index*4] = pred_ctr_x - (Dtype)0.5 * pred_w;
    proposals[index*4+1] = pred_ctr_y - (Dtype)0.5 * pred_h;
    proposals[index*4+2] = pred_ctr_x + (Dtype)0.5 * pred_w;
    proposals[index*4+3] = pred_ctr_y + (Dtype)0.5 * pred_h;

    bool is_pred_on_picture = proposals[index*4+2]<img_width && proposals[index*4]>0 && proposals[index*4+3]<img_height && proposals[index*4+1]>0;

    indexes[index] = index;
    scores[index] = is_pred_in_range && is_pred_on_picture ?
                    bottom_0[bottom_offset(0,base_anchor_num+ ba_ind, h_ind, w_ind, base_anchor_num*2, blob_h, blob_w )] : (Dtype) 0;
  }

}

template <typename Dtype>
__global__ void rawcopy_proposals_kernel(const int n_threads, const int* indexes, const Dtype* proposals, Dtype* top_0 )
{
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    int ind = indexes[index];
    top_0[index*5] = 0;
    top_0[index*5+1] = proposals[ind*4];
    top_0[index*5+2] = proposals[ind*4+1];
    top_0[index*5+3] = proposals[ind*4+2];
    top_0[index*5+4] = proposals[ind*4+3];

  }
}

template <typename Dtype>
__global__ void rawcopy_kernel(const int n_threads, const int* indexes, const Dtype* proposals , const Dtype* scores, Dtype* top_0, Dtype* top_1 )
{
  CUDA_KERNEL_LOOP(index, n_threads)
  {
    int ind = indexes[index];
    top_0[index*5] = 0;
    top_0[index*5+1] = proposals[ind*4];
    top_0[index*5+2] = proposals[ind*4+1];
    top_0[index*5+3] = proposals[ind*4+2];
    top_0[index*5+4] = proposals[ind*4+3];
    top_1[index] = scores[ind];
  }
}


template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

  if(bottom.size() == 4)
  {
    m_cut_threshold = bottom[3]->cpu_data()[0];
    m_min_size = bottom[3]->cpu_data()[1];
    m_max_size = bottom[3]->cpu_data()[2];
  }

  int count = m_proposals.count()/4; //contains the number of proposals

  const Dtype* bottom_0 = bottom[0]->gpu_data();
  const Dtype* bottom_1 = bottom[1]->gpu_data();
  const Dtype* base_anchors =  m_base_anchors.gpu_data();

  const Dtype img_width =  bottom[2]->cpu_data()[1];
  const Dtype img_height =  bottom[2]->cpu_data()[0];


  Dtype* proposals = m_proposals.mutable_gpu_data();
  Dtype* scores = m_scores.mutable_gpu_data();
  int* indexes = m_indexes.mutable_gpu_data();

  //Create all proposal in a kernel
  proposal_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
  m_feat_stride, m_base_anchors.shape(0), bottom[0]->shape(2),bottom[0]->shape(3),
  img_width, img_height, m_min_size, m_max_size,
  base_anchors, bottom_0, bottom_1, indexes, proposals, scores );
  CUDA_POST_KERNEL_CHECK;

  //Sort only indexes by the scores
  {
    thrust::device_ptr<Dtype> t_scores(m_scores.mutable_gpu_data());
    thrust::device_ptr<int> t_indexes(m_indexes.mutable_gpu_data());
    TGreater<int,Dtype> greater;
    greater.m_values=t_scores;
    thrust::stable_sort(t_indexes,t_indexes+count,greater);

    if(m_cut_threshold > 0)
    {
      TGreaterC<Dtype> greaterc;
      greaterc.c = m_cut_threshold;
      int temp = thrust::count_if(t_scores, t_scores + count, greaterc);
      count = std::min(count, temp);
    }
  }
  //If cut pre nms cut parameter is exists then  cut
  if(m_pre_nms_topN > 0)
    count = std::min(count, m_pre_nms_topN);

  count  = nms_gpu<Dtype>(count, indexes, scores, proposals, m_nms_thresh);

  if (m_post_nms_topN > 0)
    count = std::min(count,m_post_nms_topN);

  std::vector<int> shape(2,0);
  shape[0]=count;
  shape[1]=5;
  top[0]->Reshape(shape);
  if(top.size()==2)
  {
    shape[1]=1;
    top[1]->Reshape(shape);
  }

  if(count!=0)
  {
    if(top.size()==1)
    {
      rawcopy_proposals_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            indexes, proposals, top[0]->mutable_gpu_data());
    }
    else
    {
      rawcopy_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            indexes, proposals, scores, top[0]->mutable_gpu_data(), top[1]->mutable_gpu_data());
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

/*This layer not making backward computation*/
return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);

}
}

