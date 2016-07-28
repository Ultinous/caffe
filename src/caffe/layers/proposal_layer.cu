
#include "caffe/layers/proposal_layer.hpp"
#include <stdio.h>
#include <boost/graph/graph_concepts.hpp>
namespace caffe
{ 

__device__ int bottom_offset(const int n, const int c,  const int h, const int w,
                                   const int channels, const int height, const int width)
{
  return ((n * channels + c) * height + h) * width + w;
}

template <typename Dtype>
__device__ Dtype d_min(const Dtype a, const Dtype b)
{
  return a < b ? a : b; 
}

template <typename Dtype>
__device__ Dtype d_max(const Dtype a, const Dtype b)
{
  return a > b ? a : b; 
}

template <typename Dtype>
__global__ void create_full_proposals(const int nthreads, 
                                      const Dtype* const bottom_data_1 ,const int channels_1, const int height_1, const int width_1 ,
                                      const Dtype* const bottom_data_0, const int channels_0, const int height_0, const int width_0 ,
                                      Dtype* proposals, Dtype* anchors, Dtype* scores,
                                      const Dtype* const im_info, const int num_anchors , const int min_size)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int img_w = im_info[1];
    int img_h = im_info[0];
    int minimal_size = min_size * im_info[2];
    int ch_1 = index % (channels_1/4);
    int w_1 = (index / (channels_1/4)) % width_1;
    int h_1 = (index / (channels_1/4)) / width_1;
    int ch_0 = index % (channels_0-num_anchors) ;
    int w_0 = (index / (channels_0-num_anchors)) % width_0;
    int h_0 = (index / (channels_0-num_anchors)) / width_0;
    
    double width = anchors[index*4+2] - anchors[index*4] + 1;
    double height = anchors[index*4+3] - anchors[index*4+1] + 1;;
    double ctr_x = anchors[index*4] + 0.5 * width;
    double ctr_y = anchors[index*4+1] + 0.5 * height;
            
    double pred_ctr_x = bottom_data_1[bottom_offset(0,ch_1*4, h_1, w_1, channels_1, height_1, width_1)] * width + ctr_x;
    double pred_ctr_y = bottom_data_1[bottom_offset(0,ch_1*4+1, h_1, w_1, channels_1, height_1, width_1)]* height + ctr_y;
    double pred_w = exp( bottom_data_1[bottom_offset(0,ch_1*4+2, h_1, w_1, channels_1, height_1, width_1)] ) * width;
    double pred_h = exp( bottom_data_1[bottom_offset(0,ch_1*4+3, h_1, w_1, channels_1, height_1, width_1)] ) * height;
    
    proposals[index*4] = d_max<Dtype>(d_min<Dtype>(pred_ctr_x - 0.5 * pred_w, img_w-1),0); 
    proposals[index*4+1] = d_max<Dtype>(d_min<Dtype>(pred_ctr_y - 0.5 * pred_h, img_h-1),0); 
    proposals[index*4+2] = d_max<Dtype>(d_min<Dtype>(pred_ctr_x + 0.5 * pred_w,img_w-1),0);
    proposals[index*4+3] = d_max<Dtype>(d_min<Dtype>( pred_ctr_y + 0.5 * pred_h,img_h-1),0);
    
    
    scores[index] = 
    minimal_size <= (proposals[index*4+2]-proposals[index*4]+1) && minimal_size <= (proposals[index*4+3]-proposals[index*4+1]+1) ?
    bottom_data_0[bottom_offset(0,ch_0+num_anchors, h_0, w_0, channels_0, height_0, width_0 )] : 0;
    
  }
}

template <typename Dtype>
__global__ void bitonic_sort_base(const int nthreads, const int j, const int k,  int* indexes, const Dtype* scores)
{
  unsigned int i, ixj;
  int index_cache_1;
  int index_cache_2;
  CUDA_KERNEL_LOOP(index, nthreads)
  { 
    i = index;
    ixj = i^j;
    
    if ((ixj)>i) {
      index_cache_1 = indexes[i];
      index_cache_2 = indexes[ixj]; 
      
      if ((i&k)==0) {
        /* Sort ascending */        
        if (scores[index_cache_1]<scores[index_cache_2]) {
          /* exchange(i,ixj); */
          indexes[i] = index_cache_2;
          indexes[ixj] = index_cache_1;
        }
      }
      if ((i&k)!=0) {
        /* Sort descending */
        if (scores[index_cache_1]>scores[index_cache_2]) {
          /* exchange(i,ixj); */
          indexes[i] = index_cache_2;
          indexes[ixj] = index_cache_1;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void base_gpu_nms(const int nthreads, int* indexes, Dtype* scores, const Dtype* proposals, const Dtype threshold)
{
  int index_j;
  for(int j = 0; j < nthreads; ++j)
  {
    index_j = indexes[j];
    if(scores[index_j]!=0)
    {
      Dtype x1_j = proposals[index_j*4];
      Dtype x2_j = proposals[index_j*4+2];
      Dtype y1_j = proposals[index_j*4+1];
      Dtype y2_j = proposals[index_j*4+3];
      
      Dtype area_j = x2_j-x1_j+1 * y2_j-y1_j+1;
      CUDA_KERNEL_LOOP(index, nthreads)
      {
        int index_i = indexes[index];
        if(index>j && scores[index_i]!=0)
        {
          Dtype x1_i = proposals[index_i*4];
          Dtype y1_i = proposals[index_i*4+1];
          Dtype x2_i = proposals[index_i*4+2];
          Dtype y2_i = proposals[index_i*4+3];
          Dtype area_i = x2_i-x1_i+1 * y2_i-y1_i+1;
          
          Dtype  x1_inter = d_max<Dtype>(x1_j, x1_i);
          Dtype  y1_inter = d_max<Dtype>(y1_j, y1_i);
          Dtype  x2_inter = d_min<Dtype>(x2_j, x2_i);
          Dtype  y2_inter = d_min<Dtype>(y2_j, y2_i);
          
          Dtype inter = d_max<Dtype>(0.0, x2_inter - y1_inter + 1) * d_max<Dtype>(0.0, y2_inter - y1_inter + 1);
          if(threshold >= inter / (area_i + area_j - inter))
            scores[index_i]=0;
        }
      }         
    }
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void data_to_top(const int nthreads, const int* indexes, const Dtype* proposals, Dtype* top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int ind = indexes[index];
    top_data[index*5]=0;
    top_data[index*5+1] = proposals[ind*4];
    top_data[index*5+2] = proposals[ind*4+1];
    top_data[index*5+3] = proposals[ind*4+2];
    top_data[index*5+4] = proposals[ind*4+3];
  }
}



  
template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  if(bottom.size()==4)
  {
    m_pre_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,0)];
    m_post_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,1)];
    m_nms_thresh =bottom[3]->cpu_data()[bottom[3]->offset(0,2)];
    m_min_size = bottom[3]->cpu_data()[bottom[3]->offset(0,3)];
  }
  
  int count = m_anchors.count()/4;
  const Dtype* bottom_1 = bottom[1]->gpu_data();
  const Dtype* bottom_0 = bottom[0]->gpu_data();
  
  Dtype* proposals = m_proposals.mutable_gpu_data(); 
  Dtype* scores = m_scores.mutable_gpu_data();
  Dtype* anchors = m_anchors.mutable_gpu_data();
  int* indexes = m_indexes.mutable_gpu_data();
  
  create_full_proposals<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    bottom_1, bottom[1]->shape(1), bottom[1]->shape(2), bottom[1]->shape(3),
    bottom_0, bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    proposals, anchors, scores,
    bottom[2]->gpu_data(), m_num_anchors, m_min_size);
  CUDA_POST_KERNEL_CHECK;
  
  int j, k;
  for (k = 2; k <= count; k <<= 1) {
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_base<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, j, k, indexes, scores);
    }
  }
  CUDA_POST_KERNEL_CHECK;
  
  base_gpu_nms<Dtype><<<CAFFE_GET_BLOCKS(m_pre_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_pre_nms_topN, indexes, scores, proposals, m_nms_thresh);
  CUDA_POST_KERNEL_CHECK;
  
  for (k = 2; k <= m_pre_nms_topN; k <<= 1) {
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_base<<<CAFFE_GET_BLOCKS(m_pre_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_pre_nms_topN, j, k, indexes, scores);
    }
  }
  
  data_to_top<Dtype><<<CAFFE_GET_BLOCKS(m_post_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_post_nms_topN, indexes, proposals, top[0]->mutable_gpu_data() );
  CUDA_POST_KERNEL_CHECK;
  
  //TODO write scores to output blob
}


template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
   return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);
}